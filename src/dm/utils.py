from functools import wraps
import importlib.util
import os
from pathlib import Path
import re

from arcgis.env import active_gis
from arcgis.gis import GIS
from arcgis.features import FeatureSet, FeatureLayer, GeoAccessor
from arcgis.geometry import Geometry
import pandas as pd

arcpy_avail = True if importlib.util.find_spec("arcpy") else False

if arcpy_avail:
    from ._registry import get_ba_usa_key_str, get_child_key_strs


class Environment:

    def __init__(self):
        self._installed_lst = []
        self._not_installed_lst = []
        self._arcpy_extensions = []
        self._extension_lst = ['3D', 'Datareviewer', 'DataInteroperability', 'Airports', 'Aeronautical', 'Bathymetry',
                               'Nautical', 'GeoStats', 'Network', 'Spatial', 'Schematics', 'Tracking', 'JTX', 'ArcScan',
                               'Business', 'Defense', 'Foundation', 'Highways', 'StreetMap']
        self._arcpy_avail = None
        self._local_ba = None

    @property
    def arcpy_avail(self):
        if self._arcpy_avail is None:
            self._arcpy_avail = True if importlib.util.find_spec("arcpy") else False
        return self._arcpy_avail

    @property
    def local_business_analyst(self):
        if self.arcpy_avail and self._local_ba is None:
            self._local_ba = True if arcpy.CheckExtension('Business') and self.arcpy_avail else False
        elif self._local_ba is None:
            self._local_ba = False
        return self._local_ba

    def has_package(self, package_name):

        if package_name in self._installed_lst:
            return True

        elif package_name in self._not_installed_lst:
            return False

        else:
            installed = True if importlib.util.find_spec(package_name) else False
            if installed:
                self._installed_lst.append(package_name)
            else:
                self._not_installed_lst.append(package_name)

        return installed

    @property
    def arcpy_extensions(self):

        if not self.arcpy_avail:
            raise Exception('Since not in an environment with ArcPy available, it is not possible to check available '
                            'extensions.')

        elif len(self._arcpy_extensions) == 0:
            import arcpy
            for extension in self._extension_lst:
                if arcpy.CheckExtension(extension):
                    self._arcpy_extensions.append(extension)

        return self._arcpy_extensions

    def arcpy_checkout_extension(self, extension):

        if self.has_package('arcpy') and extension in self.arcpy_extensions:
            import arcpy
            arcpy.CheckOutExtension(extension)

        else:
            raise Exception(f'Cannot check out {extension}. It either is not licensed, not installed, or you are not '
                            f'using the correct reference ({", ".join(self._extension_lst)}).')

        return True


# expose instantiated environment object to namespace
env = Environment()

# import arcpy if possible
if env.arcpy_avail:
    import arcpy


def local_vs_gis(fn):
    """Decorator to facilitate bridging between local and remote resources."""
    # get the method geographic_level - this will be used to redirect the function call
    fn_name = fn.__name__

    @wraps(fn)
    def wrapped(self, *args, **kwargs):

        # if performing analysis locally, try to access the function locally, but if not implemented, catch the error
        if self.source == 'local':
            try:
                fn_to_call = getattr(self, f'_{fn_name}_local')
            except AttributeError:
                raise AttributeError(f"'{fn_name}' not available using 'local' as the source.")

        # now, if performing analysis using a Web GIS, then access the function referencing remote resources
        elif isinstance(self.source, GIS):
            try:
                fn_to_call = getattr(self, f'_{fn_name}_gis')
            except AttributeError:
                raise AttributeError(f"'{fn_name}' not available using a Web GIS as the source.")

        # if another source, we don't plan on doing that any time soon
        else:
            raise AttributeError(f"'{self.source}' is not a recognized demographic modeling source.")

        return fn_to_call(*args, **kwargs)

    return wrapped


def set_source(in_source: [str, GIS] = None) -> [str, GIS]:
    """
    Helper function to check source input. The source can be set explicitly, but if nothing is provided, it
    assumes the order of local first and then a Web GIS. Along the way, it also checks to see if a GIS object
    instance is available in the current session."""

    # if string input is provided, ensure setting to local and lowercase
    if isinstance(in_source, str):
        if in_source.lower() != 'local':
            raise Exception(f'Source must be either "local" or a Web GIS instance, not {in_source}.')

        # TODO: add check for local business analyst data
        elif in_source.lower() == 'local' and not env.local_business_analyst:
            raise Exception(f'If using local source, the Business Analyst extension must be available')

        elif in_source.lower() == 'local':
            source = 'local'

    # if nothing provided, default to local if arcpy is available, and remote if arcpy not available
    if in_source is None and env.local_business_analyst:
        source = 'local'

    # TODO: add check if web gis routing and enrich active - error if not available
    elif in_source is None and active_gis:
        source = active_gis

    # if not using local, use a GIS
    elif isinstance(in_source, GIS):
        source = in_source

    return source


def get_countries(source=None) -> pd.DataFrame:
    """Get input_dataframe of countries available."""
    # TODO: Handle contingency of BA being available, but data not locally installed.
    # TODO: match input_dataframe schema between local and remote GIS instance

    src = set_source(source)

    if src is 'local':

        keys = get_child_key_strs(r'SOFTWARE\WOW6432Node\Esri\BusinessAnalyst\Datasets')

        def _get_dataset_info(key):
            name = os.path.basename(key)

            name_parts = name.split('_')

            country = name_parts[0] if name_parts[1] is not None else None
            year = int(name_parts[2]) if name_parts[2] is not None else None

            return name, country, year

        cntry_info_lst = [_get_dataset_info(k) for k in keys]

        return pd.DataFrame(cntry_info_lst, columns=['geo_ref', 'country', 'year'])

    # TODO: return countries available from GIS object instance


def set_pro_to_usa_local():
    """
    Set the environment setting to ensure using locally installed local data.

    Return:
        Boolean indicating if data correctly enriched.
    """
    try:
        arcpy.env.baDataSource = f'LOCAL;;{os.path.basename(get_ba_usa_key_str())}'
        return True
    except:
        return False


def standardize_geographic_level_input(geo_df, geo_in):
    """Helper function to check and standardize inputs."""
    if isinstance(geo_in, str):
        if geo_in not in geo_df.name.values:
            names = ', '.join(geo_df.names.values)
            raise Exception(
                f'Your selector, "{geo_in}," is not an available selector. Please choose from {names}.')
        return geo_in

    elif isinstance(geo_in, int) or isinstance(geo_in, float):
        if geo_in > len(geo_df.index):
            raise Exception(
                f'Your selector, "{geo_in}", is beyond the maximum range of available geography_levels.')
        return geo_df.iloc[geo_in]['geo_name']

    elif geo_in is None:
        return None

    else:
        raise Exception('The geographic selector ust be a string or integer.')


def get_geographic_level_where_clause(selector=None, selection_field='NAME', query_string=None):
    """Helper function to consolidate where clauses."""
    # set up the where clause based on input
    if query_string:
        return query_string

    elif selection_field and selector:
        return f"{selection_field} LIKE '%{selector}%'"

    else:
        return None


def get_geography_preprocessing(geo_df: pd.DataFrame, geography: [str, int], selector: str = None,
                                selection_field: str = 'NAME', query_string: str = None,
                                aoi_geography: [str, int] = None, aoi_selector: str = None,
                                aoi_selection_field: str = 'NAME', aoi_query_string: str = None) -> tuple:
    """Helper function consolidating input parameters for later steps."""
    # standardize the geography_level input
    geo = standardize_geographic_level_input(geo_df, geography)
    aoi_geo = standardize_geographic_level_input(geo_df, aoi_geography)

    # consolidate selection
    where_clause = get_geographic_level_where_clause(selector, selection_field, query_string)
    aoi_where_clause = get_geographic_level_where_clause(aoi_selector, aoi_selection_field, aoi_query_string)

    return geo, aoi_geo, where_clause, aoi_where_clause


def get_lyr_flds_from_geo_df(df_geo: pd.DataFrame, geo: str, query_str: str = None):
    """
    Get a local feature layer for a geographic level optionally applying a query to filter results.

    Args:
        df_geo:
            Pandas DataFrame of available local resources.
        geo:
            Name of geographic level.
        query_str:
            Optional query string to filter results.

    Returns:
        Tuple containing an Arcpy FeatureLayer with optional query applied as a definition query filtering results,
        and a list with the geographic_level of the ID and NAME fields as a tuple to be included in the output.
    """
    # start by getting the relevant geography_level row from the data
    row = df_geo[df_geo['geographic_level'] == geo].iloc[0]

    # get the id and geographic_level fields along with the path to the data from the row
    fld_lst = [row['col_id'], row['col_name']]
    pth = row['data_path']

    # use the query string, if provided, to create and return a layer with the output fields
    if query_str:
        lyr = arcpy.management.MakeFeatureLayer(str(pth), where_clause=query_str)[0]
    else:
        lyr = arcpy.management.MakeFeatureLayer(str(pth))[0]

    return lyr, fld_lst


def add_enrich_aliases(feature_class: (Path, str), country_object_instance) -> Path:
    """
    Add human readable aliases to an enriched feature class.

    Args:
        feature_class: Path | str
            Path to the enriched feature class.
        country_object_instance: dm.Country
            County object instance for the same country used for initial enrichment.

    Returns: Path
        Path to feature class with aliases added.
    """
    if not env.arcpy_avail:
        raise Exception('add_enrich_aliases requires arcpy to be available since working with ArcGIS Feature Classes')

    # since arcpy tools cannot handle Path objects, convert to string
    feature_class = str(feature_class) if isinstance(feature_class, Path) else feature_class

    # if, at this point, the path is not a string, something is wrong
    if not isinstance(feature_class, str):
        raise Exception(f'The feature_class must be either a Path or string, not {type(feature_class)}.')

    # start by getting a list of all the field names
    fld_lst = [f.name for f in arcpy.ListFields(feature_class)]

    # iterate through the field names and if the field is an enrich field, add the alias
    for fld_nm in fld_lst:

        # get a dataframe, a single or no row dataframe, correlating to the field name
        fld_df = country_object_instance.enrich_variables[
            country_object_instance.enrich_variables.enrich_field_name.str.replace('_', '').str.contains(
                fld_nm.replace('_', ''), case=False
            )
        ]

        # if no field was found, try pattern for non-modified fields - provides pre-Python 1.8.3 support
        if len(fld_df.index) == 0:
            fld_df = country_object_instance.enrich_variables[
                country_object_instance.enrich_variables.enrich_field_name.str.contains(fld_nm, case=False)
            ]

        # if the field name was found, add the alias
        if len(fld_df.index):
            arcpy.management.AlterField(
                in_table=feature_class,
                field=fld_nm,
                new_field_alias=fld_df.iloc[0]['alias']
            )

    return feature_class


def geography_iterable_to_arcpy_geometry_list(geography_iterable: [pd.DataFrame, pd.Series, Geometry,
                                                                   list], geometry_filter: str = None) -> list:
    """
    Processing helper to convert a iterable of geography_levels to a list of ArcPy Geometry objects suitable for input
    into ArcGIS ArcPy geoprocessing tools.

    Args:
        geography_iterable:
            Iterable containing valid geometries.
        geometry_filter:
            String point|polyline|polygon to use for validation to ensure correct geometry type.

    Returns:
        List of ArcPy objects.
    """
    if not env.arcpy_avail:
        raise Exception('Converting to ArcPy geometry requires an environment with ArcPy available.')

    # do some error checking on the geometry filter
    geometry_filter = geometry_filter.lower() if geometry_filter is not None else geometry_filter
    if geometry_filter not in ['point', 'polyline', 'polygon'] and geometry_filter is not None:
        raise Exception(f'geometry_filter must be point|polyline|polygon {geometry_filter}')

    # if a DataFrame, check to ensure is spatial, and convert to list of arcgis Geometry objects
    if isinstance(geography_iterable, pd.DataFrame):
        if geography_iterable.spatial.validate() is True:
            geom_col = [col for col in geography_iterable.columns
                        if geography_iterable[col].dtype.name.lower() == 'geometry'][0]
            geom_lst = list(geography_iterable[geom_col].values)
        else:
            raise Exception('The provided geography_iterable DataFrame does not appear to be a Spatially Enabled '
                            'DataFrame or if so, all geometries do not appear to be valid.')

    # accommodate handling pd.Series input
    elif isinstance(geography_iterable, pd.Series):
        if 'SHAPE' not in geography_iterable.keys() or geography_iterable.name != 'SHAPE':
            raise Exception('SHAPE geometry field must be in the pd.Series or the pd.Series must be the SHAPE to use '
                            'a pd.Series as input.')

        # if just a single row passed in
        elif 'SHAPE' in geography_iterable.keys():
            geom_lst = [geography_iterable['SHAPE']]

        # otherwise, is a series of geometry objects, and just pass over to geom_lst since will be handled as iterable
        else:
            geom_lst = geography_iterable

    # if a list, ensure all child objects are polygon geometries and convert to list of arcpy.Geometry objects
    elif isinstance(geography_iterable, list):
        for geom in geography_iterable:
            if not isinstance(geom, Geometry):
                raise Exception('The provided geometries in the selecting_geometry list do not appear to all be '
                                'valid.')
        geom_lst = geography_iterable

    # if a single geometry object instance, ensure is polygon and make into single item list of arcpy.Geometry
    elif isinstance(geography_iterable, Geometry):
        geom_lst = [geography_iterable]

    else:
        raise Exception('geography_iterable must be either a Spatially Enabled Dataframe, pd.Series with a SHAPE '
                        f'column, list or single geometry object - not {type(geography_iterable)}.')

    # ensure all geometries are correct geometry type if provided
    if geometry_filter is not None:
        for geom in geom_lst:
            if geom.geometry_type != geometry_filter:
                raise Exception('geography_iterable geometries must be polygons. It appears you have provided at '
                                f'least one "{geom.geometry_type}" geometry.')

    # convert the objects in the geometry list to ArcPy Geometry objects
    arcpy_lst = [geom.as_arcpy for geom in geom_lst]

    return arcpy_lst


def clean_columns(column_list: list) -> list:
    """
    Little helper to clean up column names quickly.

    Args:
        column_list:
            List of column names.

    Return:
        List of cleaned up column names.
    """
    def _scrub_col(column):
        no_spc_char = re.sub(r'[^a-zA-Z0-9_\s]', '', column)
        no_spaces = re.sub(r'\s', '_', no_spc_char)
        return re.sub(r'_+', '_', no_spaces)
    return [_scrub_col(col) for col in column_list]


def get_dataframe(in_features: [pd.DataFrame, str, Path, FeatureLayer], gis: GIS = None):
    """
    Get a spatially enabled dataframe from the input features provided.

    Args:
        in_features:
            Spatially Enabled Dataframe | String path to Feature Class | pathlib.Path object to feature
            class | ArcGIS Layer object |String url to Feature Service | String Web GIS Item ID
            Resource to be evaluated and converted to a Spatially Enabled Dataframe.
        gis:
            Optional GIS object instance for connecting to resources.

    Returns:
        Spatially Enabled DataFrame
    """
    # if a path object, convert to a string for following steps to work correctly
    in_features = str(in_features) if isinstance(in_features, Path) else in_features

    # helper for determining if feature layer
    def _is_feature_layer(in_ftrs):
        if hasattr(in_ftrs, 'isFeatureLayer'):
            return in_ftrs.isFeatureLayer
        else:
            return False

    # if already a Spatially Enabled Dataframe, mostly just pass it straight through
    if isinstance(in_features, pd.DataFrame) and in_features.spatial.validate() is True:
        df = in_features

    # if a csv previously exported from a Spatially Enabled Dataframe, get it in
    elif isinstance(in_features, str) and os.path.exists(in_features) and in_features.endswith('.csv'):
        df = pd.read_csv(in_features)
        df['SHAPE'] = df['SHAPE'].apply(lambda geom: Geometry(eval(geom)))

        # this almost always is the index written to the csv, so taking care of this
        if df.columns[0] == 'Unnamed: 0':
            df = df.set_index('Unnamed: 0')
            del (df.index.name)

    # create a Spatially Enabled Dataframe from the direct url to the Feature Service
    elif isinstance(in_features, str) and in_features.startswith('http'):

        # submitted urls can be lacking a few essential pieces, so handle some contingencies with some regex matching
        regex = re.compile(r'((^https?://.*?)(/\d{1,3})?)\?')
        srch = regex.search(in_features)

        # if the layer index is included, still clean by dropping any possible trailing url parameters
        if srch.group(3):
            in_features = f'{srch.group(1)}'

        # ensure at least the first layer is being referenced if the index was forgotten
        else:
            in_features = f'{srch.group(2)}/0'

            # if the layer is unsecured, a gis is not needed, but we have to handle differently
        if gis is not None:
            df = FeatureLayer(in_features, gis).query(out_sr=4326, as_df=True)
        else:
            df = FeatureLayer(in_features).query(out_sr=4326, as_df=True)

    # create a Spatially Enabled Dataframe from a Web GIS Item ID
    elif isinstance(in_features, str) and len(in_features) == 32:

        # if publicly shared on ArcGIS Online this anonymous gis can be used to access the resource
        if gis is None:
            gis = GIS()
        itm = gis.content.get(in_features)
        df = itm.layers[0].query(out_sr=4326, as_df=True)

    elif isinstance(in_features, (str, Path)):
        df = GeoAccessor.from_featureclass(in_features)

    # create a Spatially Enabled Dataframe from a Layer
    elif _is_feature_layer(in_features):
        df = FeatureSet.from_json(arcpy.FeatureSet(in_features).JSON).sdf

    # sometimes there is an issue with modified or sliced dataframes with the SHAPE column not being correctly
    #    recognized as a geometry column, so try to set it as the geometry...just in case
    elif isinstance(in_features, pd.DataFrame) and 'SHAPE' in in_features.columns:
        in_features.spatial.set_geometry('SHAPE')
        df = in_features

        if df.spatial.validate() is False:
            raise Exception('Could not process input features for get_dataframe function. Although the input_features '
                            'appear to be in a Pandas Dataframe, the SHAPE column appears to not contain valid '
                            'geometries. The Dataframe is not validating using the *.spatial.validate function.')

    else:
        raise Exception('Could not process input features for get_dataframe function.')

    # ensure the universal spatial column is correctly being recognized
    df.spatial.set_geometry('SHAPE')

    return df
