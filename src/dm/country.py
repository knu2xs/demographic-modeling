from functools import wraps

import arcgis.gis
import arcgis.features
from arcgis.geometry import Geometry
import numpy as np
import pandas as pd

from . import util
from ._xml_interrogation import get_enrich_variables_dataframe, get_heirarchial_geography_dataframe
from ._modify_geoaccessor import GeoAccessorIO as GeoAccessor

if util.arcpy_avail:
    import arcpy


def local_vs_gis(fn):
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
        elif isinstance(self.source, arcgis.gis.GIS):
            try:
                fn_to_call = getattr(self, f'_{fn_name}_gis')
            except AttributeError:
                raise AttributeError(f"'{fn_name}' not available using a Web GIS as the source.")

        # if another source, we don't plan on doing that any time soon
        else:
            raise AttributeError(f"'{self.source}' is not a recognized demographic modeling source.")

        return fn_to_call(*args, **kwargs)

    return wrapped


class Country:

    def __init__(self, name: str, source: [str, arcgis.gis.GIS] = None):
        self.geo_name = name
        self.source = util.set_source(source)
        self._enrich_variables = None
        self._geographies = None

        # add on all the geographic resolution levels as properties
        for nm in self.geographies.geo_name:
            setattr(self, nm, GeographyLevel(nm, self))

    def __repr__(self):
        return f'<class: Country - {self.geo_name} ({self.source})>'

    def _set_arcpy_ba_country(self):
        """Helper function to set the country in ArcPy."""
        cntry_df = util.get_countries()
        geo_ref = cntry_df[cntry_df['country'] == self.geo_name]['geo_ref'].iloc[0]
        arcpy.env.baDataSource = f'LOCAL;;{geo_ref}'
        return

    @property
    def enrich_variables(self):
        """DataFrame of all the available geoenrichment variables."""
        if self._enrich_variables is None and self.source is 'local':
            self._enrich_variables = get_enrich_variables_dataframe(self.geo_name)

        elif self._enrich_variables is None and isinstance(self.source, arcgis.gis.GIS):
            raise Exception('Using a GIS instance is not yet implemented.')

        return self._enrich_variables

    @property
    def geographies(self):
        """DataFrame of available geographies."""
        if self._geographies is None and self.source is 'local':
            self._geographies = get_heirarchial_geography_dataframe(self.geo_name)

        elif self._geographies is None and isinstance(self.source, arcgis.gis.GIS):
            raise Exception('Using a GIS instance is not yet implemented.')

        return self._geographies

    @local_vs_gis
    def level(self, geography_index: [str, int]) -> pd.DataFrame:
        """
        Get a GeographyLevel at an available geography_level level in the country.

        Args:
            geography_index: Either the geographic_level geo_name or the index of the geography_level level. This can be
                discovered using the Country.geographies method.

        Returns: pd.DataFrame as Geography object instance with the requested geographies.
        """
        pass

    def _level_local(self, geography_index: [str, int]) -> pd.DataFrame:
        """Local implementation of level."""
        # get the geo_name of the geography
        if isinstance(geography_index, int):
            nm = self.geographies.iloc[geography_index]['geo_name']
        elif isinstance(geography_index, str):
            nm = geography_index
        else:
            raise Exception(f'geography_index must be either a string or integer, not {type(geography_index)}')

        # create a GeographyLevel object instance
        return GeographyLevel(nm, self)

    @local_vs_gis
    def enrich(self, data, enrich_variables: [list, np.array, pd.Series] = None,
               data_collections: [str, list, np.array, pd.Series] = None) -> pd.DataFrame:
        """
        Enrich a spatially enabled dataframe using either a list of enrichment variables, or data collections. Either
            enrich_variables or data_collections must be provided, but not both.
        Args:
            data: Spatially Enabled DataFrame with geographies to be enriched.
            enrich_variables: Optional iterable of enrich variables to use for enriching data.
            data_collections: Optional iterable of data collections to use for enriching data.
        Returns: Spatially Enabled DataFrame with enriched data now added.
        """
        pass

    def _enrich_local(self, data, enrich_variables: [list, np.array, pd.Series] = None,
                      data_collections: [str, list, np.array, pd.Series] = None) -> pd.DataFrame:
        """Implementation of enrich for local analysis."""
        # ensure only using enrich_variables or data collections
        if enrich_variables is None and data_collections is None:
            raise Exception('You must provide either enrich_variables or data_collections to perform enrichment')
        elif enrich_variables is not None and data_collections is not None:
            raise Exception('You can only provide enrich_variables or data_collections, not both.')

        # if data collections are provided, get the variables from the geographic variables dataframe
        if data_collections:
            enrich_df = self.enrich_variables[self.enrich_variables.data_collection.isin(data_collections)]
            enrich_variables = enrich_df.drop_duplicates('name')['enrich_str']

        # if just a single variable is provided pipe it into a list
        enrich_variables = [enrich_variables] if isinstance(enrich_variables, str) else enrich_variables

        # ensure all the enrich variables are available
        enrich_vars = pd.Series(enrich_variables)
        missing_vars = enrich_vars[~enrich_vars.isin(self.enrich_variables.enrich_str)]
        if len(missing_vars):
            raise Exception('Some of the variables you provided are not available for enrichment '
                            f'[{", ".join(missing_vars)}]')

        # combine all the enrichment variables into a single string for input into the enrich tool
        enrich_str = ';'.join(enrich_variables)

        # convert the geometry column to a list of arcpy geometry objects
        geom_lst = list(data['SHAPE'].apply(lambda geom: geom.as_arcpy).values)

        # set the arcpy environment to the correct country
        self._set_arcpy_ba_country()

        # invoke the enrich method to get the data
        enrich_fc = arcpy.ba.EnrichLayer(
            in_features=geom_lst,
            out_feature_class='memory/enrich_tmp',
            variables=enrich_str
        )[0]

        # get the Object ID field for schema cleanup
        oid_col = arcpy.Describe(enrich_fc).OIDFieldName

        # convert the enrich feature class to a dataframe and do some schema cleanup
        enrich_df = GeoAccessor.from_featureclass(enrich_fc)
        drop_cols = [c for c in enrich_df.columns if c in [oid_col, 'HasData', 'aggregationMethod', 'SHAPE']]
        enrich_df.drop(columns=drop_cols, inplace=True)

        # combine the two dataframes for output
        out_df = pd.concat([data, enrich_df], axis=1, sort=False)

        # organize the columns so geometry is the last column
        attr_cols = [c for c in out_df.columns if c != 'SHAPE'] + ['SHAPE']
        out_df = out_df[attr_cols].copy()

        # ensure this dataframe will be recognized as spatially enabled
        out_df.spatial.set_geometry('SHAPE')

        return out_df


class GeographyLevel:

    def __init__(self, geographic_level: [str, int], country: Country, parent_data: [pd.DataFrame, pd.Series] = None):
        self._cntry = country
        self.source = country.source
        self.geo_name = self._standardize_geographic_level_input(geographic_level)
        self._resource = None
        self._parent_data = parent_data

    def __repr__(self):
        return f'<class: GeographyLevel - {self.geo_name}>'

    def _standardize_geographic_level_input(self, geo_in: [str, int]) -> str:
        """Helper function to check and standardize named input."""

        geo_df = self._cntry.geographies

        if isinstance(geo_in, str):
            if geo_in not in geo_df.geo_name.values:
                names = ', '.join(geo_df.geo_name.values)
                raise Exception(
                    f'Your selector, "{geo_in}," is not an available selector. Please choose from {names}.')
            geo_lvl_name = geo_in

        elif isinstance(geo_in, int) or isinstance(geo_in, float):
            if geo_in > len(geo_df.index):
                raise Exception(
                    f'Your selector, "{geo_in}", is beyond the maximum range of available geographies.')
            geo_lvl_name = geo_df.iloc[geo_in]['geo_name']

        else:
            raise Exception('The geographic selector must be a string or integer.')

        return geo_lvl_name

    @property
    def resource(self):
        """The resource, either a layer or Feature Layer, for accessing the data for the geographic layer."""
        if self._resource is None and self._cntry.source is 'local':
            self._resource = self._cntry.geographies[self._cntry.geographies['geo_name'] == self.geo_name].iloc[0][
                'feature_class_path']

        elif self._resource is None and isinstance(self._cntry.source, arcgis.gis.GIS):
            raise Exception('Using a GIS instance not yet implemented.')

        return self._resource

    @local_vs_gis
    def get(self, geography: [str, int], selector: str = None, selection_field: str = 'NAME',
            query_string: str = None) -> pd.DataFrame:
        """
        Get a df at an available geography_level level. Since frequently working within an area of interest defined
        by a higher level of geography_level, typically a CBSA or DMA, the ability to specify this area using input
        parameters is also included. This dramatically speeds up the process of creating the output.

        Args:
            geography: Either the geographic_level or the index of the geography_level level. This can be discovered
                using the Country.geographies method.
            selector: If a specific value can be identified using a string, even if just part of the field value,
                you can insert it here.
            selection_field: This is the field to be searched for the string values input into selector.
            query_string: If a more custom query is desired to filter the output, please use SQL here to specify the
                query. The normal query is "UPPER(NAME) LIKE UPPER('%<selector>%'). However, if a more specific query
                is needed, this can be used as the starting point to get more specific.

        Returns: pd.DataFrame as Geography object instance with the requested geographies.
        """
        pass

    def _get_local(self, selector: [str, list] = None, selection_field: str = 'NAME',
                   query_string: str = None) -> pd.DataFrame:

        return self._get_local_df(selector, selection_field, query_string, self._parent_data)

    @local_vs_gis
    def within(self, selecting_geography: [pd.DataFrame, Geometry, list]) -> pd.DataFrame:
        """
        Get a df at an available geography_level level falling within a defined selecting geography.

        Args:
            selecting_geography: Either a Spatially Enabled DataFrame, arcgis.Geometry object instance, or list of
                arcgis.Geometry objects delineating an area of interest to use for selecting geographies for analysis.

        Returns: pd.DataFrame as Geography object instance with the requested geographies.
        """
        pass

    def _within_local(self, selecting_geography: [pd.DataFrame, Geometry, list]) -> pd.DataFrame:
        """Local implementation of within."""
        return self._get_local_df(selecting_geography=selecting_geography)

    def _get_sql_helper(self, selector: [str, list] = None, selection_field: str = 'NAME',
                      query_string: str = None):
        """Helper to handle creation of sql queries for get functions."""
        if query_string:
            sql = query_string
        elif selection_field and isinstance(selector, list):
            sql_lst = [f"UPPER({selection_field}) LIKE UPPER('%{sel}%')" for sel in selector]
            sql = ' OR '.join(sql_lst)
        elif selection_field and isinstance(selector, str):
            sql = f"UPPER({selection_field}) LIKE UPPER('%{selector}%')"
        else:
            sql = None

        return sql

    def _get_local_df(self, selector: [str, list] = None, selection_field: str = 'NAME',
                      query_string: str = None,
                      selecting_geography: [pd.DataFrame, Geometry, list] = None) -> pd.DataFrame:
        """Single function handling business logic for both _get_local and _within_local."""
        # set up the where clause based on input enabling overriding using a custom query if desired
        sql = self._get_sql_helper(selector, selection_field, query_string)

        # if a DataFrame, check to ensure is spatial, and convert to list of arcgis Geometry objects
        if isinstance(selecting_geography, pd.DataFrame):
            if selecting_geography.spatial.validate() is True:
                geom_col = [col for col in selecting_geography.columns
                            if selecting_geography[col].dtype.name.lower() == 'geometry'][0]
                geom_lst = list(selecting_geography[geom_col].values)
            else:
                raise Exception('The provided selecting_geography DataFrame does not appear to be a Spatially Enabled '
                                'DataFrame or if so, all geometries do not appear to be valid.')

        # accommodate passing a single row as a series
        elif isinstance(selecting_geography, pd.Series):
            if 'SHAPE' not in selecting_geography.keys():
                raise Exception('SHAPE geometry field must be in the pd.Series to use a pd.Series as input.')
            else:
                geom_lst = [selecting_geography['SHAPE']]

        # if a list, ensure all child objects are polygon geometries and convert to list of arcpy.Geometry objects
        elif isinstance(selecting_geography, list):
            for geom in selecting_geography:
                if not isinstance(geom, Geometry):
                    raise Exception('The provided geometries in the selecting_geometry list do not appear to all be '
                                    'valid.')
            geom_lst = selecting_geography

        # if a single geometry object instance, ensure is polygon and make into single item list of arcpy.Geometry
        elif isinstance(selecting_geography, Geometry):
            geom_lst = [selecting_geography]

        elif selecting_geography is not None:
            raise Exception('selecting_geography must be either a Spatially Enabled Dataframe, pd.Series with a SHAPE '
                            f'column, list or single geometry object, not {type(selecting_geography)}.')

        # get the relevant geography_level row from the data
        row = self._cntry.geographies[self._cntry.geographies['geo_name'] == self.geo_name].iloc[0]

        # get the id and geographic_level fields along with the path to the data from the row
        fld_lst = [row['col_id'], row['col_name']]
        pth = row['feature_class_path']

        # use the query string, if provided, to create and return a layer with the output fields
        if sql is None:
            lyr = arcpy.management.MakeFeatureLayer(pth)[0]
        else:
            lyr = arcpy.management.MakeFeatureLayer(pth, where_clause=sql)[0]

        # if there is selection data, convert to a layer and use this layer to select features from the above layer.
        if selecting_geography is not None:

            # ensure all geometries are polygons
            for geom in geom_lst:
                if geom.geometry_type != 'polygon':
                    raise Exception('selecting_geography geometries must be polygons. It appears you have provided at '
                                    f'least one "{geom.geometry_type}" geometry.')

            # create a list of arcpy geometry objects
            arcpy_geom_lst = [geom.as_arcpy for geom in geom_lst]

            # create an feature class in memory
            tmp_fc = arcpy.management.CopyFeatures(arcpy_geom_lst, 'memory/tmp_poly')[0]

            # create a layer using the temporary feature class
            sel_lyr = arcpy.management.MakeFeatureLayer(tmp_fc)[0]

            # select local features using the temporary selection layer
            arcpy.management.SelectLayerByLocation(in_layer=lyr, overlap_type='HAVE_THEIR_CENTER_IN',
                                                   select_features=sel_lyr)

        # create a spatially enabled dataframe from the data
        out_data = GeoAccessor.from_featureclass(lyr, fields=fld_lst)

        # get the index of the current geography level
        self_idx = self._cntry.geographies[self._cntry.geographies['geo_name'] == self.geo_name].index[0]

        # add all the geographic levels as properties on the dataframe
        for idx in self._cntry.geographies.index:
            if idx < self_idx:
                geo_name = self._cntry.geographies.iloc[idx]['geo_name']
                setattr(out_data, geo_name, GeographyLevel(geo_name, self._cntry, out_data))

        # add the ability to also get the geography by level index as well
        def get_geo_level_by_index(geo_idx):
            if geo_idx >= self_idx:
                raise Exception('The index for the sub-geography level must be less than the parent. You provided an '
                                f'index of {geo_idx}, which is greater than the parent index of {self_idx}.')
            geo_nm = self._cntry.geographies.iloc[geo_idx]['geo_name']
            return GeographyLevel(geo_nm, self._cntry, out_data)

        setattr(out_data, 'level', get_geo_level_by_index)

        # tack on the country for potential use later
        setattr(out_data, '_cntry', self._cntry)

        return out_data

    @local_vs_gis
    def get_names(self, selector: [str, list] = None, selection_field: str = 'NAME',
                  query_string: str = None) -> pd.Series:
        """
        Get a Pandas Series of available names based on a test input. This runs the same query as the 'get' method,
            except does not return geometry, so it runs a lot faster - providing the utility to test names. If
            no selector string is provided it also provides the ability to see all available names.
        Returns: pd.Series of name strings.
        """
        pass

    def _get_names_local(self, selector: [str, list] = None, selection_field: str = 'NAME',
                         query_string: str = None) -> pd.Series:
        """Local implementation of 'get_names'."""
        # create or use the input query parameters
        sql = self._get_sql_helper(selector, selection_field, query_string)

        # create an output series of names filtered using the query
        out_srs = pd.Series(r[0] for r in arcpy.da.SearchCursor(self.resource, field_names='NAME', where_clause=sql))
        out_srs.name = 'geo_name'

        return out_srs
