from pathlib import Path

import arcgis.gis
import arcgis.features
from arcgis.features import GeoAccessor  # adds spatial property to dataframe
from arcgis.features.geo._internals import register_dataframe_accessor
from arcgis.geometry import Geometry, SpatialReference
import numpy as np
import pandas as pd

from . import utils
from .businesses import Business
from .utils import local_vs_gis, env
from ._xml_interrogation import get_enrich_variables_dataframe, get_heirarchial_geography_dataframe

if env.arcpy_avail:
    import arcpy

    arcpy.env.overwriteOutput = True


class Country:
    """
    Country objects are instantiated by providing the three letter country identifier
    and optionally also specifying the source. If the source is not explicitly specified
    Country will use local resources if the environment is part of an ArcGIS Pro
    installation with Business Analyst enabled and local data installed. If this is not
    the case, Country will then attempt to use the active GIS object instance if available.
    Also, if a GIS object is explicitly passed in, this will be used.

    Args:
        name:
            Three letter country identifier.
        source:
            Either 'local' or a GIS object instance referencing an ArcGIS Enterprise
            instance with enrichment configured or ArcGIS Online.

    .. code-block:: python

        from dm import Country

        # instantiate a country
        usa = Country('USA', source='local')

        # get the seattle CBSA as a study area
        aoi_df = usa.cbsas.get('seattle')

        # use the DemographicAnalysis DataFrame accessor to retrieve geography_levels
        bg_df = aoi_df.dm.block_groups.get()

        # get the available enrich variables as as DataFrame
        e_vars = usa.enrich_variables

        # filter the variables to just the current year key variables
        key_vars = e_vars[(e_vars.data_collection.str.startswith('Key')) &
                          (e_vars.name.str.endswith('CY'))]

        # enrich the data through the DemographicModeling DataFrame accessor
        e_df = bg_df.dm.enrich(key_vars)

    """

    def __init__(self, name: str, source: (str, arcgis.gis.GIS) = None, year: [int, str] = None):
        self.geo_name = name
        self.source = utils.set_source(source)
        self.year = year

        # set up the year
        if self.year is not None:
            assert isinstance(self.year, (str, int)), 'The year parameter must be either a string or integer, not ' \
                                                  f'{type(self.year)}'
            if isinstance(self.year, str):
                self.year = int(self.year)

        self._enrich_variables = None
        self._geographies = None
        self._business = None

        # add on all the geographic resolution levels as properties
        for nm in self.geographies.geo_name:
            setattr(self, nm, GeographyLevel(nm, self))

    def __repr__(self):
        if self.year is None:
            repr_str = f'<dm.Country - {self.geo_name} ({self.source})>'
        else:
            repr_str = f'<dm.Country - {self.geo_name} {self.year} ({self.source})>'
        return repr_str

    def _set_arcpy_ba_country(self):
        """Helper function to set the country in ArcPy."""
        cntry_df = utils.get_countries()
        geo_ref = cntry_df[cntry_df['country'] == self.geo_name]['geo_ref'].iloc[0]
        arcpy.env.baDataSource = f'LOCAL;;{geo_ref}'
        return

    @property
    def business(self):
        """
        Access to business object instance for the country.

        Returns:
            dm.Business object instance.
        """
        if self._business is None:
            self._business = Business(self)
        return self._business

    @property
    def enrich_variables(self):
        """DataFrame of all the available geoenrichment variables."""
        if self._enrich_variables is None and self.source is 'local':
            self._enrich_variables = get_enrich_variables_dataframe(self.geo_name)

        elif self._enrich_variables is None and isinstance(self.source, arcgis.gis.GIS):
            raise Exception('Using a GIS instance is not yet implemented.')

        # add on the country for subsequent analysis
        self._enrich_variables.attrs['_cntry'] = self

        return self._enrich_variables

    @property
    def geography_levels(self):
        """DataFrame of available geography_levels."""
        if self._geographies is None and self.source is 'local':
            self._geographies = get_heirarchial_geography_dataframe(self.geo_name, self.year)

        elif self._geographies is None and isinstance(self.source, arcgis.gis.GIS):
            raise Exception('Using a GIS instance is not yet implemented.')

        return self._geographies

    @local_vs_gis
    def level(self, geography_index: (str, int)) -> pd.DataFrame:
        """
        Get a GeographyLevel at an available geography_level level in the
        country.

        Args:
            geography_index:
                Either the geographic_level geo_name or the
                index of the geography_level level. This can be discovered
                using the ``Country.geography_levels`` method.

        Returns:
            Spatially Enabled DataFrame of the requested geography_levels with
            the DemographicModeling accessor properties initialized.
        """
        pass

    def _level_local(self, geography_index: (str, int)) -> pd.DataFrame:
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
    def enrich(self, data: pd.DataFrame, enrich_variables: (list, np.array, pd.Series, pd.DataFrame) = None,
               data_collections: (str, list, np.array, pd.Series) = None) -> pd.DataFrame:
        """
        Enrich a spatially enabled dataframe using either a list of enrichment
        variables, or data collections. Either ``enrich_variables`` or
        ``data_collections`` must be provided, but not both.

        Args:
            data:
                Spatially Enabled DataFrame with geography_levels to be
                enriched.
            enrich_variables:
                Optional iterable of enrich variables to use for
                enriching data. A filtered output from
                Country.enrich_variables can also be used.
            data_collections:
                Optional iterable of data collections to use for
                enriching data.

        Returns:
            Spatially Enabled DataFrame with enriched data now added.
        """
        pass

    def _enrich_local(self, data, enrich_variables: (list, np.array, pd.Series, pd.DataFrame) = None,
                      data_collections: (str, list, np.array, pd.Series) = None) -> pd.DataFrame:
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

        # if the enrich dataframe is passed in, recognize and work with it
        if isinstance(enrich_variables, pd.DataFrame):
            if 'enrich_str' in enrich_variables.columns:
                enrich_variables = enrich_variables['enrich_str']

        # ensure all the enrich variables are available
        enrich_vars = pd.Series(enrich_variables)
        missing_vars = enrich_vars[~enrich_vars.isin(self.enrich_variables.enrich_str)]
        if len(missing_vars):
            raise Exception('Some of the variables you provided are not available for enrichment '
                            f'[{", ".join(missing_vars)}]')

        # check to make sure there are variables for enrichment
        if len(enrich_variables) == 0:
            raise Exception('There appear to be no variables being selected for enrichment.')

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
        out_df = out_df[attr_cols]

        # ensure this dataframe will be recognized as spatially enabled
        out_df.spatial.set_geometry('SHAPE')

        # ensure WGS84
        out_data = out_df.dm.project(4326)

        # add the country onto the metadata
        out_data.attrs['_cntry'] = self

        return out_data


class GeographyLevel:

    def __init__(self, geographic_level: (str, int), country: Country, parent_data: (pd.DataFrame, pd.Series) = None):
        self._cntry = country
        self.source = country.source
        self.geo_name = self._standardize_geographic_level_input(geographic_level)
        self._resource = None
        self._parent_data = parent_data

    def __repr__(self):
        return f'<class: GeographyLevel - {self.geo_name}>'

    def _standardize_geographic_level_input(self, geo_in: (str, int)) -> str:
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
                    f'Your selector, "{geo_in}", is beyond the maximum range of available geography_levels.')
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
    def get(self, geography: (str, int), selector: str = None, selection_field: str = 'NAME',
            query_string: str = None) -> pd.DataFrame:
        """
        Get a DataFrame at an available geography_level level. Since frequently
        working within an area of interest defined by a higher level of
        geography_level, typically a CBSA or DMA, the ability to specify this
        area using input parameters is also included. This dramatically speeds
        up the process of creating the output.

        Args:
            geography:
                Either the geographic_level or the index of the geography_level
                level. This can be discovered using the Country.geography_levels method.
            selector:
                If a specific value can be identified using a string, even if
                just part of the field value, you can insert it here.
            selection_field:
                This is the field to be searched for the string values
                input into selector.
            query_string:
                If a more custom query is desired to filter the output, please
                use SQL here to specify the query. The normal query is "UPPER(NAME) LIKE
                UPPER('%<selector>%')". However, if a more specific query is needed, this
                can be used as the starting point to get more specific.

        Returns:
            pd.DataFrame as Geography object instance with the requested geography_levels.
        """
        pass

    def _get_local(self, selector: (str, list) = None, selection_field: str = 'NAME',
                   query_string: str = None) -> pd.DataFrame:

        return self._get_local_df(selector, selection_field, query_string, self._parent_data)

    @local_vs_gis
    def within(self, selecting_geography: (pd.DataFrame, Geometry, list)) -> pd.DataFrame:
        """
        Get a input_dataframe at an available geography_level level falling within
        a defined selecting geography.

        Args:
            selecting_geography: Either a Spatially Enabled DataFrame, arcgis.Geometry object instance, or list of
                arcgis.Geometry objects delineating an area of interest to use for selecting geography_levels for analysis.

        Returns: pd.DataFrame as Geography object instance with the requested geography_levels.
        """
        pass

    def _within_local(self, selecting_geography: (pd.DataFrame, Geometry, list)) -> pd.DataFrame:
        """Local implementation of within."""
        return self._get_local_df(selecting_geography=selecting_geography)

    def _get_sql_helper(self, selector: (str, list) = None, selection_field: str = 'NAME',
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

    def _get_local_df(self, selector: (str, list) = None, selection_field: str = 'NAME',
                      query_string: str = None,
                      selecting_geography: (pd.DataFrame, pd.Series, Geometry, list) = None) -> pd.DataFrame:
        """Single function handling business logic for both _get_local and _within_local."""
        # set up the where clause based on input enabling overriding using a custom query if desired
        sql = self._get_sql_helper(selector, selection_field, query_string)

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

            # convert all the selecting geography_levels to a list of ArcPy Geometries
            arcpy_geom_lst = utils.geography_iterable_to_arcpy_geometry_list(selecting_geography, 'polygon')

            # create an feature class in memory
            tmp_fc = arcpy.management.CopyFeatures(arcpy_geom_lst, 'memory/tmp_poly')[0]

            # create a layer using the temporary feature class
            sel_lyr = arcpy.management.MakeFeatureLayer(tmp_fc)[0]

            # select local features using the temporary selection layer
            arcpy.management.SelectLayerByLocation(in_layer=lyr, overlap_type='HAVE_THEIR_CENTER_IN',
                                                   select_features=sel_lyr)

            # clean up arcpy litter
            for arcpy_resource in [tmp_fc, sel_lyr]:
                arcpy.management.Delete(arcpy_resource)

        # ensure something was actually selected
        assert int(arcpy.management.GetCount(lyr)[0]), 'It appears there are not any features in your selection to ' \
                                                       '"get".'

        # create a spatially enabled dataframe from the data in WGS84
        out_data = GeoAccessor.from_featureclass(lyr, fields=fld_lst).dm.project(4326)

        # tack on the country and geographic level name for potential use later
        out_data.attrs['_cntry'] = self._cntry
        out_data.attrs['geo_name'] = self.geo_name

        return out_data

    @local_vs_gis
    def get_names(self, selector: (str, list) = None, selection_field: str = 'NAME',
                  query_string: str = None) -> pd.Series:
        """
        Get a Pandas Series of available names based on a test input. This runs the
            same query as the 'get' method, except does not return geometry, so it
            runs a lot faster - providing the utility to test names. If no selector
            string is provided it also provides the ability to see all available names.

        Args:
            selector: If a specific value can be identified using a string, even if
                just part of the field value, you can insert it here.
            selection_field: This is the field to be searched for the string values
                input into selector.
            query_string: If a more custom query is desired to filter the output, please
                use SQL here to specify the query. The normal query is "UPPER(NAME) LIKE
                UPPER('%<selector>%'). However, if a more specific query is needed, this
                can be used as the starting point to get more specific.

        Returns: pd.Series of name strings.
        """
        pass

    def _get_names_local(self, selector: (str, list) = None, selection_field: str = 'NAME',
                         query_string: str = None) -> pd.Series:
        """Local implementation of 'get_names'."""
        # create or use the input query parameters
        sql = self._get_sql_helper(selector, selection_field, query_string)

        # create an output series of names filtered using the query
        out_srs = pd.Series(r[0] for r in arcpy.da.SearchCursor(self.resource, field_names='NAME', where_clause=sql))
        out_srs.name = 'geo_name'

        return out_srs


@register_dataframe_accessor('dm')
class DemographicModeling:
    """
    DemograpphicModeling is a Pandas DataFrame accessor, a standalone namespace for
    accessing demographic modeling functionality. If the DataFrame was created using
    a Country object, then the DemographicModeling (``dm``) namespace will automatically
    be available. However, if you want to use this functionality, and have not created
    the DataFrame using the Country object, you must import DemographicModeling, and
    then you will have this functionality available.
    """

    def __init__(self, obj):
        self._data = obj
        self._index = obj.index

        # save the country if it is passed from the invoking parent
        if '_cntry' in obj.attrs.keys():
            self._cntry = obj.attrs['_cntry']
        elif hasattr(obj, '_cntry'):
            self._cntry = obj._cntry
        else:
            self._cntry = None

        # if geo_name is a property of the dataframe, is the output of a chained function, and we can add capability
        if 'geo_name' in obj.attrs.keys():

            # get the geographic level index
            self._geo_idx = self._cntry.geographies[self._cntry.geographies['geo_name'] == self._data.attrs['geo_name']].index[0]

            # add all the geographic levels below the current geographic level as properties
            for idx in self._cntry.geographies.index:
                if idx < self._geo_idx:
                    geo_name = self._cntry.geographies.iloc[idx]['geo_name']
                    setattr(self, geo_name, GeographyLevel(geo_name, self._cntry, obj))

    def level(self, geographic_level: int) -> GeographyLevel:
        """
        Retrieve a Spatially Enabled DataFrame of geometries corresponding
        to the index returned by the Country.geography_levels property. This is
        most useful when retrieving the lowest, most granular, level of
        geography within a country.

        Args:
            geographic_level:
                Integer referencing the index of the geographic level desired.

        Returns:
            GeographyLevel object instance

        .. code-block:: python

            from dm import Country

            # create an instance of the country object
            cntry = Country('USA')

            # the get function returns a dataframe with the 'dm' property
            metro_df = cntry.cbsas('seattle')

            # level returns a CountryLevel object enabling getting all geography_levels
            # falling within the parent dataframe
            lvl_df = metro_df.dm.level(0).get()

        """
        assert self._cntry is not None, "The 'dm.level' method requires the parent dataframe be created by the" \
                                        "Country object."

        assert geographic_level <= self._geo_idx, 'The index for the sub-geography level must be less than the ' \
                                                  f'parent. You provided an index of {geographic_level}, ' \
                                                  f'which is greater than the parent index of {self._geo_idx}. '

        # get the name of the geographic level corresponding to the provided index
        geo_nm = self._cntry.geography_levels.iloc[geographic_level]['geo_name']

        # create a geographic level object
        geo_lvl = GeographyLevel(geo_nm, self._cntry, self._data)

        return geo_lvl

    def enrich(self, enrich_variables: (list, np.array, pd.Series, pd.DataFrame) = None,
               data_collections: (str, list, np.array, pd.Series) = None, country: Country = None) -> pd.DataFrame:
        """
        Enrich the DataFrame using the provided enrich variable list or data
        collections list. Either a variable list or list of data
        collections can be provided, but not both.

        Args:
            enrich_variables:
                List of data variables for enrichment. This can optionally
                be a filtered subset of the dataframe property of an instance
                of the Country object.
            data_collections:
                List of data collections for enrichment.
            country: Optional
                Country object instance. This must be included if the parent
                dataframe was not created using this package's standard
                geography methods, or if the enrichment variables are not
                defined by passing in an enrich variables dataframe created
                using this package's introspection methods.

        Returns:
            pd.DataFrame with enriched data.

        .. code-block:: python

            from pathlib import Path

            from arcgis import GeoAccessor
            from dm import Country, DemographicModeling
            import pandas as pd

            # get a path to the trade area data
            prj_pth = Path(__file__).parent
            gdb_pth = dir_data/'data.gdb'
            fc_pth = gdb/'trade_areas'

            # load the trade areas into a Spatially Enabled DataFrame
            ta_df = pd.DataFrame.spatial.from_featureclass(fc_pth)

            # create a country object instance
            usa = Country('USA', source='local')

            # get all the available enrichment variables
            e_vars = usa.enrich_variables

            # filter to just the current year key variables
            key_vars = e_vars[(e_vars.data_collection.str.startswith('Key')) &
                              (e_vars.name.str.endswith('CY'))]

            # enrich the Spatially Enabled DataFrame
            tae_df = ta_df.dm.enrich(key_vars)

        """
        # prioirtize the country parameter
        if country is not None:
            cntry = country

        # next, if the enrich variables has the country defined
        elif '_cntry' in enrich_variables.attrs.keys():
            cntry = enrich_variables.attrs['_cntry']

        # now, see if the parent dataframe has a country property
        elif hasattr(self, '_cntry'):
            cntry = self._cntry

        # otherwise, we don't know what to do
        else:
            cntry = None

        assert isinstance(cntry, Country), "The 'dm.enrich' method requires the parent dataframe be created by the " \
                                           "Country object, the enrich variables to be provided as a dataframe " \
                                           "retrieved from a Country object, or a valid Country object must be " \
                                           "explicitly provided as input into the country parameter."

        # get the data from the GeoAccessor _data property
        data = self._data

        # invoke the enrich method from the country
        out_df = cntry.enrich(data, enrich_variables, data_collections)

        return out_df

    def project(self, output_spatial_reference: (SpatialReference, int) = 4326):
        """
        Project to a new spatial reference, applying an applicable transformation if necessary.

        Args:
            output_spatial_reference:
                Optional - The output spatial reference. Default is 4326 (WGS84).

        Returns:
            Spatially Enabled DataFrame projected to the new spatial reference.
        """
        # import needed resources
        from .spatial import project_as

        # perform the projection
        return project_as(self._data, output_spatial_reference)

    def get_nearest(self, destination_dataframe: pd.DataFrame, source: (str, Path, Country, arcgis.gis.GIS) = None,
                    single_row_per_origin: bool = True, origin_id_column: str = 'LOCNUM',
                    destination_id_column: str = 'LOCNUM', destination_count: int = 4, near_prefix: str = None,
                    destination_columns_to_keep: (str, list) = None) -> pd.DataFrame:
        """
        Create a closest destination dataframe using a destination Spatially Enabled
        Dataframe relative to the parent Spatially enabled DataFrame, but keep each
        origin and destination still in a discrete row instead of collapsing to a
        single row per origin. The main reason to use this is if needing the geometry
        for visualization.

        Args:
            destination_dataframe:
                Destination points in one of the supported input formats.
            source:
                Optional - Either the path to the network dataset, the Country object
                associated with the Business Analyst source being used, or a GIS object
                instance. If invoked from a dataframe created for a country's standard
                geography levels using the dm accessor, get_nearest will use the parent
                country properties to ascertain how to perform the networks solve.
            single_row_per_origin:
                Optional - Whether or not to pivot the results to return
                only one row for each origin location. Default is True.
            origin_id_column:
                Optional - Column in the origin points Spatially Enabled Dataframe
                uniquely identifying each feature. Default is 'LOCNUM'.
            destination_id_column:
                Column in the destination points Spatially Enabled Dataframe
                uniquely identifying each feature
            destination_count:
                Integer number of destinations to search for from every origin
                point.
            near_prefix:
                String prefix to prepend onto near column names in the output.
            destination_columns_to_keep:
                List of columns to keep in the output. Commonly, if
                businesses, this includes the column with the business names.

        Returns:
            Spatially Enabled Dataframe with a row for each origin id, and metrics for
            each nth destinations.
        """
        # retrieve resources needed
        from .proximity import get_nearest

        # if the source is provided,
        source = self._cntry if source is None else source

        # solve get nearest
        near_df = get_nearest(self._data, destination_dataframe, source, single_row_per_origin, origin_id_column,
                              destination_id_column, destination_count, near_prefix, destination_columns_to_keep)

        # if the source is a country, tack it on for any follow-on analysis
        if isinstance(source, Country):
            setattr(near_df, '_cntry', source)

        return near_df
