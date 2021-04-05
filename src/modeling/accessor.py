"""Provide modeling accessor object namespace and methods."""
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Union
from warnings import warn

from arcgis.features import GeoAccessor, FeatureSet
from arcgis.features.geo._internals import register_dataframe_accessor
from arcgis.gis import GIS
from arcgis.geometry import Geometry, SpatialReference

from .country import Country, GeographyLevel
from .utils import avail_arcpy, local_vs_gis, geography_iterable_to_arcpy_geometry_list, validate_spatial_reference, \
    get_top_codes, preproces_code_inputs

if avail_arcpy:
    import arcpy
    from ._xml_interrogation import get_business_points_data_path


@register_dataframe_accessor('mdl')
class ModelingAccessor:
    """
    The ModelingAccessor is a Pandas DataFrame accessor, a standalone namespace for
    accessing geographic modeling functionality. If the DataFrame was created using
    a Country object, then the Modeling (``mdl``) namespace will automatically
    be available. However, if you want to use this functionality, and have not created
    the DataFrame using the Country object, you must import arcgis.modeling.Modeling
    to have this functionality available.
    """

    def __init__(self, obj):
        self._data = obj
        self._index = obj.index
        self.business = Business(self)

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
            self._geo_idx = self._cntry.levels[self._cntry.levels['geo_name'] == self._data.attrs['geo_name']].index[0]

            # add all the geographic levels below the current geographic level as properties
            for idx in self._cntry.levels.index:
                if idx < self._geo_idx:
                    geo_name = self._cntry.levels.iloc[idx]['geo_name']
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
            lvl_df = metro_df.mdl.level(0).get()

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

    def enrich(self, enrich_variables: Union[list, np.array, pd.Series, pd.DataFrame] = None,
               country: Country = None) -> pd.DataFrame:
        """
        Enrich the DataFrame using the provided enrich variable list.

        Args:
            enrich_variables:
                List of data variables for enrichment. This can optionally
                be a filtered subset of the dataframe property of an instance
                of the Country object.
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
            ta_df = ta_df.dm.enrich(key_vars)

        """
        # prioritize the country parameter
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

        assert isinstance(cntry, Country), "The 'modeling.enrich' method requires the parent dataframe be created by " \
                                           "the Country object, the enrich variables to be provided as a dataframe " \
                                           "retrieved from a Country object, or a valid Country object must be " \
                                           "explicitly provided as input into the country parameter."

        # get the data from the GeoAccessor _data property
        data = self._data

        # invoke the enrich method from the country
        out_df = cntry.enrich(data, enrich_variables)

        return out_df

    def project(self, output_spatial_reference: Union[SpatialReference, int] = 4326):
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

    def get_nearest(self, destination_dataframe: pd.DataFrame, source: Union[str, Path, Country, GIS] = None,
                    single_row_per_origin: bool = True, origin_id_column: str = 'LOCNUM',
                    destination_id_column: str = 'LOCNUM', destination_count: int = 4, near_prefix: str = None,
                    destination_columns_to_keep: Union[str, list] = None) -> pd.DataFrame:
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


### All below is ported from business module to inegrate into accessor ###

def get_businesses_gis(area_of_interest: pd.DataFrame, gis: GIS, search_string: str = None,
                       code_naics: Union[str, list] = None, code_sic: Union[str, list] = None,
                       exclude_headquarters: bool = True, country: Country = None,
                       output_spatial_reference: Union[str, int, dict, SpatialReference] = {'wkid': 4326}
                       ) -> pd.DataFrame:
    """Method to get business using a Web GIS object instance, so retrieving through REST. This is wrapped by methods
    in the Business object."""
    # list of fields to lighten the output payload
    out_flds = ['LOCNUM', 'CONAME', 'NAICSDESC', 'NAICS', 'SIC', 'SOURCE', 'PUBPRV', 'FRNCOD', 'ISCODE', 'CITY', 'ZIP',
                'STATE', 'HDBRCHDESC']

    # begin to build up the request parameter payload
    params = {
        'f': 'json',
        'returnGeometry': True,
        'outSr': validate_spatial_reference(output_spatial_reference),
        'fields': out_flds
    }

    # make sure a country is explicitly specified
    if '_cntry' in area_of_interest.attrs:
        params['sourceCountry'] = area_of_interest.attrs['_cntry'].iso2

    # if a country object was explicitly passed in - easy peasy lemon squeezy
    elif country is not None and 'iso2' in country.__dict__.keys():
        params['sourceCountry'] = country.iso2

    # if there is not a country to work with, bingo out
    else:
        raise Exception('Either the input dataframe must have been created using the modeling '
                        'module to retrieve standard geographies, or a country object must be '
                        'explicitly specified in the input parameters.')

    # make sure some sort of filter is being applied
    assert (search_string is not None) or (code_naics is not None), 'You must provide either a search string or ' \
                                                                    'NAICS code or list of codes to search for ' \
                                                                    'businesses.'

    # populate the rest of the search parameters
    params['searchstring'] = search_string
    params['businesstypefilters'] = [
        {'Classification': 'NAICS', 'Codes': preproces_code_inputs(code_naics)},
        {'Classification': 'SIC', 'Codes': preproces_code_inputs(code_sic)}
    ]

    # if the input Spatially Enabled DataFrame was created using the modeling module to get standard geographies
    if 'parent_geo' in area_of_interest.attrs.keys():
        params['spatialfilter'] = {
            "Boundaries": {
                "StdLayer": {
                    "ID": area_of_interest.attrs['parent_geo']['resource'],
                    "GeographyIDs": area_of_interest.attrs['parent_geo']['id']
                }
            }
        }

    # if just a normal spatially enabled dataframe, we can use a FeatureSet with the geometry
    else:
        params['spatialfilter'] = {
            "Boundaries": {
                "recordSet": area_of_interest.spatial.to_featureset().to_dict()
            }
        }

    # retrieve the businesses from the REST endpoint
    url = f'{gis.properties.helperServices.geoenrichment.url}/SelectBusinesses'
    r_json = gis._con.post(url, params=params)

    # ensure a valid result is received
    if 'error' in r_json.keys():
        err = r_json['error']
        raise Exception(f'Error in searching using Business Analyst SelectBusinesses REST endpoint. Error Code '
                        f'{err["code"]}: {err["message"]}')

    else:

        # plow through all the messages to see if there are any errors
        err_msg_lst = []
        for val in r_json['messages']:
            if 'description' in val.keys():
                if 'error' in val['description'].lower():
                    err_msg_lst.append(val)

        # if an error message is found
        if len(err_msg_lst):
            err = err_msg_lst[0]
            raise Exception(
                f'Server error encoutered in searching using Business Analyst SelectBusinesses REST endpoint. '
                f'Error ID: {err["id"]}, Type: {err["type"]}, Description: {err["description"]}')

    # extract the feature list out of the json response
    feature_lst = r_json['results'][0]['value']['features']

    # make sure something was found
    if len(feature_lst) == 0:
        warn('Although the request was valid and no errors were encountered, no businesses were found.')

    # convert the features to a Spatially Enabled Pandas DataFrame
    res_df = FeatureSet(feature_lst).sdf

    # reorganize the schema a little
    cols = [c for c in out_flds if c in res_df.columns] + ['SHAPE']
    res_df = res_df[cols].copy()

    # if not wanting to keep headquarters, normally the case for forecasting modeling, filter them out
    if exclude_headquarters and 'HDBRCHDESC' in res_df.columns:
        res_df = res_df[~res_df['HDBRCHDESC'].str.lower().str.match('headquarters')].reset_index(drop=True)

    # drop the headquarters or branch column since only used to filter if necessary
    if 'HDBRCHDESC' in res_df.columns:
        res_df.drop(columns='HDBRCHDESC', inplace=True)

    return res_df


class Business:
    """
    Just like it sounds, this is a way to search for and find
    businesses of your own brand for analysis, but more importantly
    competitor locations facilitating modeling the effects of
    competition as well. The business object is accessed as a property
    of the ModelingAccessor (``df.mdl.business``).

    .. code-block:: python

        from modeling import Country

        # start by creating a country object instance
        usa = Country('USA')

        # get a geography to work with from locally installed data
        aoi_df = usa.cbsas.get('Seattle')

        # get all Ace Hardware locations
        brnd_df = aoi_df.mdl.business.get_by_name('Ace Hardware')

        # get all competitors for Ace Hardware in Seattle using the
        # template of the brand dataframe
        comp_df = aoi_df.mdl.business.get_competition(brnd_df)

        # ...or get competitors by using the same search term
        comp_df = aoi_df.mdl.business.get_competition('Ace Hardware')

    """

    def __init__(self, mdl: ModelingAccessor):
        self._data = mdl._data

        if '_cntry' in self._data.attrs.keys():
            self._cntry = self._data.attrs['_cntry']
            self.source = self._cntry.source
        else:
            self._cntry, self.source = None, None

    def __repr__(self):
        """What to show when representing an instance of the object."""
        if self._cntry is not None and self.source is not None:
            repr_str = f'<dm.Business in {self._cntry.geo_name} ({self.source})>'
        else:
            repr_str = f'<dm.Business>'
        return repr_str

    def _get_arcpy_lyr(self, sql_clause=None):
        """Helper function to create an arcpy layer of business listings"""
        return arcpy.management.MakeFeatureLayer(get_business_points_data_path(self._cntry.geo_name),
                                                 where_clause=sql_clause)[0]

    @staticmethod
    def _local_select_by_location(area_of_interest: [pd.DataFrame, pd.Series, Geometry, list], lyr,
                                  selection_type: str = 'SUBSET_SELECTION'):
        """Helper function to facilitate selecting by location in an area of interest."""
        # get from the aoi input to an aoi layer
        aoi_lst = geography_iterable_to_arcpy_geometry_list(area_of_interest, 'polygon')
        aoi_fc = arcpy.management.CopyFeatures(aoi_lst, 'memory/tmp_aoi')[0]
        aoi_lyr = arcpy.management.MakeFeatureLayer(aoi_fc)[0]

        # use the area of interest to select features
        arcpy.management.SelectLayerByLocation(lyr, select_features=aoi_lyr, selection_type=selection_type)

        return lyr

    def _local_get_by_attribute_and_aoi(self, sql: str,
                                        area_of_interest: [pd.DataFrame, pd.Series, Geometry, list]) -> pd.DataFrame:
        """Helper function following DRY to enable selecting by attributes and filtering to an AOI."""
        lyr = self._get_arcpy_lyr(sql)

        # filter to the area of interest, and get the results
        lyr = self._local_select_by_location(area_of_interest, lyr)

        # convert results to a spatially enabled dataframe
        out_df = GeoAccessor.from_featureclass(lyr)

        # project to WGS84 - if there is data even retrieved
        if len(out_df.index) > 0:
            out_df = out_df.dm.project(4326)

        return out_df

    def _add_std_cols(self, biz_df: pd.DataFrame, id_col: str, name_col: str, local_threshold: int = 0) -> pd.DataFrame:
        """Helper function adding values in a standard column making follow on analysis workflows easier."""
        # assign the location id and brand name to standardized columns
        biz_df['location_id'] = biz_df[id_col]
        biz_df['brand_name'] = biz_df[name_col]

        # calculate the brand name category column
        biz_df.mdl.business.calculate_brand_name_category(local_threshold, inplace=True)

        return biz_df

    def calculate_brand_name_category(self, local_threshold: int = 0,
                                      inplace: bool=False) -> Union[pd.DataFrame, None]:
        """
        For the output of any Business.get* function, calculate a column named 'brand_name_category'. This function is
        frequently used to re-calculate the category identifying unique local retailers, and group them collectively
        into a 'local_brand'. This is useful in markets where there is a distinct preference for local retailers. This
        is particularly true for speciality coffee shops in many urban markets. While this is performed
        automatically for the 'get_by_code' and 'get_competitors' methods, this function enables you to recalculate it
        if you need to massage some of the brand name outputs.

        Args:
            local_threshold: Integer count below which a brand name will be consider a local brand.
            inplace: Boolean indicating if the dataframe should be modified in place, or a new one created and returned.
                The default is False to not inadvertently

        Returns:
            Pandas Spatially Enabled DataFrame of store locations with the updated column if inplace is False.
            Otherwise, returns None.

        .. code-block:: python

            from arcgis.gis import GIS
            from modeling import Country

            # connect to a Web GIS
            gis = GIS('https://path.to.arcgis.enterprise.com/portal',
                      username='batman', password='P3nnyw0rth!')

            # create a country object instance
            cntry = Country('USA', source=gis)

            # create an area of interest
            aoi_df = cntry.cbsas.get('Seattle')

            # use this area of interest to get brand locations
            brand_df = aoi_df.mdl.business.get_by_name('Ace Hardware')

            # get competitors and categorize all brands with less than
            # three locations as a local brand
            comp_df = aoi_df.mdl.business.get_competitors(brand_df, local_threshold=3)

            # with hardware stores, each True Value has a unique name,
            # so it helps to rename these to be correctly recognized
            # as a brand of stores
            brand_filter = comp_df.brand_name.str.contains(r'TRUE VALUE|TRUE VL', regex=True)
            comp_df.loc[brand_filter, 'brand_name'] = 'TRUE VALUE'

            # now, with the True Values renamed, we need to recalculate which
            # locations are actually local brands
            comp_df.mdl.business.calculate_brand_name_category(3, inplace=True)

        """
        # get the dataframe
        brnd_df = self._data

        assert 'brand_name' in brnd_df.columns, 'The "brand_name" column was not found in the input. ' \
                                                'Please ensure the input is the output from a ' \
                                                'Business.get* function.'

        # get the unique values below the threshold
        local_srs = brnd_df['brand_name'].value_counts() > local_threshold
        brand_names = local_srs[local_srs].index.values

        # if not inplace, need to copy the dataframe
        biz_df = brnd_df if inplace else brnd_df.copy()

        # calculate the local_brand records based on the stated threshold
        brnd_cat_fltr = biz_df['brand_name'].isin(brand_names)
        biz_df.loc[~brnd_cat_fltr, 'brand_name_category'] = 'local_brand'
        biz_df.loc[brnd_cat_fltr, 'brand_name_category'] = biz_df.loc[brnd_cat_fltr]['brand_name']

        return None if inplace else biz_df

    def drop_by_id(self, drop_dataframe: pd.DataFrame, source_id_column: str = 'location_id',
                   drop_id_column: str = 'location_id') -> pd.DataFrame:
        """
        Drop values from the parent dataframe based on unique identifiers found in another dataframe. This is a common
        task when removing brand locations from a dataframe of all locations to create a dataframe of only competitors.

        Args:
            drop_dataframe: Required Pandas DataFrame with a unique identifier column. Values in this column will be
                used to identify and remove values from the dataframe.
            source_id_column: Optional string for the column in the original dataframe with values to be used for
                identifying rows to either drop or retain. Default is 'location_id'.
            drop_id_column: Optional string for the column in the drop dataframe with values to be used for identifying
                rows to drop or retain. Default is 'location_id'.

        Returns:
            Pandas DataFrame with rows removed based on common identifier values.

        .. code-block:: python

            from arcgis.gis import GIS
            from modeling import Country

            # connect to a Web GIS
            gis = GIS('https://path.to.arcgis.enterprise.com/portal',
                      username='batman', password='P3nnyw0rth!')

            # create a country object instance
            cntry = Country('USA', source=gis)

            # create an area of interest
            aoi_df = cntry.cbsas.get('Seattle')

            # use this area of interest to get brand locations
            brand_df = aoi_df.mdl.business.get_by_name('Ace Hardware')

            # get the top NAICS codes
            top_codes = brand_df.mdl.business.get_top_codes()

            # truncate the top code retrieved to widen the scope of codes
            # retrieved - a broader category
            top_code = top_codes.iloc[0]
            naics_code = top_code[:-4]

            # use this truncated code to retrieve competitors
            naics_df = aoi_df.mdl.business.get_by_code(naics_code)

            # now, remove the brand locations from the retrieved dataframe
            # to retain just the competition
            comp_df = naics_df.mdl.business.drop_by_id(brand_df)

        """
        # retrieve the data from the parent source
        source_df = self._data

        # ensure columns are actually in both the source and drop dataframes
        assert source_id_column in source_df.columns, f'It appears the source_id_column, {source_id_column}, is not ' \
                                                      f'a column in the source dataframe.'
        assert drop_id_column in drop_dataframe.columns, f'It appears the drop_id_column, {drop_id_column}, is not ' \
                                                         f'a column in the drop dataframe.'

        # ensure the data column data types match
        src_typ, drp_typ = source_df[source_id_column].dtype, drop_dataframe[drop_id_column].dtype
        assert src_typ == drp_typ, f'The data type of the id columns must match. They appear to be {src_typ} and ' \
                                   f'{drp_typ}.'

        # remove brand locations from the result based on the unique identifier column
        drop_filter = source_df[source_id_column].isin(drop_dataframe[drop_id_column])
        out_df = source_df[~drop_filter].copy().reset_index(drop=True)

        return out_df

    def get_top_codes(self, code_type: str = 'NAICS', threshold: float = 0.5) -> pd.Series:
        """
        Get the industry identifier codes used to identify MOST of the records in a business DataFrame. This is
        useful for getting the identifier values to retrieve other business locations to identify competitors.

        Args:
            code_type: Optional string identifying the industry codes being used. Must be either NAICS or SIC. Default
                is 'NAICS'.
            threshold: Optional float determining what percentage to use as threshold cutoff from input records to
                select codes from. Default is 0.5. This means the top 50%, or half, of the rows in the DataFrame will
                be sampled to return the industry code values identifying the locations.

        Returns:
            Pandas Series of code values.

        .. code-block:: python

            from arcgis.gis import GIS
            from modeling import Country

            # connect to a Web GIS
            gis = GIS('https://path.to.arcgis.enterprise.com/portal',
                      username='batman', password='P3nnyw0rth!')

            # create a country object instance
            cntry = Country('USA', source=gis)

            # create an area of interest
            aoi_df = cntry.cbsas.get('Minneapolis')

            # use this area of interest to get brand locations
            brand_df = aoi_df.mdl.business.get_by_name('Ulta Beauty')

            # get the top NAICS codes
            top_codes = brand_df.mdl.business.get_top_codes()

            # truncate the top code retrieved to widen the scope of codes
            # retrieved - a broader category
            top_code = top_codes.iloc[0]
            naics_code = top_code[:-2]

            # use this truncated code to retrieve competitors
            naics_df = aoi_df.mdl.business.get_by_code(naics_code)

            # now, remove the brand locations from the retrieved dataframe
            # to retain just the competition
            comp_df = naics_df.mdl.business.drop_by_id(brand_df)

        """
        # make sure the codes are available
        assert code_type in self._data.columns, f'The code column provided, {code_type}, does not appear to be ' \
                                                f'available in the dataframe.'

        # get a Pandas Series of the codes
        code_srs = self._data[code_type]

        # get the unique values
        top_codes = pd.Series(get_top_codes(code_srs, threshold))

        return top_codes

    @local_vs_gis
    def get_by_name(self, business_name: str, name_column: str = 'CONAME', id_column: str = 'LOCNUM',
                    local_threshold: int = 0) -> pd.DataFrame:
        """
        Search business listings for a specific business name string.

        Args:
            business_name:
                String business name to search for.
            name_column:
                Optional - Name of the column with business names to be searched. Default is 'CONAME'
            id_column:
                Optional - Name of the column with the value uniquely identifying each business location. Default
                is 'LOCNUM'.
            local_threshold:
                Number of locations to consider, albeit only in the study area, to categorize the each
                business location as either a major brand, and keep the name, or as a local brand with 'local_brand'
                in a new column. This enables considering local brands in a market collectively to quantitatively
                evaluate the power of "buying local."

        Returns:
            Spatially Enabled DataFrame of businesses

        .. code-block:: python

            from modeling import Country

            # start by creating a country object instance
            usa = Country('USA', source='local')

            # get a geography to work with from locally installed data
            aoi_df = usa.cbsas.get('seattle')

            # get all Ace Hardware locations in Seattle
            comp_df = aoi_df.mdl.business.get_by_name('Ace Hardware')

        """
        pass

    @local_vs_gis
    def get_by_code(self, category_code: [str, list], code_type: str = 'NAICS', name_column: str = 'CONAME',
                    id_column: str = 'LOCNUM', local_threshold: int = 0) -> pd.DataFrame:
        """
        Search for businesses based on business category code. In North America, this typically is either the NAICS or
        SIC code.

        Args:
            category_code: Required
                Business category code, such as 4568843, input as a string. This does not have to be a
                complete code. The tool will search for the category code with a partial code starting from the
                beginning.
            code_type: Optional
                The column in the business listing data to search for the input business code. In the United
                States, this is either NAICS or SIC. The default is NAICS.
            name_column: Optional
                Name of the column with business names to be searched. Default is 'CONAME'
            id_column: Optional
                Name of the column with the value uniquely identifying each business location. Default
                is 'LOCNUM'.
            local_threshold: Optional
                Number of locations to consider, albeit only in the study area, to categorize the each
                business location as either a major brand, and keep the name, or as a local brand with 'local_brand'
                in a new column.

        Returns:
            Spatially Enabled pd.DataFrame

        .. code-block:: python

            from modeling import Country

            # start by creating a country object instance
            usa = Country('USA', source='local')

            # get a geography to work with from locally installed data
            aoi_df = usa.cbsas.get('seattle')

            # get all the businesses for a NAICS code category
            naics_big_df = aoi_df.mdl.business.get_by_code(4441, local_threshold=2)

        """
        pass

    @local_vs_gis
    def get_competition(self, brand_businesses: pd.DataFrame, code_column: str = 'NAICS', name_column: str = 'CONAME',
                        id_column: str = 'LOCNUM', local_threshold: int = 0) -> pd.DataFrame:
        """
        Get competitors from previously retrieved business listings.

        Args:
            brand_businesses:
                Previously retrieved business listings.
            name_column:
                Optional - Name of the column with business names to be searched. Default is 'CONAME'
            code_column:
                Optional - The column in the data to search for business category codes. Default is 'NAICS'
            id_column:
                Optional - Name of the column with the value uniquely identifying each business location. Default
                is 'LOCNUM'.
            local_threshold:
                Number of locations to consider, albeit only in the study area, to categorize the each
                business location as either a major brand, and keep the name, or as a local brand with 'local_brand'
                in a new column.

        Returns:
            Spatially Enabled DataFrame

        .. code-block:: python

            from modeling import Country

            # start by creating a country object instance
            usa = Country('USA', source='local')

            # get a geography to work with from locally installed data
            aoi_df = usa.cbsas.get('seattle')

            # get all competitors for Ace Hardware in Seattle
            comp_df = aoi_df.mdl.business.get_competition('Ace Hardware')

        """
        pass

    def _get_by_code_local(self, category_code: [str, list], code_type: str = 'NAICS', name_column: str = 'CONAME',
                           id_column: str = 'LOCNUM', local_threshold: int = 0) -> pd.DataFrame:
        """Local implementation for get by code."""
        # if the code was input as a number, convert to a string
        category_code = str(category_code) if isinstance(category_code, int) else category_code

        # get the area of interest dataframe
        aoi_df = self._data

        # create the sql query based on whether a single or multiple codes were submitted
        if isinstance(category_code, str):
            sql = f"{code_type} LIKE '{category_code}%'"
        elif isinstance(category_code, list):
            sql_lst = [f"{code_type} LIKE '{cd}%'" for cd in category_code]
            sql = ' OR '.join(sql_lst)
        else:
            raise Exception("The category code must be either a string or list of strings, not "
                            f"'{type(category_code)}'.")

        # get using the query and aoi
        biz_df = self._local_get_by_attribute_and_aoi(sql, aoi_df)

        # add standard schema columns onto output
        biz_std_df = self._add_std_cols(biz_df, id_column, name_column, local_threshold)

        return biz_std_df

    def _get_by_code_gis(self, category_code: [str, list], code_type: str = 'NAICS', name_column: str = 'CONAME',
                         id_column: str = 'LOCNUM', local_threshold: int = 0) -> pd.DataFrame:
        """Web GIS implementation for get by code."""
        # get the area of interest
        aoi_df = self._data

        # check the code type to ensure it is valid
        code_type = code_type.upper()
        assert code_type in ['NAICS', 'SIC'], 'code_type must be either "NAICS" or "SIC"'

        # get the businesses
        if code_type == 'NAICS':
            biz_df = get_businesses_gis(aoi_df, self.source, code_naics=category_code)
        else:
            biz_df = get_businesses_gis(aoi_df, self.source, code_sic=category_code)

        # tweak the schema for output
        biz_std_df = self._add_std_cols(biz_df, id_column, name_column, local_threshold)

        return biz_std_df

    def _get_by_name_local(self, business_name: str, name_column: str = 'CONAME',
                           id_col: str = 'LOCNUM') -> pd.DataFrame:
        """Local implementation for get by name."""
        # get the area of interest
        aoi_df = self._data

        # select by attributes
        sql = f"UPPER({name_column}) LIKE UPPER('%{business_name}%')"

        # get using the query and aoi
        biz_df = self._local_get_by_attribute_and_aoi(sql, aoi_df)

        # add standard schema columns onto output
        biz_std_df = self._add_std_cols(biz_df, id_col, name_column)

        return biz_std_df

    def _get_by_name_gis(self, business_name: str, name_column: str = 'CONAME', id_col: str = 'LOCNUM') -> pd.DataFrame:
        """Web GIS implementation for get by name."""
        # get the area of interest
        aoi_df = self._data

        # get businesses from the Web GIS
        biz_df = get_businesses_gis(aoi_df, self.source, business_name)

        # standardize the output
        biz_std_df = self._add_std_cols(biz_df, id_col, name_column)

        return biz_std_df

    def _get_competition_local(self, brand_businesses: [str, pd.DataFrame], name_column: str = 'CONAME',
                               code_column: str = 'NAICS', id_column: str = 'LOCNUM',
                               local_threshold: int = 0) -> pd.DataFrame:
        """Local implementation for get competition."""
        # get the area of interest
        aoi_df = self._data

        # create a layer for selection
        aoi_lst = geography_iterable_to_arcpy_geometry_list(aoi_df, 'polygon')
        aoi_fc = arcpy.management.CopyFeatures(aoi_lst, 'memory/tmp_aoi')[0]
        aoi_lyr = arcpy.management.MakeFeatureLayer(aoi_fc)[0]

        # if the input is a string, create a layer of the brand to use for filtering
        if isinstance(brand_businesses, str):

            # create sql query for identifying the features we are interested in
            brnd_sql = f"UPPER({name_column}) LIKE UPPER('%{brand_businesses}%')"

            # copy the layer local layer instance
            brnd_lyr = self._get_arcpy_lyr(brnd_sql)

            # refine the selection to only the features in the area of interest
            arcpy.management.SelectLayerByLocation(brnd_lyr, select_features=aoi_lyr,
                                                   selection_type='SUBSET_SELECTION')

            # get a list of category codes from the dataset
            cd_lst = pd.Series(r[0] for r in arcpy.da.SearchCursor(brnd_lyr, code_column))

        # if if is a dataframe, create a layer from the input business data
        elif isinstance(brand_businesses, pd.DataFrame):

            # get all the category codes
            cd_lst = brand_businesses[code_column]

            # create an arcpy layer from the dataframe
            brnd_geo_lst = geography_iterable_to_arcpy_geometry_list(brand_businesses)
            brnd_fc = arcpy.management.CopyFeatures(brnd_geo_lst, 'memory/tmp_brnd')[0]
            brnd_lyr = arcpy.management.MakeFeatureLayer(brnd_fc)[0]

        # get the top n category codes by retaining only those describing more than 50% of the brand locations
        cd_vals = get_top_codes(cd_lst)

        # combine the retrieved codes into a concise sql expression
        cat_sql = ' OR '.join([f"{code_column} = '{cd}'" for cd in cd_vals])

        # use the category sql along with the area of interest to get a layer with all in business category code
        cat_lyr = self._get_arcpy_lyr(cat_sql)

        # select features within the area of interest
        cat_aoi_lyr = arcpy.management.SelectLayerByLocation(cat_lyr, overlap_type='INTERSECT',
                                                             select_features=aoi_lyr,
                                                             selection_type='SUBSET_SELECTION')[0]

        # deselect the features in the category layer that are the same as the brand layer to leave only competitors
        comp_lyr = arcpy.management.SelectLayerByLocation(cat_aoi_lyr, overlap_type='INTERSECT',
                                                          select_features=brnd_lyr,
                                                          selection_type='REMOVE_FROM_SELECTION')[0]

        # convert the layer to a spatially enabled dataframe in WGS84
        comp_df = GeoAccessor.from_featureclass(comp_lyr).dm.project(4326)

        # add standard schema columns onto output
        biz_std_df = self._add_std_cols(comp_df, id_column, name_column, local_threshold)

        return biz_std_df

    def _get_competition_gis(self, brand_businesses: [str, pd.DataFrame], name_column: str = 'CONAME',
                             code_column: str = 'NAICS', id_column: str = 'LOCNUM',
                             local_threshold: int = 0) -> pd.DataFrame:
        """Web GIS implementation for get_competition"""
        # get the area of interest
        aoi_df = self._data

        # prepare brand businesses
        if isinstance(brand_businesses, str):

            # search businesses to get brand businesses to use for locating the competition
            brnd_df = aoi_df.mdl.business._get_by_name_gis(brand_businesses)

        # otherwise, if (hopefully) a Spatially Enabled DataFrame
        elif isinstance(brand_businesses, pd.DataFrame):

            # make sure the dataframe is Spatially Enabled - checking for name MUCH faster than validating
            assert brand_businesses.spatial.name is not None, 'brand_business DataFrame must be a Spatially Enabled ' \
                                                              'DataFrame. You may need to run ' \
                                                              'df.spatial.set_geometry("SHAPE") for the GeoAccessor ' \
                                                              'to recognize the spatial column.'

            # simply reassign to dataframe variable
            brnd_df = brand_businesses

        # catch anything else
        else:
            raise Exception('"brand_business" must be either a string describing the input name, or a Spatially Enabled'
                            f'DataFrame of the brand locations - not {type(brand_businesses)}.')

        # get a series of codes from the brand dataframe
        cd_lst = brnd_df[code_column]

        # get the codes comprising the majority of codes describing the brand locations
        top_cd_lst = get_top_codes(cd_lst)

        # use these top codes to get the locations in the area of interest matching these codes
        code_df = aoi_df.mdl.business._get_by_code_gis(top_cd_lst, code_type=code_column,
                                                       local_threshold=local_threshold)

        # remove brand locations from the result based on the unique identifier column
        comp_df = code_df[~code_df['location_id'].isin(brnd_df['location_id'])].copy().reset_index(drop=True)

        # add standard schema columns onto output
        biz_std_df = self._add_std_cols(comp_df, id_column, name_column, local_threshold)

        return biz_std_df
