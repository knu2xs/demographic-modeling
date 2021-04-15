"""Provide modeling accessor object namespace and methods."""
import math
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Union, Optional
from warnings import warn

from arcgis.features import GeoAccessor, FeatureSet
from arcgis.features.geo._internals import register_dataframe_accessor
from arcgis.gis import GIS
from arcgis.geometry import Geometry, SpatialReference
from arcgis.network import ClosestFacilityLayer

from .country import Country, GeographyLevel
from .utils import avail_arcpy, local_vs_gis, geography_iterable_to_arcpy_geometry_list, validate_spatial_reference, \
    get_top_codes, preproces_code_inputs, has_networkanalysis_gis, LocalNetworkEnvironment

if avail_arcpy:
    import arcpy
    from ._xml_interrogation import get_business_points_data_path
    from ._registry import get_ba_key_value


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
        self.proximity = Proximity(self)

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
                                      inplace: bool = False) -> Union[pd.DataFrame, None]:
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
            comp_df = aoi_df.mdl.business.get_competition(brand_df, local_threshold=3)

            # with hardware stores, each True Value has a unique name,
            # so it helps to rename these to be correctly recognized
            # as a brand of stores
            replace_lst = [
                ('TRUE VALUE|TRUE VL', 'TRUE VALUE'),
                ('MC LENDON|MCLENDON', 'MCLENDON HARDWARE')
            ]
            for repl in replace_lst:
                brand_filter = comp_df.brand_name.str.contains(repl[0], regex=True)
                comp_df.loc[brand_filter, 'brand_name'] = repl[1]

            # now, with the True Values renamed, we need to recalculate which
            # locations are actually local brands
            comp_df.mdl.business.calculate_brand_name_category(3, inplace=True)

        The output of ``comp_df.head()`` from the above sample looks similar to the following.

        ====  =========  ===========================  ===============  ========  ======  =========  ========  ========  ========  ========  =====  =======  ==============================================================================  =============  ======================  =====================
          ..     LOCNUM  CONAME                       NAICSDESC           NAICS     SIC  SOURCE     PUBPRV    FRNCOD    ISCODE    CITY        ZIP  STATE    SHAPE                                                                             location_id  brand_name              brand_name_category
        ====  =========  ===========================  ===============  ========  ======  =========  ========  ========  ========  ========  =====  =======  ==============================================================================  =============  ======================  =====================
           0  002890986  MC LENDON HARDWARE           HARDWARE-RETAIL  44413005  525104  INFOGROUP                                SUMNER    98390  WA       {'x': -122.242365, 'y': 47.2046040000001, 'spatialReference': {'wkid': 4326}}       002890986  MCLENDON HARDWARE       MCLENDON HARDWARE
           1  006128854  MCLENDON HARDWARE INC        HARDWARE-RETAIL  44413005  525104  INFOGROUP                                RENTON    98057  WA       {'x': -122.2140195, 'y': 47.477943, 'spatialReference': {'wkid': 4326}}             006128854  MCLENDON HARDWARE       MCLENDON HARDWARE
           2  174245191  DUVALL TRUE VALUE HARDWARE   HARDWARE-RETAIL  44413005  525104  INFOGROUP            2                   DUVALL    98019  WA       {'x': -121.9853835, 'y': 47.738907, 'spatialReference': {'wkid': 4326}}             174245191  TRUE VALUE              TRUE VALUE
           3  174262691  GATEWAY TRUE VALUE HARDWARE  HARDWARE-RETAIL  44413005  525104  INFOGROUP            2                   ENUMCLAW  98022  WA       {'x': -121.9876155, 'y': 47.2019940000001, 'spatialReference': {'wkid': 4326}}      174262691  TRUE VALUE              TRUE VALUE
           4  174471722  TWEEDY & POPP HARDWARE       HARDWARE-RETAIL  44413005  525104  INFOGROUP            2                   SEATTLE   98103  WA       {'x': -122.3357134, 'y': 47.6612959300001, 'spatialReference': {'wkid': 4326}}      174471722  TWEEDY & POPP HARDWARE  local_brand
        ====  =========  ===========================  ===============  ========  ======  =========  ========  ========  ========  ========  =====  =======  ==============================================================================  =============  ======================  =====================

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

        # make sure the geometry is correctly set
        biz_df.spatial.set_geometry('SHAPE')

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

        # tack on the country for potential follow on analysis
        biz_std_df.attrs['_cntry'] = aoi_df.attrs['_cntry']

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

        # tack on the country for potential follow on analysis
        biz_std_df.attrs['_cntry'] = aoi_df.attrs['_cntry']

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

        # tack on the country for potential follow on analysis
        biz_std_df.attrs['_cntry'] = aoi_df.attrs['_cntry']

        return biz_std_df

    def _get_by_name_gis(self, business_name: str, name_column: str = 'CONAME', id_col: str = 'LOCNUM') -> pd.DataFrame:
        """Web GIS implementation for get by name."""
        # get the area of interest
        aoi_df = self._data

        # get businesses from the Web GIS
        biz_df = get_businesses_gis(aoi_df, self.source, business_name)

        # standardize the output
        biz_std_df = self._add_std_cols(biz_df, id_col, name_column)

        # tack on the country for follow on analysis
        biz_std_df.attrs['_cntry'] = aoi_df.attrs['_cntry']

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

        # tack on the country for potential follow on analysis
        biz_std_df.attrs['_cntry'] = aoi_df.attrs['_cntry']

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

        # ensure valid spatially enabled dataframe
        biz_std_df.spatial.set_geometry('SHAPE')

        # tack on the country for potential follow on analysis
        biz_std_df.attrs['_cntry'] = aoi_df.attrs['_cntry']

        return biz_std_df


## Proximity analysis section
def get_travel_modes(source: Union[Path, GIS, Country], all_properties=False):
    """
    Retrieve travel modes for the specified routing source.
    Args:
        source: Path to an ArcGIS Network Dataset if working in an environment with
            ArcGIS Pro providing access to ``arcpy``. If using a connection to a
            Web GIS thorough a GIS object instance, the GIS object instance.
            Finally, if using a Country object instance, the Country.
        all_properties: Get all available properties for all the travel modes.
            Default is False.

    Returns:
        Dataframe of available travel modes and descriptions for each.
    """
    # if the source is a country, extract the source from the country
    if isinstance(source, Country):
        source = source.source

    # if a string, make sure simply set to local, and make sure arcpy is available
    if isinstance(source, str):
        source = source.lower()
        assert source == 'local', 'If intending to use local resources for analysis, you must use the "local" ' \
                                  f'keyword. You provided, "{source}".'
        assert avail_arcpy, 'To use local resources for routing, you must be using an environment with arcpy ' \
                            'available (ArcGIS Pro installed).'

        raise NotImplementedError('Retrieving travel modes not yet supprted for "local" analysis.')

    # otherwise, we SHOULD be dealing with a GIS
    elif isinstance(source, GIS):

        # get the url from the gis properties, retrieve the route properties, and format travel modes into a DataFrame
        prop = source._con.get(source.properties.helperServices.route.url)
        trvl_df = pd.DataFrame(prop['supportedTravelModes'])

        # populate the key for easy lookups
        trvl_df['key'] = trvl_df['name'].str.lower().str.replace(' ', '_')
        trvl_df.set_index('key', inplace=True, drop=True)

        # unless more is desired, keep it simple
        if not all_properties:
            trvl_df = trvl_df.loc[:, ['name', 'type', 'description']]

    else:
        raise Exception(
            f'The source must be either "local", a GIS object, or a Country object, not "{type(source)}".')

    return trvl_df


class Proximity:
    """
    Provides access to proximity calculation functions.
    """

    def __init__(self, mdl: ModelingAccessor):

        self._data = mdl._data

        if '_cntry' in self._data.attrs.keys():
            self._cntry = self._data.attrs['_cntry']
            self.source = self._cntry.source
        else:
            self._cntry, self.source = None, None

        self._travel_modes = None

        if isinstance(self.source, GIS):
            self._properties = self.source._con.get(self.source.properties.helperServices.route.url)
        else:
            self._properties = None

    def __repr__(self):
        if isinstance(self.source, GIS):
            repr = f'<modeling.Proximity - {self.source.__repr__()}>'
        else:
            repr = '<modeling.Proximity>'
        return repr

    @property
    def travel_modes(self) -> pd.DataFrame:
        if self._travel_modes is None:
            self._travel_modes = get_travel_modes(self.source)
        return self._travel_modes

    @staticmethod
    def _prep_sdf_for_nearest(input_dataframe: pd.DataFrame, id_column: str):
        """
        Given an input Spatially Enabled Dataframe, prepare it to work
            well with the nearest solver.

        Args:
            input_dataframe: Spatially Enabled Dataframe with really
                any geometry.
            id_column: Field uniquely identifying each of location to
                be used for routing to nearest.

        Returns: Spatially Enabled Dataframe of points with correct
            columns for routing to nearest.
        """
        # check inputs
        assert isinstance(input_dataframe, pd.DataFrame), f'The input dataframe must be a Pandas DataFrame, not ' \
                                                          f'{type(input_dataframe)}.'

        # ensure the geometry is set
        geom_col_lst = [c for c in input_dataframe.columns if input_dataframe[c].dtype.name.lower() == 'geometry']
        assert len(geom_col_lst) > 0, 'The DataFrame does not appear to have a geometry column defined. This can be ' \
                                      'accomplished using the "df.spatial.set_geometry" method.'
        geom_col = geom_col_lst[0]

        # ensure the column is in the dataframe columns
        assert id_column in input_dataframe.columns, f'The provided id_column, "{id_column}," does not appear to be ' \
                                                     f'in the columns [{", ".join(input_dataframe.columns)}]"'

        # par down the input dataframe to just the columns needed
        input_dataframe = input_dataframe.loc[:, [id_column, geom_col]]

        # rename the columns to follow the schema needed for routing
        input_dataframe.columns = ['ID', 'SHAPE']

        # ensure the spatial reference is WGS84 - if not, make it so
        if input_dataframe.spatial.sr.wkid != 4326:
            input_dataframe = input_dataframe.mdl.project(4326)

        # if the geometry is not points, we still need points, so get the geometric centroids
        if input_dataframe.spatial.geometry_type != ['point']:
            input_dataframe['SHAPE'] = input_dataframe[geom_col].apply(
                lambda geom: Geometry({'x': geom.centroid[0], 'y': geom.centroid[1],
                                       'spatialReference': geom.spatial_reference}))
            input_dataframe.spatial.set_geometry('SHAPE')

        # add a second column for the ID as Name
        input_dataframe['Name'] = input_dataframe['ID']

        # ensure the geometry is correctly being recognized
        input_dataframe.spatial.set_geometry('SHAPE')

        # set the order of the columns and return
        return input_dataframe.loc[:, ['ID', 'Name', 'SHAPE']]

    @staticmethod
    def _get_max_near_dist_arcpy(origin_lyr):
        """Get the maximum geodesic distance between stores."""
        # create a location for temporary data
        temp_table = r'in_memory\near_table_{}'.format(uuid.uuid4().hex)

        # if only one location, cannot generate a near table, and default to 120 miles
        if int(arcpy.management.GetCount(origin_lyr)[0]) <= 1:
            max_near_dist = 120 * 1609.34

        else:
            # use arcpy to get a table of all distances between stores
            near_tbl = arcpy.analysis.GenerateNearTable(
                in_features=origin_lyr,
                near_features=origin_lyr,
                out_table=temp_table,
                method="GEODESIC"
            )[0]

            # get the maximum near distance, which will be in meters
            meters = max([row[0] for row in arcpy.da.SearchCursor(near_tbl, 'NEAR_DIST')])

            # remove the temporary table to ensure not stuff lying around and consuming RAM
            arcpy.management.Delete(temp_table)

            # get the maximum near distance (in meters)
            max_near_dist = meters * 0.00062137

        return max_near_dist

    @staticmethod
    def _get_nearest_solve_local(origin_dataframe: pd.DataFrame, destination_dataframe: pd.DataFrame,
                                 destination_count: int, network_dataset: [Path, str],
                                 maximum_distance: [int, float] = None):
        """
        Perform network solve using local resources with assumption of standard input.

        Args:
            origin_dataframe: Origin points Spatially Enabled Dataframe
            destination_dataframe: Destination points Spatially Enabled Dataframe
            destination_count: Destination points Spatially Enabled Dataframe
            network_dataset: Path to ArcGIS Network dataset for performing routing.
            maximum_distance: Maximum nearest routing distance in miles.

        Returns: Spatially Enabled Dataframe of solved closest facility routes.
        """
        # make sure the path to the network dataset is a string
        network_dataset = str(network_dataset) if isinstance(network_dataset, Path) else network_dataset

        # get the mode of travel from the network dataset - rural so gravel roads are fair game
        nd_lyr = arcpy.nax.MakeNetworkDatasetLayer(network_dataset)[0]
        trvl_mode_dict = arcpy.nax.GetTravelModes(nd_lyr)
        trvl_mode = trvl_mode_dict['Rural Driving Time']

        # create the closest solver object instance
        # https://pro.arcgis.com/en/pro-app/arcpy/network-analyst/closestfacility.htm
        closest_solver = arcpy.nax.ClosestFacility(network_dataset)

        # set parameters for the closest solver
        closest_solver.travelMode = trvl_mode
        closest_solver.travelDirection = arcpy.nax.TravelDirection.ToFacility
        # TODO: How to set this to distance?
        closest_solver.timeUnits = arcpy.nax.TimeUnits.Minutes
        closest_solver.distanceUnits = arcpy.nax.DistanceUnits.Miles
        closest_solver.defaultTargetFacilityCount = destination_count
        closest_solver.routeShapeType = arcpy.nax.RouteShapeType.TrueShapeWithMeasures
        closest_solver.searchTolerance = 5000
        closest_solver.searchToleranceUnits = arcpy.nax.DistanceUnits.Meters

        # since maximum distance is optional, well, make it optional
        if maximum_distance is not None:
            closest_solver.defaultImpedanceCutoff = maximum_distance

        # load the origin and destination feature data frames into memory and load into the solver object instance
        # TODO: test if can use 'memory' workspace instead of scratch
        origin_fc = origin_dataframe.spatial.to_featureclass(os.path.join(arcpy.env.scratchGDB, 'origin_tmp'))
        closest_solver.load(arcpy.nax.ClosestFacilityInputDataType.Incidents, origin_fc)

        dest_fc = destination_dataframe.spatial.to_featureclass(os.path.join(arcpy.env.scratchGDB, 'dest_tmp'))
        closest_solver.load(arcpy.nax.ClosestFacilityInputDataType.Facilities, dest_fc)

        # run the solve, and get comfortable
        closest_result = closest_solver.solve()

        # export the results to a spatially enabled data frame, and do a little cleanup
        # TODO: test if can use 'memory/routes' instead - the more current method
        route_fc = 'in_memory/routes'
        closest_result.export(arcpy.nax.ClosestFacilityOutputDataType.Routes, route_fc)
        route_oid_col = arcpy.Describe(route_fc).OIDFieldName
        closest_df = GeoAccessor.from_featureclass(route_fc)
        arcpy.management.Delete(route_fc)
        if route_oid_col:
            closest_df.drop(columns=[route_oid_col], inplace=True)

        # get rid of the extra empty columns the local network solve adds
        closest_df.dropna(axis=1, how='all', inplace=True)

        # populate the origin and destination fields so the schema matches what online solve returns
        name_srs = closest_df.Name.str.split(' - ')
        closest_df['IncidentID'] = name_srs.apply(lambda val: val[0])
        closest_df['FacilityID'] = name_srs.apply(lambda val: val[1])

        return closest_df

    @staticmethod
    def _reformat_closest_result_dataframe(closest_df: pd.DataFrame):
        """
        Reformat the schema, dropping unneeded columns and renaming those kept to be more in line with this workflow.

        Args:
            closest_df: Dataframe of the raw output routes from the find closest analysis.

        Returns: Spatially Enabled Dataframe reformatted.
        """
        # create a list of columns containing proximity metrics
        proximity_src_cols = [col for col in closest_df.columns if col.startswith('Total_')]

        # if both miles and kilometers, drop miles, and keep kilometers
        miles_lst = [col for col in proximity_src_cols if 'miles' in col.lower()]
        kilometers_lst = [col for col in proximity_src_cols if 'kilometers' in col.lower()]
        if len(miles_lst) and len(kilometers_lst):
            proximity_src_cols = [col for col in proximity_src_cols if col != miles_lst[0]]

        # calculate side of street columns
        closest_df['proximity_side_street_right'] = (closest_df['FacilityCurbApproach'] == 1).astype('int64')
        closest_df['proximity_side_street_left'] = (closest_df['FacilityCurbApproach'] == 2).astype('int64')
        side_cols = ['proximity_side_street_left', 'proximity_side_street_right']

        # filter the dataframe to just the columns we need
        src_cols = ['IncidentID', 'FacilityRank', 'FacilityID'] + proximity_src_cols + side_cols + ['SHAPE']
        closest_df = closest_df[src_cols].copy()

        # replace total in proximity columns for naming convention
        closest_df.columns = [col.lower().replace('total', 'proximity') if col.startswith('Total_') else col
                              for col in closest_df.columns]

        # rename the columns for the naming convention
        rename_dict = {'IncidentID': 'origin_id', 'FacilityRank': 'destination_rank', 'FacilityID': 'destination_id'}
        closest_df = closest_df.rename(columns=rename_dict)

        return closest_df

    @staticmethod
    def _explode_closest_rank_dataframe(closest_df: pd.DataFrame, origin_id_col: str = 'origin_id',
                                        rank_col: str = 'destination_rank',
                                        dest_id_col: str = 'destination_id',
                                        dest_keep_cols: list = None):
        """
        Effectively explode out or pivot the data so there is only a single record for each origin.

        Args:
            closest_df: Spatially Enabled Dataframe reformatted from the raw output of find nearest.
            origin_id_col: Column uniquely identifying each origin - default 'origin_id'
            rank_col: Column identifying the rank of each destination - default 'destination_rank'
            dest_id_col: Column uniquely identifying each destination - default 'destination_id'

        Returns: Dataframe with a single row for each origin with multiple destination metrics for each.
        """
        column_identifying_prefix = 'proximity_'

        # create a dataframe to start working with comprised of only the unique origin_dataframe to start with
        origin_dest_df = pd.DataFrame(closest_df[origin_id_col].unique(), columns=[origin_id_col])

        # get a list of the proximity columns
        proximity_cols = [col for col in closest_df.columns if col.startswith(column_identifying_prefix)]

        # add any destination columns
        if len(dest_keep_cols):
            proximity_cols = proximity_cols + dest_keep_cols

        # iterate the closest destination ranking
        for rank_val in closest_df[rank_col].unique():

            # filter the dataframe to just the records with this destination ranking
            rank_df = closest_df[closest_df[rank_col] == rank_val]

            # create a temporary dataframe to begin building the columns onto
            df_temp = rank_df[origin_id_col].to_frame()

            # iterate the relevant columns
            for col in [dest_id_col] + proximity_cols:

                # create a new column name from the unique value and the original row name
                new_name = f'{col}_{rank_val:02d}'

                # filter the data in the column with the unique value
                df_temp[new_name] = rank_df[col].values

            # set the index to the origin id for joining
            df_temp.set_index(origin_id_col, inplace=True)

            # join the temporary dataframe to the master
            origin_dest_df = origin_dest_df.join(df_temp, on=origin_id_col)

        return origin_dest_df

    def _get_nearest_local(self, origin_dataframe: pd.DataFrame, destination_dataframe: pd.DataFrame,
                           network_dataset: [str, Path], origin_id_column: str = 'LOCNUM',
                           destination_id_column: str = 'LOCNUM', destination_count: int = 4) -> pd.DataFrame:
        """Local implementation of get nearest solution."""
        # check to make sure network analyst is available using the env object to make it simplier
        env = LocalNetworkEnvironment()
        if 'Network' in env.arcpy_extensions:
            env.arcpy_checkout_extension('Network')
        else:
            raise Exception('To perform network routing locally you must have access to the ArcGIS Network Analyst '
                            'extension. It appears this extension is either not installed or not licensed.')

        # ensure the dataframes are in the right schema and have the right geometry (points)
        origin_net_df = self._prep_sdf_for_nearest(origin_dataframe, origin_id_column)
        dest_net_df = self._prep_sdf_for_nearest(destination_dataframe, destination_id_column)

        # run the closest analysis locally
        closest_df = self._get_nearest_solve_local(origin_net_df, dest_net_df, destination_count, network_dataset)

        return closest_df

    def _get_nearest_gis(self, origin_dataframe: pd.DataFrame, destination_dataframe: pd.DataFrame,
                         source: [str, Country, GIS], origin_id_column: str = 'LOCNUM',
                         destination_id_column: str = 'LOCNUM', destination_count: int = 4) -> pd.DataFrame:
        """Web GIS implementation of get nearest solution."""

        # TODO: backport these to be optional input parameters
        return_geometry = True
        output_spatial_reference = 4326

        # populate the correct request parameter for returning geometry
        out_geom = 'esriNAOutputLineTrueShape' if return_geometry else 'esriNAOutputLineNone'

        # build the spatial reference dict
        out_sr = {'wkid': output_spatial_reference}

        # if a source is not explicitly specified, try to get it out of the input data
        source = self.source if not source else source

        # if a country instance, get the GIS object from it
        if isinstance(source, Country):
            assert isinstance(Country.source, GIS), 'The source Country must be reference an ArcGIS Web GIS object ' \
                                                    'instance to solve using a GIS.'
            source = Country.source

        # run a couple of checks to make sure we do not encounter strange errors later
        assert isinstance(source, GIS), 'The source must be a GIS object instance.'
        assert has_networkanalysis_gis(source.users.me), 'You must have the correct permissions in the Web GIS to ' \
                                                         'perform routing solves. It appears you do not.'

        # prep the datasets for routing
        origin_df = self._prep_sdf_for_nearest(origin_dataframe, origin_id_column)
        dest_df = self._prep_sdf_for_nearest(destination_dataframe, destination_id_column)

        # get the url for doing routing
        clst_url = source.properties.helperServices.closestFacility.url

        # call the server for the solve - left in a lot of parameters in case want to expand capability
        res = ClosestFacilityLayer(clst_url, self.source).solve_closest_facility(
            incidents=origin_df,
            facilities=dest_df,
            travel_mode=None,
            attribute_parameter_values=None,
            return_cf_routes=True,
            output_lines=out_geom,
            default_cutoff=None,
            default_target_facility_count=destination_count,
            travel_direction=None,
            out_sr=out_sr,
            accumulate_attribute_names=None,
            impedance_attribute_name=None,
            restriction_attribute_names=None,
            restrict_u_turns=None,
            use_hierarchy=True,
            output_geometry_precision=None,
            output_geometry_precision_units=None,
            time_of_day=None,
            time_of_day_is_utc=None,
            time_of_day_usage=None,
            return_z=False,
            overrides=None,
            preserve_objectid=False,
            future=False,
            ignore_invalid_locations=True
        )

        # unpack the results from the response
        route_df = FeatureSet.from_dict(res['routes']).sdf

        # clean up any empty columns
        notna_srs = route_df.isna().all()
        drop_cols = notna_srs[notna_srs].index.values
        route_df.drop(columns=drop_cols, inplace=True)

        # populate the origin and destination id columns so the output will be as expected
        id_srs = route_df['Name'].str.split(' - ')
        route_df['IncidentID'] = id_srs.apply(lambda val: val[0])
        route_df['FacilityID'] = id_srs.apply(lambda val: val[1])

        return route_df

    def get_nearest(self, destination_dataframe: pd.DataFrame,
                    source: Union[str, Path, Country, GIS] = None, single_row_per_origin: Optional[bool] = True,
                    origin_id_column: Optional[str] = 'LOCNUM', destination_id_column: Optional[str] = 'LOCNUM',
                    destination_count: Optional[int] = 4, near_prefix: Optional[str] = None,
                    destination_columns_to_keep: Union[str, list] = None) -> pd.DataFrame:
        """
        Get nearest enables getting the nth (default is four) nearest locations
        based on drive distance between two Spatially Enabled DataFrames. If the
        origins are polygons, the centroids will be used as the start locations.
        This is useful for getting the nearest store brand locations to every
        origin block group in a metropolitan area along with the nearest
        competition locations to every block group in the same metropolitan area.

        Args:
            destination_dataframe: Destination points in one of the supported input formats.
            source: Either the path to the network dataset, the Country object associated with
                the Business Analyst source being used, or a GIS object instance.
            single_row_per_origin: Optional - Whether or not to pivot the results to return
                only one row for each origin location. Default is True.
            origin_id_column: Optional - Column in the origin points Spatially Enabled Dataframe
                uniquely identifying each feature. Default is 'LOCNUM'.
            destination_id_column: Column in the destination points Spatially Enabled Dataframe
                uniquely identifying each feature
            destination_count: Integer number of destinations to search for from every origin
                point.
            near_prefix: String prefix to prepend onto near column names in the output.
            destination_columns_to_keep: List of columns to keep in the output. Commonly, if
                businesses, this includes the column with the business names.

        Returns:
            Spatially Enabled Dataframe with a row for each origin id, and metrics for
            each nth destinations.

        .. code-block:: python

            from modeling import Country

            brand_name = 'ace hardware'

            # create a country ojbect to work with
            usa = Country('USA')

            # get a metropolitan area, a CBSA, to use as the study area
            aoi_df = usa.cbsas.get('seattle')

            # get the current year key variables to use for enrichment
            evars = usa.enrich_variables
            key_vars = evars[
                (evars.name.str.endswith('CY'))
                & (evars.data_collection.str.lower().str.contains('key'))
            ].reset_index(drop=True)

            # get the block groups and enrich them with the ~20 key variables
            bg_df = aoi_df.mdl.level(0).get().mdl.enrich(key_vars)

            # get the store brand locations and competition locations
            biz_df = aoi_df.mdl.business.get_by_name(brand_name)
            comp_df = aoi_df.mdl.business.get_competition(biz_df)

            # get the nearest three brand locations to every block group
            bg_near_biz = bg_df.mdl.proximity.get_nearest(biz_df,
                origin_id_column='ID', destination_count=3, near_prefix='brand')

            # get the nearest six competition locations to every block group
            bg_near_df = bg_near_biz.mdl.proximity.get_nearest(bg_near_biz,
                origin_id_column='ID', near_prefix='comp', destination_count=6,
                destination_columns_to_keep=['brand_name', 'brand_name_category'])

        """
        # Max GIS batch count
        batch_size = 99

        assert isinstance(destination_dataframe, pd.DataFrame), 'Origin and destination dataframes must both be ' \
                                                                'pd.DataFrames'
        assert destination_dataframe.spatial.validate(), 'Origin and destination dataframes must be valid Spatially ' \
                                                         'enabled DataFrames. This can be checked using ' \
                                                         'df.spatial.validate().'

        if self.source is None:
            if '_cntry' in self._data.attrs.keys():
                self._cntry = self._data.attrs['_cntry']
                self.source = self._cntry.source
            source = self.source
            assert isinstance(source, (str, Path, Country, GIS)), 'source must be either a path to the network ' \
                                                                  'dataset, a modeling.Country object instance, or a ' \
                                                                  'reference to a GIS.'

        assert isinstance(single_row_per_origin, bool)

        assert origin_id_column in self._data.columns, f'The provided origin_id_column does not appear to be in ' \
                                                       f'the origin_dataframe columns ' \
                                                       f'[{", ".join(self._data.columns)}]'

        assert destination_id_column in destination_dataframe.columns, f'The provided destination_id_column does not ' \
                                                                       f'appear to be in the destination_dataframe ' \
                                                                       f'columns ' \
                                                                       f'[{", ".join(destination_dataframe.columns)}]'

        # if the source is a country set to local, we are using Business Analyst, so interrogate the source
        if isinstance(source, Country):

            # if local, get the path to the network dataset
            if source.source == 'local':
                source = get_ba_key_value('StreetsNetwork', source.geo_name)

            # if not local, set the source to the GIS object instance
            else:
                source = source.source

        # if the source is a path, convert it to a string because arcpy doesn't do well with path objects
        source = str(source) if isinstance(source, Path) else source

        # if a path, ensure it exists
        if isinstance(source, str):
            assert arcpy.Exists(source), f'The path to the network dataset provided does not appear to exist - ' \
                                         f'"{str(source)}".'

        # include any columns to be retained in the output
        if destination_columns_to_keep is not None:

            # if just a single column is provided in a string, make it into a list
            if isinstance(destination_columns_to_keep, list):
                dest_cols = destination_columns_to_keep
            else:
                dest_cols = [destination_columns_to_keep]

            # make sure the destination columns include the id columns
            dest_cols = dest_cols if destination_id_column in dest_cols else [destination_id_column] + dest_cols

            # check all the columns to make sure they are in the output dataframe
            for col in dest_cols:
                assert col in destination_dataframe.columns, f'One of the destination_columns_to_keep {col}, does ' \
                                                             f'not appear to be in the destination_dataframe columns ' \
                                                             f'[{", ".join(destination_dataframe.columns)}].'

        # if no columns, just populate an empty list so nested functions work
        else:
            dest_cols = []

        # now, the source is either a path to the network source or a GIS object instance, so call each as necessary
        if isinstance(source, str):
            raw_near_df = self._get_nearest_local(self._data, destination_dataframe, source, origin_id_column,
                                                  destination_id_column, destination_count)

        else:

            # since having issues with over 1,000 origins, break into batches
            batch_cnt = math.ceil(len(self._data.index) / batch_size)

            # iterate through batches and get response dataframes
            near_df_lst = []
            for idx in range(batch_cnt):
                idx_start = batch_size * idx
                idx_end = batch_size * (idx + 1)
                idx_end = idx_end if idx_end < len(self._data) else len(self._data) + 1
                orig_df = self._data.loc[idx_start: idx_end]
                near_df_tmp = self._get_nearest_gis(orig_df, destination_dataframe, source, origin_id_column,
                                                    destination_id_column, destination_count)
                near_df_lst.append(near_df_tmp)

            # consolidate and remove temp df list
            raw_near_df = pd.concat(near_df_lst)
            del near_df_lst

        # reformat and standardize the output
        std_clstst_df = self._reformat_closest_result_dataframe(raw_near_df)

        if dest_cols:
            if len(dest_cols):
                # add the columns onto the near dataframe for output
                dest_join_df = destination_dataframe[dest_cols].set_index(destination_id_column)
                std_clstst_df = std_clstst_df.join(dest_join_df, on='destination_id')

        # pivot and explode the results to be a single row for each origin if desired
        if single_row_per_origin:
            xplod_dest_cols = [c for c in dest_cols if c != destination_id_column]
            near_df = self._explode_closest_rank_dataframe(std_clstst_df, dest_keep_cols=xplod_dest_cols)
        else:
            near_df = std_clstst_df

        # add prefixes to columns if provided
        if near_prefix is not None:
            near_df.columns = [f'{near_prefix}_{c}' for c in near_df.columns]
            near_oid_col = f'{near_prefix}_origin_id'
        else:
            near_oid_col = 'origin_id'

        # add results to input data
        if single_row_per_origin:
            out_df = self._data.join(near_df.set_index(near_oid_col), on=origin_id_column)

        else:
            out_df = near_df.join(self._data.drop(columns='SHAPE').set_index(origin_id_column), on=near_oid_col)
            out_df.columns = [c if not c.endswith('_SHAPE') else 'SHAPE' for c in out_df.columns]

        # shuffle the columns so the geometry is at the end
        if out_df.spatial.name is not None:
            out_df = out_df.loc[:, [c for c in out_df.columns if c != out_df.spatial.name] + [out_df.spatial.name]]
            
        # make sure there are not any duplicates lingering
        out_df.drop_duplicates(origin_id_column, inplace=True)

        # recognize geometry
        out_df.spatial.set_geometry('SHAPE')

        # tack on the attrs for any subsequent modeling analysis
        out_df.attrs = self._data.attrs

        return out_df
