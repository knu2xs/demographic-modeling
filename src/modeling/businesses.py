from typing import Union
from warnings import warn

from arcgis.gis import GIS
from arcgis.features import GeoAccessor, FeatureSet
from arcgis.geometry import Geometry, SpatialReference
import pandas as pd
import numpy as np

from .utils import avail_arcpy, local_vs_gis, geography_iterable_to_arcpy_geometry_list, validate_spatial_reference, \
    get_spatially_enabled_dataframe

if avail_arcpy:
    import arcpy
    from ._xml_interrogation import get_business_points_data_path


def get_top_codes(codes: Union[pd.Series, list, tuple], threshold=0.5) -> list:
    """Get the top category codes by only keeping those compromising 50% or greater of the records.

    Args:
        codes: Iterable, preferable a Pandas Series, of code values to filter.
        threshold: Decimal value representing the proportion of values to use for creating
            the list of top values.

    Returns:
        List of unique code values.
    """
    # check the threshold to ensure it is deicmal
    assert 0 < threshold < 1, f'"threshold" must be a decimal value between zero and one, not {threshold}'

    # ensure the input codes iterable is a Pandas Series
    cd_srs = codes if isinstance(codes, pd.Series) else pd.Series(codes)

    # get the instance count for each unique code value
    cnt_df = cd_srs.value_counts().to_frame('cnt')

    # calculate the percent each of the codes comprises of the total values as a decimal
    cnt_df['pct'] = cnt_df.cnt.apply(lambda x: x / cnt_df.cnt.sum())

    # calculate a running total of the percent for each value (running total percent)
    cnt_df['pct_cumsum'] = cnt_df['pct'].cumsum()

    # finally, get the values comprising the code values for the top threshold
    cd_vals = list(cnt_df[(cnt_df['pct_cumsum'] < threshold) | (cnt_df['pct'] > threshold)].index)

    return cd_vals


def _preproces_code_inputs(codes):
    """helper funtion to preprocess naics or sic codes"""
    if isinstance(codes, (str, int)):
        codes = [codes]

    elif isinstance(codes, (pd.Series, list, tuple, np.ndarray)):
        codes = [str(cd) if isinstance(cd, int) else cd for cd in codes]

    return codes


def get_businesses_gis(area_of_interest: pd.DataFrame, gis: GIS, search_string: str = None,
                       code_naics: Union[str, list] = None,
                       code_sic: Union[str, list] = None, exclude_headquarters: bool = True, country: object = None,
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

    # ensure

    # populate the rest of the search parameters
    params['searchstring'] = search_string
    params['businesstypefilters'] = [
        {'Classification': 'NAICS', 'Codes': _preproces_code_inputs(code_naics)},
        {'Classification': 'SIC', 'Codes': _preproces_code_inputs(code_sic)}
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
    res_df = res_df[cols]

    # if not wanting to keep headquarters, normally the case for forecasting modeling, filter them out
    if exclude_headquarters:
        res_df = res_df[~res_df['HDBRCHDESC'].str.lower().str.match('headquarters')].reset_index(drop=True)

    # drop the headquarters or branch column since only used to filter if necessary
    res_df.drop(columns='HDBRCHDESC', inplace=True)

    return res_df


class Business(object):
    """
    Just like it sounds, this is a way to search for and find
    businesses of your own brand for analysis, but more importantly
    competitor locations facilitating modeling the effects of
    competition as well. While the Business object can be instantiated
    directly, it is much easier to simply instantiate from a Country
    object instance.

    .. code-block::python

        from modeling import Country

        # start by creating a country object instance
        usa = Country('USA', source='local')

        # get a geography to work with from locally installed data
        aoi_df = usa.cbsas.get('seattle')

        # get all competitors for Ace Hardware in Seattle
        comp_df = usa.business.get_competition('Ace Hardware', aoi_df)

    """

    def __init__(self, country):
        self._cntry = country
        self.source = country.source

    def __repr__(self):
        """What to show when representing an instance of the object."""
        return f'<dm.Business in {self._cntry.geo_name} ({self.source})>'

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
        biz_df['id'] = biz_df[id_col]
        biz_df['brand_name'] = biz_df[name_col]

        # calculate the brand name category column
        biz_df = self.calculate_brand_name_category(biz_df, local_threshold)

        return biz_df

    @staticmethod
    def calculate_brand_name_category(business_dataframe: pd.DataFrame, local_threshold: int = 0,
                                      inplace=False) -> Union[pd.DataFrame, None]:
        """
        For the output of any Business.get* function, calculate a column named 'brand_name_category'. This function is
        frequently used to re-calculate the category identifying unique local retailers, and group them collectively
        into a 'local_brand'. This is useful in markets where there is a distinct preference for local retailers. This
        is particularly true for speciality coffee shops in many urban markets. While this is performed
        automatically for the 'get_by_code' and 'get_competitors' methods, this function enables you to recalculate it
        if you need to massage some of the brand name outputs.

        Args:
            business_dataframe: Pandas Spatially Enabled DataFrame output from one of the Business.get* functions.
            local_threshold: Integer count below which a brand name will be consider a local brand.
            inplace: Boolean indicating if the dataframe should be modified in place, or a new one created and returned.
                The default is False to not inadvertently

        Returns:
            Pandas Spatially Enabled DataFrame of store locations with the updated column if inplace is False.
            Otherwise, returns None.
        """
        assert 'brand_name' in business_dataframe.columns, 'The "brand_name" column was not found in the input. ' \
                                                           'Please ensure the input is the output from a ' \
                                                           'Business.get* function.'

        # get the unique values below the threshold
        local_srs = business_dataframe['brand_name'].value_counts() > local_threshold
        brand_names = local_srs[local_srs].index.values

        # if not inplace, need to copy the dataframe
        biz_df = business_dataframe if inplace else business_dataframe.copy()

        # calculate the local_brand records based on the stated threshold
        # biz_df['brand_name_category'] = biz_df.brand_name.apply(lambda v: v if v in brand_names else 'local_brand')
        biz_df.loc[biz_df['brand_name'].isin(brand_names), 'brand_name_category'] = 'local_brand'

        return business_dataframe

    @local_vs_gis
    def get_by_name(self, area_of_interest: (pd.DataFrame, pd.Series, Geometry, list),
                    business_name: str, name_column: str = 'CONAME', id_column: str = 'LOCNUM',
                    local_threshold: int = 0) -> pd.DataFrame:
        """
        Search business listings for a specific business name string.

        Args:
            area_of_interest:
                Geometry delineating the area of interest to search for businesses.
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

            from dm import Country

            # start by creating a country object instance
            usa = Country('USA', source='local')

            # get a geography to work with from locally installed data
            aoi_df = usa.cbsas.get('seattle')

            # get all Ace Hardware locations in Seattle
            comp_df = usa.business.get_by_name('Ace Hardware', aoi_df)

        """
        pass

    @local_vs_gis
    def get_by_code(self, area_of_interest: [pd.DataFrame, pd.Series, Geometry, list],
                    category_code: [str, list], code_type: str = 'NAICS', name_column: str = 'CONAME',
                    id_column: str = 'LOCNUM', local_threshold: int = 0) -> pd.DataFrame:
        """
        Search for businesses based on business category code. In North America, this typically is either the NAICS or
        SIC code.

        Args:
            area_of_interest: Required
                Geographic area to search business listings for businesses in the category.
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

            from dm import Country

            # start by creating a country object instance
            usa = Country('USA', source='local')

            # get a geography to work with from locally installed data
            aoi_df = usa.cbsas.get('seattle')

            # get all the businesses for a NAICS code category
            naics_big_df = usa.business.get_by_code(4441, aoi_df, local_threshold=2)

        """
        pass

    @local_vs_gis
    def get_competition(self, area_of_interest: [pd.DataFrame, pd.Series, Geometry, list],
                        brand_businesses: pd.DataFrame, code_column: str = 'NAICS', name_column: str = 'CONAME',
                        id_column: str = 'LOCNUM', local_threshold: int = 0) -> pd.DataFrame:
        """
        Get competitors from previously retrieved business listings.

        Args:
            area_of_interest:
                Geographic area to search business listings for competitors.
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

            from dm import Country

            # start by creating a country object instance
            usa = Country('USA', source='local')

            # get a geography to work with from locally installed data
            aoi_df = usa.cbsas.get('seattle')

            # get all competitors for Ace Hardware in Seattle
            comp_df = usa.business.get_competition('Ace Hardware', aoi_df)

        """
        pass

    def _get_by_code_local(self, area_of_interest: [pd.DataFrame, pd.Series, Geometry, list],
                           category_code: [str, list], code_type: str = 'NAICS', name_column: str = 'CONAME',
                           id_column: str = 'LOCNUM', local_threshold: int = 0) -> pd.DataFrame:
        """Local implementation for get by code."""
        # if the code was input as a number, convert to a string
        category_code = str(category_code) if isinstance(category_code, int) else category_code

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
        biz_df = self._local_get_by_attribute_and_aoi(sql, area_of_interest)

        # add standard schema columns onto output
        biz_std_df = self._add_std_cols(biz_df, id_column, name_column, local_threshold)

        return biz_std_df

    def _get_by_code_gis(self, area_of_interest: [pd.DataFrame, pd.Series, Geometry, list],
                         category_code: [str, list], code_type: str = 'NAICS', name_column: str = 'CONAME',
                         id_column: str = 'LOCNUM', local_threshold: int = 0) -> pd.DataFrame:
        """Web GIS implementation for get by code."""
        # ensure the AOI is a spatially enabled dataframe
        aoi_df = get_spatially_enabled_dataframe(area_of_interest)

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

    def _get_by_name_local(self, area_of_interest: [pd.DataFrame, pd.Series, Geometry, list],
                           business_name: str, name_column: str = 'CONAME', id_col: str = 'LOCNUM') -> pd.DataFrame:
        """Local implementation for get by name."""
        # select by attributes
        sql = f"UPPER({name_column}) LIKE UPPER('%{business_name}%')"

        # get using the query and aoi
        biz_df = self._local_get_by_attribute_and_aoi(sql, area_of_interest)

        # add standard schema columns onto output
        biz_std_df = self._add_std_cols(biz_df, id_col, name_column)

        return biz_std_df

    def _get_by_name_gis(self, area_of_interest: [pd.DataFrame, pd.Series, Geometry, list],
                         business_name: str, name_column: str = 'CONAME', id_col: str = 'LOCNUM') -> pd.DataFrame:
        """Web GIS implementation for get by name."""
        # make sure the area of interest is a spatially enabled dataframe
        aoi_df = get_spatially_enabled_dataframe(area_of_interest)

        # get businesses from the Web GIS
        biz_df = get_businesses_gis(aoi_df, self.source, business_name)

        # standardize the output
        biz_std_df = self._add_std_cols(biz_df, id_col, name_column)

        return biz_std_df

    def _get_competition_local(self, area_of_interest: [pd.DataFrame, pd.Series, Geometry, list],
                               brand_businesses: [str, pd.DataFrame], name_column: str = 'CONAME',
                               code_column: str = 'NAICS', id_column: str = 'LOCNUM',
                               local_threshold: int = 0) -> pd.DataFrame:
        """Local implementation for get competition."""
        # create a layer for selection
        aoi_lst = geography_iterable_to_arcpy_geometry_list(area_of_interest, 'polygon')
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

    def _get_competition_gis(self, area_of_interest: [pd.DataFrame, pd.Series, Geometry, list],
                             brand_businesses: [str, pd.DataFrame], name_column: str = 'CONAME',
                             code_column: str = 'NAICS', id_column: str = 'LOCNUM',
                             local_threshold: int = 0) -> pd.DataFrame:
        """Web GIS implementation for get_competition"""

        # prepare brand businesses
        if isinstance(brand_businesses, str):

            # search businesses to get brand businesses to use for locating the competition
            brnd_df = self._get_by_name_gis(brand_businesses, area_of_interest)

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
        code_df = self.get_by_code(top_cd_lst, area_of_interest, code_type=code_column, local_threshold=local_threshold)

        # remove brand locations from the result based on the unique identifier column
        comp_df = code_df[~code_df['id'].isin(brnd_df['id'])].copy().reset_index(drop=True)

        # add standard schema columns onto output
        biz_std_df = self._add_std_cols(comp_df, id_column, name_column, local_threshold)

        return biz_std_df
