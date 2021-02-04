from arcgis.geometry import Geometry
import pandas as pd

from arcgis.features import GeoAccessor

#from .country import DemographicModeling  # so 'dm' pd.DataFrame accessor works
# from .country import Country
from .utils import arcpy_avail, local_vs_gis, geography_iterable_to_arcpy_geometry_list
# from ._modify_geoaccessor import GeoAccessorIO as GeoAccessor
from ._xml_interrogation import get_business_points_data_path

if arcpy_avail:
    import arcpy


class Business(object):

    """
    Just like it sounds, this is a way to search for and find
    businesses of your own brand for analysis, but more importantly
    competitor locations facilitating modeling the effects of
    competition as well. While the Business object can be instantiated
    directly, it is much easier to simply instantiate from a Country
    object instance.

    .. code-block::python

        from dm import Country, DemographicModeling

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

    @staticmethod
    def _add_std_cols(biz_df: pd.DataFrame, id_col: str, name_col: str, local_threshold: int = 0) -> pd.DataFrame:
        """Helper function adding values in a standard column making follow on analysis workflows easier."""
        # assign the location id and brand name to standardized columns
        biz_df['id'] = biz_df[id_col]
        biz_df['brand_name'] = biz_df[name_col]

        # assign brand name category, identifying local brands based on the count of stores
        local_srs = biz_df.brand_name.value_counts() > local_threshold
        brand_names = local_srs[local_srs == True].index.values
        biz_df['brand_name_category'] = biz_df.brand_name.apply(
            lambda val: val if val in brand_names else 'local_brand')

        return biz_df

    @local_vs_gis
    def get_by_name(self, business_name: str,
                    area_of_interest: (pd.DataFrame, pd.Series, Geometry, list),
                    name_column: str = 'CONAME', id_column: str = 'LOCNUM', local_threshold: int = 0) -> pd.DataFrame:
        """
        Search business listings for a specific business name string.

        Args:
            business_name:
                String business name to search for.
            area_of_interest:
                Geometry delineating the area of interest to search for businesses.
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
    def get_by_code(self, category_code: str, area_of_interest: [pd.DataFrame, pd.Series, Geometry, list],
                    code_column: str = 'NAICS', name_column: str = 'CONAME', id_column: str = 'LOCNUM',
                    local_threshold: int = 0) -> pd.DataFrame:
        """
        Search for businesses based on business category code. In North America, this typically is either the NAICS or
        SIC code.

        Args:
            category_code: Required
                Business category code, such as 4568843, input as a string. This does not have to be a
                complete code. The tool will search for the category code with a partial code starting from the
                beginning.
            area_of_interest: Required
                Geographic area to search business listings for businesses in the category.
            code_column: Optional
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
    def get_competition(self, brand_businesses: pd.DataFrame,
                        area_of_interest: [pd.DataFrame, pd.Series, Geometry, list],
                        code_column: str = 'NAICS', name_column: str = 'CONAME',
                        id_column: str = 'LOCNUM', local_threshold: int = 0) -> pd.DataFrame:
        """
        Get competitors from previously retrieved business listings.

        Args:
            brand_businesses:
                Previously retrieved business listings.
            area_of_interest:
                Geographic area to search business listings for competitors.
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

    def _get_by_code_local(self, category_code: [str, list],
                           area_of_interest: [pd.DataFrame, pd.Series, Geometry, list],
                           code_column: str = 'NAICS', name_column: str = 'CONAME',
                           id_column: str = 'LOCNUM', local_threshold: int = 0) -> pd.DataFrame:
        """Local implementation for get by code."""
        # if the code was input as a number, convert to a string
        category_code = str(category_code) if isinstance(category_code, int) else category_code

        # create the sql query based on whether a single or multiple codes were submitted
        if isinstance(category_code, str):
            sql = f"{code_column} LIKE '{category_code}%'"
        elif isinstance(category_code, list):
            sql_lst = [f"{code_column} LIKE '{cd}%'" for cd in category_code]
            sql = ' OR '.join(sql_lst)
        else:
            raise Exception("The category code must be either a string or list of strings, not "
                            f"'{type(category_code)}'.")

        # get using the query and aoi
        biz_df = self._local_get_by_attribute_and_aoi(sql, area_of_interest)

        # add standard schema columns onto output
        biz_std_df = self._add_std_cols(biz_df, id_column, name_column, local_threshold)

        return biz_std_df

    def _get_by_name_local(self, business_name: str,
                           area_of_interest: [pd.DataFrame, pd.Series, Geometry, list],
                           name_column: str = 'CONAME', id_col: str = 'LOCNUM') -> pd.DataFrame:
        """Local implementation for get by name."""
        # select by attributes
        sql = f"UPPER({name_column}) LIKE UPPER('%{business_name}%')"

        # get using the query and aoi
        biz_df = self._local_get_by_attribute_and_aoi(sql, area_of_interest)

        # add standard schema columns onto output
        biz_std_df = self._add_std_cols(biz_df, id_col, name_column)

        return biz_std_df

    def _get_competition_local(self, brand_businesses: [str, pd.DataFrame],
                               area_of_interest: [pd.DataFrame, pd.Series, Geometry, list],
                               name_column: str = 'CONAME',
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
        cnt_df = cd_lst.value_counts().to_frame('cnt')
        cnt_df['pct'] = cnt_df.cnt.apply(lambda x: x / cnt_df.cnt.sum())
        cnt_df['pct_cumsum'] = cnt_df['pct'].cumsum()
        cd_vals = cnt_df[(cnt_df['pct_cumsum'] < 0.5) | (cnt_df['pct'] > 0.5)].index.values

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
