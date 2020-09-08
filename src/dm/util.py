import importlib.util
import os

import arcgis
from ba_tools import data as ba_data
import pandas as pd

from ._registry import get_ba_usa_key_str

# run some checks to see what is available
arcpy_avail = True if importlib.util.find_spec("arcpy") else False
if arcpy_avail:
    import arcpy

if arcpy_avail:
    local_business_analyst = True if arcpy.CheckExtension('Business') else False
else:
    local_business_analyst = False


def set_source(in_source:[str, arcgis.gis.GIS]=None) -> [str, arcgis.gis.GIS]:
    """
    Helper function to check source input. The source can be set explicitly, but if nothing is provided, it is
        assumes the order of local first and then a Web GIS. Along the way, it also checks to see if a """

    # if string input is provided, ensure setting to local and lowercase
    if isinstance(in_source, str):
        if in_source.lower() != 'local':
            raise Exception(f'Source must be either "local" or a Web GIS instance, not {in_source}.')

        # TODO: add check for local business analyst data
        elif in_source.lower() == 'local' and not local_business_analyst:
            raise Exception(f'If using local source, the Business Analyst extension must be available')

        elif in_source.lower() == 'local':
            source = 'local'

    # if nothing provided, default to local if arcpy is available, and remote if arcpy not available
    if in_source is None and arcpy_avail and local_business_analyst:
        source = 'local'

    # TODO: add check if web gis routing and enrich active - error if not available
    elif in_source is None and arcgis.env.active_gis:
        source = arcgis.env.active_gis

    # if not using local, use a GIS
    elif isinstance(in_source, arcgis.gis.GIS):
        source = in_source

    return source


def get_countries(source=None) -> pd.DataFrame:
    """Get df of countries available."""
    # TODO: Handle contingency of BA being available, but data not locally installed.
    # TODO: match df schema between local and remote GIS instance

    src = set_source(source)

    if src is 'local':
        keys = ba_data._get_child_keys(r'SOFTWARE\WOW6432Node\Esri\BusinessAnalyst\Datasets')

        def _get_dataset_info(key):
            name = os.path.basename(key)

            name_parts = name.split('_')

            country = name_parts[0] if name_parts[1] is not None else None
            year = int(name_parts[2]) if name_parts[2] is not None else None

            return name, country, year

        cntry_info_lst = [_get_dataset_info(k) for k in keys]

        return pd.DataFrame(cntry_info_lst, columns=['geographic_level', 'country', 'year'])

    # TODO: return countries available from GIS object instance


def set_pro_to_usa_local():
    """
    Set the environment setting to ensure using locally installed local data.
    :return: Boolean indicating if data correctly enriched.
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
                f'Your selector, "{geo_in}", is beyond the maximum range of available geographies.')
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


def get_lyr_flds_from_geo_df(df_geo:pd.DataFrame, geo:str, query_str:str=None):
    """
    Get a local feature layer for a geographic level optionally applying a query to filter results.

    Args:
        df_geo: Pandas DataFrame of available local resources.
        geo: Name of geographic level.
        query_str: Optional query string to filter results.

    Returns: Tuple containing an Arcpy FeatureLayer with optional query applied as a definition query filtering results,
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
