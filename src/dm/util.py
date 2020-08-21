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


def _set_source(self, in_source:[str, arcgis.gis.GIS]=None) -> [str, arcgis.gis.GIS]:
    """Helper function to check source input."""

    # if string input is provided, ensure setting to local and lowercase
    if isinstance(in_source, str):
        if in_source.lower() != 'local':
            raise Exception(f'Source must be either "local" or a Web GIS instance, not {in_source}.')
        elif in_source.lower() == 'local':
            source = 'local'

    # if nothing provided, default to local if arcpy is available, and remote if arcpy not available
    if in_source is None and arcpy_avail and local_business_analyst:
        source = 'local'

    # TODO: add check for web gis route and enrich active and error if not available
    elif in_source is None and arcgis.env.active_gis:
        source = arcgis.env.active_gis

    # if using local, ensure business analyst is available
    elif source == 'local' and not local_business_analyst:
        raise Exception('Local analysis requires the Business Analyst extension. If you want to use a Web GIS'
                        ' for doing enrichment and routing, please set the source to an instance of'
                        ' arcgis.gis.GIS.')

    # if not using local, use a GIS
    elif isinstance(in_source, arcgis.gis.GIS):
        source = in_source

    return source


def get_countries(source=None) -> pd.DataFrame:
    """Get df of countries available."""
    # TODO: Handle contingency of BA being available, but data not locally installed.
    # TODO: match df schema between local and remote GIS instance

    src = _set_source(source)

    if src is 'local':
        keys = ba_data._get_child_keys(r'SOFTWARE\WOW6432Node\Esri\BusinessAnalyst\Datasets')

        def _get_dataset_info(key):
            name = os.path.basename(key)

            name_parts = name.split('_')

            country = name_parts[0] if name_parts[1] is not None else None
            year = int(name_parts[2]) if name_parts[2] is not None else None

            return name, country, year

        cntry_info_lst = [_get_dataset_info(k) for k in keys]

        return pd.DataFrame(cntry_info_lst, columns=['name', 'country', 'year'])

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