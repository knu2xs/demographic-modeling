import os.path
from pathlib import Path
import sys

from arcgis.features import GeoAccessor
import arcpy

sys.path.insert(0, os.path.abspath('../'))
import dm


def test_env_init_local():

    assert dm.utils.arcpy_avail is True


def test_geography_iterable_to_arcpy_geometry_list():

    pass
    assert dm.utils.geography_iterable_to_arcpy_geometry_list()


def test_add_enrich_aliases():
    if dm.utils.env.arcpy_avail:
        cntry = dm.Country('USA')
        query_str = "ID IN ('530670117102','530770034001','530150009003','530770008001','530050108033','530459613002')"
        bg_df = cntry.level(0).get(query_string=query_str)
        bg_fc = bg_df.spatial.to_featureclass(Path(arcpy.env.scratchGDB)/'bg_test')
        dm.utils.add_enrich_aliases(bg_fc, cntry)
        fld_lst = arcpy.ListFields(bg_fc)
        assert fld_lst
    else:
        assert False, 'This test requires arcpy and USA local data.'
