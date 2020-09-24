import os.path
import sys

sys.path.insert(0, os.path.abspath('../'))
import dm


def test_env_init_local():

    assert dm.utils.arcpy_avail is True


def test_geography_iterable_to_arcpy_geometry_list():

    pass
    assert dm.utils.geography_iterable_to_arcpy_geometry_list()
