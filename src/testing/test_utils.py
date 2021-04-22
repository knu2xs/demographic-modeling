"""
arcgis.modeling._utils tests
"""
import sys
from pathlib import Path

dir_src = Path(__file__).parent.parent.parent.parent/'src'
assert dir_src.exists()
sys.path.insert(0, str(dir_src))

import modeling

from .fixtures import agol_gis, ent_gis


def test_can_enrich_true(agol_gis):
    usr = agol_gis.users.me
    can_enrich = arcgis.modeling._utils.can_enrich_gis(usr)
    assert can_enrich is True


def test_set_source_local_explicit():
    src = modeling.utils.set_source('local')
    assert src == 'local'


def test_module_avail_false():
    avail = modeling.utils.module_avail('scrapy')
    assert avail is False


def test_avail_arcpy():
    avail = modeling.utils.avail_arcpy
    assert avail is True


def test_local_ba_avail():
    # requires Pro + BA to be installed locally to work correctly
    avail = modeling.utils.local_business_analyst_avail()
    assert avail is True
