import os.path
import sys

from arcgis.gis import GIS
from arcgis.features import FeatureSet
import pandas as pd
from pathlib import Path
import pytest

sys.path.insert(0, os.path.abspath('../'))
import dm


@pytest.fixture
def usa_local():
    return dm.Country('USA', source='local')


@pytest.fixture
def usa_agol():
    gis = GIS()
    return dm.Country('USA', source=gis)


def test_geography_implicit():
    usa = dm.Country('USA')
    df = usa.geographies
    assert isinstance(df, pd.DataFrame)


def test_create_geography_level(usa_local):
    geo_lvl = dm.country.GeographyLevel('block_groups', usa_local)
    assert isinstance(geo_lvl, dm.country.GeographyLevel)


def test_get_geography_level_int(usa_local):
    geo_lvl = usa_local.level(0)
    assert isinstance(geo_lvl, dm.country.GeographyLevel)


def test_get_local_implicit(usa_local):
    df = usa_local.states.get()
    assert isinstance(df, pd.DataFrame)


def test_get_local_explicit_int(usa_local):
    df = usa_local.level(0)
    assert isinstance(df, dm.country.GeographyLevel)


def test_get_geography_local_implicit_int(usa_local):
    df = usa_local.level(9).get()
    assert isinstance(df, pd.DataFrame)


def test_get_geography_local_subgeography(usa_local):
    cbsa_df = usa_local.cbsas.get('seattle')
    cbsa_dm = cbsa_df.dm
    cntys = cbsa_dm.counties
    df = cntys.get()
    assert len(df.index) == 3


def test_get_geography_local_within_df(usa_local):
    sel_df = usa_local.cbsas.get('seattle')
    df = usa_local.level(0).within(sel_df)
    assert isinstance(df, pd.DataFrame)


def test_get_enrich_variables(usa_local):
    ev_df = usa_local.enrich_variables
    kv_df = ev_df[ev_df.data_collection.str.startswith('Key')]
    assert kv_df.attrs['_cntry'].geo_name == 'USA'


def test_enrich_usa_seattle_level0_keyusfacts(usa_local):
    e_vars = usa_local.enrich_variables
    enrich_vars = e_vars[(e_vars.data_collection.str.startswith('Key')) & (e_vars.name.str.endswith('CY'))]
    bg_df = usa_local.cbsas.get('seattle').dm.level(0).get()
    df = bg_df.dm.enrich(enrich_vars)
    assert isinstance(df, pd.DataFrame) and df.spatial.validate()


def test_enrich_usa_local_trade_areas_enrich_country_attrs(usa_local):
    dir_prj = Path(__file__).absolute().parent.parent.parent

    dir_data = dir_prj / 'data'
    dir_test = dir_data / 'test'

    ta_pth = dir_test / 'trade_areas.json'
    assert ta_pth.exists()

    drop_cols = ['OBJECTID', 'AREA_ID', 'AREA_DESC', 'AREA_DESC2', 'AREA_DESC3', 'RING',
                 'RING_DEFN', 'STORE_LAT', 'STORE_LON', 'STORE_ID', 'LOCNUM', 'CONAME',
                 'STREET', 'CITY', 'STATE', 'STATE_NAME', 'ZIP', 'ZIP4', 'NAICS', 'SIC',
                 'SALESVOL', 'HDBRCH', 'ULTNUM', 'PUBPRV', 'EMPNUM', 'FRNCOD', 'ISCODE',
                 'SQFTCODE', 'LOC_NAME', 'STATUS', 'SCORE', 'SOURCE', 'REC_TYPE']

    with open(ta_pth) as fl:
        df = FeatureSet.from_json(fl.read()).sdf.drop(columns=drop_cols)
        df.spatial.set_geometry('SHAPE')

    e_vars = usa_local.enrich_variables

    key_vars = e_vars[(e_vars.name.str.endswith('CY')) & (e_vars.data_collection.str.startswith('Key'))]

    assert '_cntry' in key_vars.attrs.keys()
    assert isinstance(key_vars.attrs['_cntry'], dm.Country)

    e_df = df.dm.enrich(key_vars)

    assert isinstance(e_df, pd.DataFrame)
    assert '_cntry' in e_df.attrs.keys()
    assert isinstance(e_df.attrs['_cntry'], dm.Country)


def test_geographies_gis_implicit():
    with pytest.raises(Exception):
        gis = GIS()
        usa_agol = dm.Country('USA', source=gis)
        df = usa_agol.geographies
        assert isinstance(df, pd.DataFrame)


def test_country_enrich_method_local(usa_local):
    enrich_vars = usa_local.enrich_variables[(usa_local.enrich_variables.data_collection.str.startswith('KeyUSFacts'))
                                             & (usa_local.enrich_variables.name.str.endswith('CY'))].enrich_str
    cnty_df = usa_local.cbsas.get('seattle').dm.counties.get()
    df = usa_local.enrich(cnty_df, enrich_vars)
    assert isinstance(df, pd.DataFrame) and df.spatial.validate()


def test_country_get_nearest_biz_comp(usa_local):
    aoi_df = usa_local.cbsas.get('portland-vancouver')
    biz_df = usa_local.business.get_by_name('starbucks', aoi_df).iloc[:2]
    biz_df.spatial.set_geometry('SHAPE')
    comp_df = usa_local.business.get_competition(biz_df, aoi_df)
    near_df = biz_df.dm.get_nearest(comp_df, usa_local)
    assert isinstance(near_df, pd.DataFrame)
