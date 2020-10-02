import os.path
import sys

from arcgis.gis import GIS
import pandas as pd
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
    df = cbsa_df.dm.counties.get()
    assert len(df.index) == 3


def test_get_geography_local_within_df(usa_local):
    sel_df = usa_local.cbsas.get('seattle')
    df = usa_local.level(0).within(sel_df)
    assert isinstance(df, pd.DataFrame)


def test_enrich_usa_seattle_level0_keyusfacts(usa_local):
    enrich_vars = usa_local.enrich_variables[(usa_local.enrich_variables.data_collection == 'KeyUSFacts')
                                             & (usa_local.enrich_variables.vintage == '2019')].enrich_str
    df = usa_local.cbsas.get('seattle').dm.level(0).get().dm.enrich(enrich_vars)
    assert isinstance(df, pd.DataFrame) and df.spatial.validate()


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
