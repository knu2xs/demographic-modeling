"""
arcgis.country tests
"""
import sys
from pathlib import Path

dir_src = Path(__file__).parent.parent.parent / 'src'
assert dir_src.exists()
sys.path.insert(0, str(dir_src))

from modeling import get_countries, Country
import pandas as pd
import pytest

from .fixtures import agol_gis, ent_gis


@pytest.fixture
def local_usa():
    return Country('USA', source='local')


@pytest.fixture
def agol_usa(agol_gis):
    return Country('USA', source=agol_gis)


@pytest.fixture
def ent_usa(ent_gis):
    return Country('USA', source=ent_gis)


# get countries
def test_get_countries_local():
    cntry_df = get_countries('local')
    assert isinstance(cntry_df, pd.DataFrame)


def test_get_countries_agol(agol_gis):
    cntry_df = get_countries(agol_gis)
    assert isinstance(cntry_df, pd.DataFrame)


def test_get_countries_ent(ent_gis):
    cntry_df = get_countries(ent_gis)
    assert isinstance(cntry_df, pd.DataFrame)


# create country
def test_create_country_local_implicit():
    cntry = Country('USA')
    assert isinstance(cntry, Country)


def test_create_country_local():
    cntry = Country('USA', source='local')
    assert isinstance(cntry, Country)


def test_create_country_local_explicit_year():
    cntry = Country('USA', source='local', year=2020)
    assert isinstance(cntry, Country)


def test_create_country_local_explicit_year_not_avail():
    with pytest.raises(AssertionError):
        Country('USA', source='local', year=1960)


def test_create_country_agol(agol_gis):
    cntry = Country('USA', source=agol_gis)
    assert isinstance(cntry, Country)


def test_create_country_ent(ent_gis):
    cntry = Country('USA', source=ent_gis)
    assert isinstance(cntry, Country)


# enrich variables
def get_enrich_variables_test(ba_src: Country):
    evars = ba_src.enrich_variables
    assert isinstance(evars, pd.DataFrame)


def test_get_enrich_variables_local(local_usa):
    get_enrich_variables_test(local_usa)


def test_get_enrich_variables_agol(agol_usa):
    get_enrich_variables_test(agol_usa)


def test_get_enrich_variables_ent(ent_usa):
    get_enrich_variables_test(ent_usa)


# get geo - aoi
def get_aoi_geo_test(ba_src: Country):
    aoi_df = ba_src.cbsas.get('seattle')
    assert isinstance(aoi_df, pd.DataFrame)
    assert len(aoi_df.index) == 1


def test_get_aoi_geo_local(local_usa):
    get_aoi_geo_test(local_usa)


def test_get_aoi_geo_agol(agol_usa):
    get_aoi_geo_test(agol_usa)


def test_get_aoi_geo_ent(ent_usa):
    get_aoi_geo_test(ent_usa)


# get geo - subgeo
def get_subgeo_test(ba_src: Country):
    cnty_df = ba_src.cbsas.get('seattle').mdl.counties.get()
    assert isinstance(cnty_df, pd.DataFrame)
    assert len(cnty_df.index) == 3


def test_get_subgeo_local(local_usa):
    get_subgeo_test(local_usa)


def test_get_subgeo_agol(agol_usa):
    get_subgeo_test(agol_usa)


def test_get_subgeo_ent(ent_usa):
    get_subgeo_test(ent_usa)


# enrich
def get_key_cy_vars(ba_src: Country):
    ev = ba_src.enrich_variables
    kv = ev[
        (ev.data_collection.str.lower().str.contains('key'))  # get the key variables
        & (ev.name.str.endswith('CY'))  # just current year (2019) variables
        ].reset_index(drop=True)
    return kv


def enrich_keycy_test(ba_src: Country):
    kv = get_key_cy_vars(ba_src)
    cnty_df = ba_src.cbsas.get('seattle').mdl.counties.get()
    enrch_df = cnty_df.mdl.enrich(kv)
    assert isinstance(enrch_df, pd.DataFrame)


def test_enrich_keycy_local(local_usa):
    enrich_keycy_test(local_usa)


def test_enrich_keycy_ent(ent_usa):
    enrich_keycy_test(ent_usa)
