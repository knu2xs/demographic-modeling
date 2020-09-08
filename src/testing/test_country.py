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


def test_geographies_gis_implicit(usa_agol):
    df = usa_agol.geographies
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


def test_get_geography_local_within(usa_local):
    df = usa_local.cbsas.get('seattle').counties
    assert isinstance(df, pd.DataFrame)
