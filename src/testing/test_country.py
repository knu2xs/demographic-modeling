"""
arcgis.country tests
"""
import sys
from pathlib import Path
from typing import Union

dir_src = Path(__file__).parent.parent.parent / 'src'
assert dir_src.exists()
sys.path.insert(0, str(dir_src))

from modeling import get_travel_modes, get_countries, Country
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
def get_countries_test(dm_source: Union[str, Country]):
    cntry_df = get_countries(dm_source)
    assert isinstance(cntry_df, pd.DataFrame)

def test_get_countries_local():
    get_countries_test('local')


def test_get_countries_agol(agol_gis):
    get_countries_test(agol_gis)


def test_get_countries_ent(ent_gis):
    get_countries_test(ent_gis)


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


def get_subgeo_levels_test(ba_src: Country):
    bg_df = ba_src.cbsas.get('seattle').mdl.level(0).get()
    assert isinstance(bg_df, pd.DataFrame)
    assert len(bg_df.index) == 2480


def test_get_subgeo_levels_local(local_usa):
    get_subgeo_levels_test(local_usa)


def test_get_subgeo_levels_agol(agol_usa):
    get_subgeo_levels_test(agol_usa)


def test_get_subgeo_levels_ent(ent_usa):
    get_subgeo_levels_test(ent_usa)


# enrich variable preprocessing
@pytest.fixture
def enrich_vars_df(local_usa: Country):
    ev = local_usa.enrich_variables
    kv = ev[
        (ev.data_collection.str.lower().str.contains('key'))  # get the key variables
        & (ev.name.str.endswith('CY'))  # just current year (2019) variables
        ].reset_index(drop=True)
    return kv


def test_enrich_variable_preprocessing_name_list(local_usa, enrich_vars_df):
    name_lst = list(enrich_vars_df['name'])
    in_len = len(name_lst)
    enrich_vars = local_usa._enrich_variable_preprocessing(name_lst)
    assert isinstance(enrich_vars, pd.Series)
    assert len(enrich_vars.index) == in_len


def test_enrich_variable_preprocessing_name_list_extra_var(local_usa, enrich_vars_df):
    name_lst = list(enrich_vars_df['name'])
    in_len = len(name_lst)
    name_lst.append('whack_a_mole')
    with pytest.warns(UserWarning):
        enrich_vars = local_usa._enrich_variable_preprocessing(name_lst)
        assert isinstance(enrich_vars, pd.Series)
        assert len(enrich_vars.index) == in_len


def test_enrich_variable_preprocessing_proname_nparray(local_usa, enrich_vars_df):
    proname_arr = enrich_vars_df['enrich_name'].values
    in_len = len(proname_arr)
    enrich_vars = local_usa._enrich_variable_preprocessing(proname_arr)
    assert isinstance(enrich_vars, pd.Series)
    assert len(enrich_vars.index) == in_len


def test_enrich_variable_preprocessing_fieldname_series(local_usa, enrich_vars_df):
    fldnm_srs = enrich_vars_df['enrich_field_name']
    in_len = len(fldnm_srs)
    enrich_vars = local_usa._enrich_variable_preprocessing(fldnm_srs)
    assert isinstance(enrich_vars, pd.Series)
    assert len(enrich_vars.index) == in_len


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
    assert enrch_df.notna().all(axis=1).all()


def test_enrich_keycy_local(local_usa):
    enrich_keycy_test(local_usa)


def test_enrich_keycy_ent(ent_usa):
    enrich_keycy_test(ent_usa)


def enrich_keycy_set_source_test(ba_src: Country):
    kv = get_key_cy_vars(ba_src)
    cnty_df = ba_src.cbsas.get('seattle').mdl.counties.get()
    cnty_df.attrs = {}  # flush the attrs so the ModelingAccessor has no clue
    enrch_df = cnty_df.mdl.enrich(kv, source=ba_src)
    assert isinstance(enrch_df, pd.DataFrame)
    assert enrch_df.notna().all(axis=1).all()


def test_enrich_keycy_set_source_local(local_usa):
    enrich_keycy_set_source_test(local_usa)


def test_enrich_keycy_set_source_ent(ent_usa):
    enrich_keycy_set_source_test(ent_usa)


def test_enrich_keycy_set_source_ent(agol_usa):
    enrich_keycy_set_source_test(agol_usa)


def test_enrich_keycy_ent_batch(ent_usa):
    kv = get_key_cy_vars(ent_usa)
    cnty_df = ent_usa.cbsas.get('seattle').mdl.level(0).get()
    enrch_df = cnty_df.mdl.enrich(kv)
    assert isinstance(enrch_df, pd.DataFrame)
    assert enrch_df.notna().all(axis=1).all()


def test_enrich_keycy_ent_set_source_batch(ent_usa):
    kv = get_key_cy_vars(ent_usa)
    geo_df = ent_usa.cbsas.get('seattle').mdl.level(1).get()
    geo_df.attrs = {}  # make the ModelingAccessor work for it
    enrch_df = geo_df.mdl.enrich(kv)
    assert isinstance(enrch_df, pd.DataFrame)
    assert enrch_df.notna().all(axis=1).all()


def enrich_keycy_by_ids_test(cntry: Country):
    bg_df = cntry.cities_and_towns_places.get('seattle').mdl.level(0).get().drop(columns=['SHAPE'])
    bg_df.attrs = {}  # flush the attrs to make sure we are not inadvertently cheating
    kv = get_key_cy_vars(cntry)
    enrch_df = bg_df.mdl.enrich(kv, enrich_geography_level='block_groups', enrich_id_column='ID', source=cntry)
    assert isinstance(enrch_df, pd.DataFrame)
    assert enrch_df.notna().all(axis=1).all()


def test_enrich_keycy_by_ids_local(local_usa):
    enrich_keycy_by_ids_test(local_usa)


def test_enrich_keycy_by_ids_ent(ent_usa):
    enrich_keycy_by_ids_test(ent_usa)


def test_enrich_keycy_by_ids_agol(agol_usa):
    enrich_keycy_by_ids_test(agol_usa)


# businesses functionality testing
@pytest.fixture
def aoi_local(local_usa):
    aoi_df = local_usa.cbsas.get('seattle')
    return aoi_df


@pytest.fixture
def aoi_gis_agol(agol_usa):
    aoi_df = agol_usa.cbsas.get('seattle')
    return aoi_df


@pytest.fixture
def aoi_gis_ent(ent_usa):
    aoi_df = ent_usa.cbsas.get('seattle')
    return aoi_df


def get_business_by_name_test(aoi_df):
    biz_df = aoi_df.mdl.business.get_by_name('ace hardware')
    assert isinstance(biz_df, pd.DataFrame)
    assert pd.Series(('location_id', 'brand_name', 'brand_name_category')).isin(biz_df.columns).all()


def test_get_business_by_name_local(aoi_local):
    get_business_by_name_test(aoi_local)


def test_get_business_by_name_agol(aoi_gis_agol):
    get_business_by_name_test(aoi_gis_agol)


def test_get_business_by_name_ent(aoi_gis_ent):
    get_business_by_name_test(aoi_gis_ent)


def get_business_by_code_naics_test(aoi_df):
    biz_df = aoi_df.mdl.business.get_by_code('44413005')
    assert isinstance(biz_df, pd.DataFrame)
    assert pd.Series(('location_id', 'brand_name', 'brand_name_category')).isin(biz_df.columns).all()


def test_get_business_by_code_naics_local(aoi_local):
    get_business_by_code_naics_test(aoi_local)


def test_get_business_by_code_naics_agol(aoi_gis_agol):
    get_business_by_code_naics_test(aoi_gis_agol)


def test_get_business_by_code_naics_ent(aoi_gis_ent):
    get_business_by_code_naics_test(aoi_local)


def get_business_by_code_naics_truncated_test(aoi_df):
    biz_df = aoi_df.mdl.business.get_by_code('444130')
    assert isinstance(biz_df, pd.DataFrame)
    assert pd.Series(('location_id', 'brand_name', 'brand_name_category')).isin(biz_df.columns).all()


def test_get_business_by_code_naics_truncated_local(aoi_local):
    get_business_by_code_naics_truncated_test(aoi_local)


def test_get_business_by_code_naics_truncated_agol(aoi_gis_agol):
    get_business_by_code_naics_truncated_test(aoi_gis_agol)


def test_get_business_by_code_naics_truncated_ent(aoi_gis_ent):
    get_business_by_code_naics_truncated_test(aoi_local)


def get_business_by_code_sic_test(aoi_df):
    biz_df = aoi_df.mdl.business.get_by_code('525104', code_type='SIC')
    assert isinstance(biz_df, pd.DataFrame)
    assert pd.Series(('location_id', 'brand_name', 'brand_name_category')).isin(biz_df.columns).all()


def test_get_business_by_code_sic_local(aoi_local):
    get_business_by_code_sic_test(aoi_local)


def test_get_business_by_code_sic_agol(aoi_gis_agol):
    get_business_by_code_sic_test(aoi_gis_agol)


def test_get_business_by_code_sic_ent(aoi_gis_ent):
    get_business_by_code_sic_test(aoi_local)


def get_business_by_code_sic_truncated_test(aoi_df):
    biz_df = aoi_df.mdl.business.get_by_code('5251', code_type='SIC')
    assert isinstance(biz_df, pd.DataFrame)
    assert pd.Series(('location_id', 'brand_name', 'brand_name_category')).isin(biz_df.columns).all()


def test_get_business_by_code_sic_truncated_local(aoi_local):
    get_business_by_code_sic_truncated_test(aoi_local)


def test_get_business_by_code_sic_truncated_agol(aoi_gis_agol):
    get_business_by_code_sic_truncated_test(aoi_gis_agol)


def test_get_business_by_code_sic_truncated_ent(aoi_gis_ent):
    get_business_by_code_sic_truncated_test(aoi_local)


def get_business_competition_using_brand_name_test(aoi_df):
    cmp_df = aoi_df.mdl.business.get_competition('ace hardware')
    assert isinstance(cmp_df, pd.DataFrame)
    assert pd.Series(('location_id', 'brand_name', 'brand_name_category')).isin(cmp_df.columns).all()


def test_get_business_competition_using_brand_name_local(aoi_local):
    get_business_competition_using_brand_name_test(aoi_local)


def test_get_business_competition_using_brand_name_agol(aoi_gis_agol):
    get_business_competition_using_brand_name_test(aoi_gis_agol)


def test_get_business_competition_using_brand_name_ent(aoi_gis_ent):
    get_business_competition_using_brand_name_test(aoi_gis_ent)


def get_business_competition_using_brand_df_test(aoi_df):
    biz_df = aoi_df.mdl.business.get_by_name('ace hardware')
    cmp_df = aoi_df.mdl.business.get_competition(biz_df)
    assert isinstance(cmp_df, pd.DataFrame)
    assert pd.Series(('location_id', 'brand_name', 'brand_name_category')).isin(cmp_df.columns).all()
    assert cmp_df.spatial.validate()


def test_get_business_competition_using_brand_df_local(aoi_local):
    get_business_competition_using_brand_df_test(aoi_local)


def test_get_business_competition_using_brand_df_agol(aoi_gis_agol):
    get_business_competition_using_brand_df_test(aoi_gis_agol)


def test_get_business_competition_using_brand_df_ent(aoi_gis_ent):
    get_business_competition_using_brand_df_test(aoi_local)


# routing tests
def get_travel_modes_test(src):
    trvl_df = get_travel_modes(src)
    assert isinstance(trvl_df, pd.DataFrame)


def test_get_travel_modes_gis_ent(ent_gis):
    get_travel_modes_test(ent_gis)


def test_get_travel_modes_gis_agol(agol_gis):
    get_travel_modes_test(agol_gis)


def test_get_travel_modes_local():
    get_travel_modes_test('local')


def test_get_travel_modes_cntry_gis_ent(ent_usa):
    get_travel_modes_test(ent_usa)


def test_get_travel_modes_cntry_gis_agol(agol_usa):
    get_travel_modes_test(agol_usa)


def test_get_travel_modes_cntry_local(local_usa):
    get_travel_modes_test(local_usa)


def tracts_routing_test(aoi_df:pd.DataFrame):
    geo_df = aoi_df.mdl.level(1).get()
    biz_df = aoi_df.mdl.business.get_by_name('ace hardware')
    geo_near_biz_df = geo_df.mdl.proximity.get_nearest(biz_df, origin_id_column='ID', near_prefix='brand')
    assert isinstance(geo_near_biz_df, pd.DataFrame)
    assert geo_near_biz_df.spatial.validate()
    assert len(geo_near_biz_df.index) == len(geo_df.index)


def test_tracts_routing_local(aoi_local):
    test_tracts_routing_local(aoi_local)


def test_tracts_routing_gis_ent(aoi_gis_ent):
    tracts_routing_test(aoi_gis_ent)


def test_tracts_routing_gis_agol(aoi_gis_agol):
    tracts_routing_test(aoi_gis_agol)


def biz_to_comp_routing_test(aoi_df:pd.DataFrame):
    biz_df = aoi_df.mdl.business.get_by_name('ace hardware')
    comp_df = aoi_df.mdl.business.get_competition(biz_df)
    biz_near_comp_df = biz_df.mdl.proximity.get_nearest(comp_df, origin_id_column='LOCNUM',
                                                        destination_id_column='LOCNUM')
    assert isinstance(biz_near_comp_df, pd.DataFrame)
    assert biz_near_comp_df.spatial.validate()
    assert len(biz_near_comp_df) == len(biz_df)


def test_biz_to_comp_routing_local(aoi_local):
    biz_to_comp_routing_test(aoi_local)


def test_biz_to_comp_routing_ent(aoi_gis_ent):
    biz_to_comp_routing_test(aoi_gis_ent)


def test_biz_to_comp_routing_agol(aoi_gis_agol):
    biz_to_comp_routing_test(aoi_gis_agol)