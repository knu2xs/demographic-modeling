import os.path
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath('../'))
import dm


@pytest.fixture
def usa():
    return dm.Country('USA')


@pytest.fixture
def business(usa):
    return usa.business


@pytest.fixture
def cbsa(usa):
    return usa.cbsas.get('seattle')


def test_business_search(usa, cbsa):
    biz = usa.business.get_by_name(
        business_name='Ace Hardware',
        area_of_interest=cbsa
    )
    assert isinstance(biz, pd.DataFrame)


def test_business_get_competition_dataframe(usa, cbsa):
    biz = usa.business.get_by_name('Ace Hardware', cbsa)
    comp = usa.business.get_competition(
        brand_businesses=biz,
        code_column='NAICS',
        area_of_interest=cbsa
    )
    assert 500 > len(comp.index)


def test_business_get_competition_string(usa, cbsa):
    comp = usa.business.get_competition(
        brand_businesses='Ace Hardware',
        code_column='NAICS',
        area_of_interest=cbsa
    )
    assert 500 > len(comp.index)


def test_get_by_code(usa, cbsa):
    biz = usa.business.get_by_code(
        category_code=44413005,
        code_column='NAICS',
        area_of_interest=cbsa
    )
    assert 500 > len(biz.index)


def test_get_by_code_partial(usa, cbsa):
    biz = usa.business.get_by_code(
        category_code=444130,
        code_column='NAICS',
        area_of_interest=cbsa
    )
    assert 500 > len(biz.index)
