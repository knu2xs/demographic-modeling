import os.path
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath('../'))
import dm


def test_get_geography_local_explicit():
    usa = dm.Country('USA')
    df = usa._get_geography_local('states')
    assert isinstance(df, pd.DataFrame)


def test_get_geography_local_implicit():
    usa = dm.Country('USA', source='local')
    df = usa.get_geography('states')
    assert isinstance(df, pd.DataFrame)


def test_get_geography_local_explicit_int():
    usa = dm.Country('USA')
    df = usa._get_geography_local(0)
    assert isinstance(df, pd.DataFrame)


def test_get_geography_local_implicit_int():
    usa = dm.Country('USA')
    df = usa.get_geography(9)
    assert isinstance(df, pd.DataFrame)
