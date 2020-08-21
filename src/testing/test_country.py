import os.path
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath('../'))
import dm


def test_get_geography_local_explicit():

    usa = dm.Country('USA')
    states_df = usa.get_geography_local('states')
    assert isinstance(states_df, pd.DataFrame)


def test_get_geography_local_implicit():

    usa = dm.Country('USA', source='local')
    states_df = usa.get_geography('states')
    assert isinstance(states_df, pd.DataFrame)