import os.path
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath('../'))
from dm._xml_interrogation import get_heirarchial_geography_dataframe


def test_get_heirarchial_geography_dataframe():

    df = get_heirarchial_geography_dataframe()

    assert isinstance(df, pd.DataFrame)
