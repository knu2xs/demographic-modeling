import os.path
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath('../'))
from dm import _xml_interrogation as xml_interrogation


def test_get_heirarchial_geography_dataframe():
    df = xml_interrogation.get_heirarchial_geography_dataframe()
    assert isinstance(df, pd.DataFrame)


def test_get_enrich_variables_dataframe():
    df = xml_interrogation.get_enrich_variables_dataframe()
    assert isinstance(df, pd.DataFrame)


def test_get_collection_dataframe():
    file = Path('./')
    df = xml_interrogation._get_collection_dataframe(file)
    assert isinstance(df, pd.DataFrame)