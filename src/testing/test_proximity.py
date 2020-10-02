import importlib
import os.path
from pathlib import Path
import sys

from arcgis.features import GeoAccessor
import pandas as pd
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath('../'))
import dm

from dm import Country
from dm import proximity as prx
from dm.country import DemographicModeling

arcpy_avail = True if importlib.util.find_spec("arcpy") else False

if arcpy_avail:
    import arcpy

biz_drop_cols = ['OBJECTID', 'CONAME','SALESVOL', 'HDBRCH', 'ULTNUM', 'PUBPRV', 'EMPNUM', 'FRNCOD', 'ISCODE',
                 'SQFTCODE', 'LOC_NAME', 'STATUS', 'SCORE', 'SOURCE', 'REC_TYPE']

@pytest.fixture
def usa():
    return Country('USA')


@pytest.fixture
def aoi_df(usa):
    return usa.cbsas.get('portland-vancouver')


@pytest.fixture
def biz_df(usa, aoi_df):
    biz_df = usa.business.get_by_name('starbucks', aoi_df).iloc[:2]
    biz_df.spatial.set_geometry('SHAPE')
    return biz_df


@pytest.fixture
def comp_df(usa, biz_df, aoi_df):
    return usa.business.get_competition(biz_df, aoi_df)


def test_get_nearest_biz_comp(usa, biz_df, comp_df):
    near_df = prx.get_nearest(biz_df, comp_df, usa)
    len_biz = len(biz_df.index)
    len_near = len(near_df.index)
    prox_cols = len([c for c in near_df.columns if c.startswith('proximity')])
    assert len_biz == len_near and prox_cols


def test_get_nearest_biz_comp(usa):
    aoi_df = usa.cbsas.get('seattle').dm.counties.get('king')
    lvl_df = aoi_df.dm.level(2).get()
    biz_df = usa.business.get_by_name('ace hardware', aoi_df).drop(columns=biz_drop_cols)
    biz_df.spatial.set_geometry('SHAPE')
    comp_df = usa.business.get_competition(biz_df, aoi_df, local_threshold=1).drop(columns=biz_drop_cols)
    comp_df.spatial.set_geometry('SHAPE')

    bg_near_biz_df = lvl_df.dm.get_nearest(biz_df, origin_id_column='ID', near_prefix='brand')
    bg_near_biz_comp_df = bg_near_biz_df.dm.get_nearest(comp_df, origin_id_column='ID',
                                                        near_prefix='comp',
                                                        destination_columns_to_keep=['brand_name',
                                                                                     'brand_name_category'])

    assert isinstance(bg_near_biz_comp_df, pd.DataFrame)
