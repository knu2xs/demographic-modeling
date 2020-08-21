from pathlib import Path
import pytest
import sys
import tempfile

sys.path.insert(0, '../src')
from dm import Country
from dm._modify_geoaccessor import GeoAccessorIO


@pytest.fixture
def sedf():
    usa = Country('USA', source='local')
    fc_row = usa.geographies.iloc[9]
    fc_pth = fc_row.feature_class_path
    fc_cols = [fc_row.col_id, fc_row.col_name]
    return GeoAccessorIO.from_featureclass(fc_pth, fields=fc_cols)


def test_to_csv(sedf):
    tmp_pth = Path(tempfile.mktemp(suffix='.csv'))
    sedf.spatial.to_csv(tmp_pth)
    print(tmp_pth)
    assert tmp_pth.exists()


def test_to_parquet(sedf):
    tmp_pth = Path(tempfile.mkdtemp(suffix='.parquet'))
    sedf.spatial.to_parquet(tmp_pth)
    print(tmp_pth)
    assert tmp_pth.exists()
