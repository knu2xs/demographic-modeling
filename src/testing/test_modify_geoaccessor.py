from pathlib import Path
import pytest
import sys
import tempfile

sys.path.insert(0, '../src')
from dm import Country
from dm._modify_geoaccessor import GeoAccessorIO

usa = Country('USA', source='local')
fc_pth = usa.geographies.iloc[9].feature_class_path


@pytest.fixture
def sedf():
    return GeoAccessorIO.from_featureclass(fc_pth)


def test_to_csv(sedf):
    tmp_pth = Path(tempfile.mktemp(suffix='.csv'))
    sedf.spatial.to_csv(tmp_pth)
    assert tmp_pth.exists()
