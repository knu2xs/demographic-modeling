from arcgis.gis import GIS
import pytest

# TODO: migrate these to a accounts owned by development for testing
agol_url = 'https://geoai.maps.arcgis.com'
agol_usr = 'headless_geoai'
agol_pswd = 'Esri380!'

ent_url = 'https://geoai-ent.bd.esri.com/portal'
ent_usr = 'headless'
ent_pswd = 'Esri380!'


@pytest.fixture
def agol_gis():
    gis = GIS(agol_url, username=agol_usr, password=agol_pswd)
    return gis

@pytest.fixture
def ent_gis():
    gis = GIS(ent_url, username=ent_usr, password=ent_pswd)
    return gis
