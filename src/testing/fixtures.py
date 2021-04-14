import os

from arcgis.gis import GIS
from dotenv import load_dotenv, find_dotenv
import pytest

load_dotenv(find_dotenv())

# TODO: migrate these to a accounts owned by development for testing
# agol_url = 'https://geoai.maps.arcgis.com'
# agol_usr = 'headless_geoai'
# agol_pswd = 'Esri380!'
agol_url = os.getenv('BA_QA_URL')
agol_usr = os.getenv('BA_QA_USERNAME')
agol_pswd = os.getenv('BA_QA_PASSWORD')

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

@pytest.fixture
def counties_df():
    from .test_data import counties
    counties_df = counties.data_frame
    return counties_df
