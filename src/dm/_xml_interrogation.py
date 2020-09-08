import os
import xml.etree.ElementTree as ET

import pandas as pd

from ._registry import get_ba_key_value


def _get_path_from_bds(bds_dir, bds_pth):
    bds_tree = ET.parse(bds_pth)
    bds_root = bds_tree.getroot()

    e_src = bds_root.find('.//MappingLayer/DataSource')
    gdb = e_src.find('WS').attrib['PathName'].replace('.\\', '')
    fds = e_src.find('FeatureDataset').attrib['FeatureDataset']
    fc = e_src.find('Dataset').attrib['Dataset']

    return os.path.join(bds_dir, gdb, fds, fc)


def _get_lyr(bds_dir, e_lyr):
    alias = e_lyr.find('./SIMPLE_PROPERTY[@name="Caption"]').attrib['value']
    name = alias.lower().replace(' ', '_').replace('(', '').replace(')', '')

    e_fl = e_lyr.find('./COMPLEX_PROPERTY[@name="DataFeatureLayer"]')

    if e_fl is not None:
        col_id = e_fl.find('./SIMPLE_PROPERTY[@name="IDField"]').attrib['value']
        col_name = e_fl.find('./SIMPLE_PROPERTY[@name="NameField"]').attrib['value']

        bds_lyr_nm = e_fl.find(
            './ARRAY_PROPERTY[@name="DetalizationLayers"]/COMPLEX_PROPERTY/SIMPLE_PROPERTY[@name="LevelName"]').attrib[
            'value']
        bds_pth = os.path.join(bds_dir, bds_lyr_nm)

        fc_pth = _get_path_from_bds(bds_dir, bds_pth)

    else:
        col_id, col_name, fc_pth = None, None, None

    return name, alias, col_id, col_name, fc_pth


def get_heirarchial_geography_dataframe(three_letter_country_identifier: str = 'USA')->pd.DataFrame:
    """
    Get a df of available demographic geography_level area resolutions.
    Args:
        three_letter_country_identifier: Just like it sounds, the three letter country identifier. Defaults to 'USA'.

    Returns: pd.DataFrame ordered from smallest area (block group in USA) to largest area (typically entire country).
    """
    settings_xml = get_ba_key_value('SettingsFile', three_letter_country_identifier)
    bds_dir = get_ba_key_value('DemographyDataDir1', three_letter_country_identifier)

    tree = ET.parse(settings_xml)
    root = tree.getroot()

    x_pth_lyrs = './/TASK[@id="GeographyLevelCategory.StandartGeographyTA"]' \
                 '//ARRAY_PROPERTY[@name="GeographyLevelLayers"]'

    # get the parent element for all layers
    e_lyrs = root.find(x_pth_lyrs)

    # get the parent element for each respective layer
    e_lst_lyrs = e_lyrs.findall('COMPLEX_PROPERTY')

    row_lst = [_get_lyr(bds_dir, e) for e in e_lst_lyrs]

    df = pd.DataFrame(row_lst, columns=['name', 'alias', 'col_id', 'col_name', 'feature_class_path']).dropna()

    return df.iloc[::-1].reset_index(drop=True)
