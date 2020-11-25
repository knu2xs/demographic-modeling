import itertools
import os
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import pandas as pd

from ._registry import get_ba_key_value


def _get_path_from_bds(bds_dir, bds_pth):
    bds_tree = ET.parse(bds_pth)
    bds_root = bds_tree.getroot()

    # dig through the xml to get the path to the file geodatabase
    e_src = bds_root.find('.//DataSource')
    gdb = e_src.find('WS').attrib['PathName'].replace('.\\', '')

    # use path to get back the correctly cased path to the geodatabase
    gdb_pth = Path(os.path.join(bds_dir, gdb))
    gdb_pth = [pth for pth in gdb_pth.parent.glob(f'*{gdb_pth.name}')][0]
    gdb = str(gdb_pth)

    # now, get the rest of the path components
    fds = e_src.find('FeatureDataset').attrib['FeatureDataset']
    fc = e_src.find('Dataset').attrib['Dataset']

    return os.path.join(gdb, fds, fc)


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


def get_heirarchial_geography_dataframe(three_letter_country_identifier: str = 'USA',
                                        year: [str, int] = None) -> pd.DataFrame:
    """
    Get a input_dataframe of available demographic geography_level area resolutions.

    Args:
        three_letter_country_identifier: Required
            Just like it sounds, the three letter country identifier. Defaults to 'USA'.
        year: Optional


    Returns: pd.DataFrame ordered from smallest area (block group in USA) to largest area (typically entire country).
    """
    year = str(year) if isinstance(year, int) else year

    settings_xml = get_ba_key_value('SettingsFile', three_letter_country_identifier, year)
    bds_dir = get_ba_key_value('DemographyDataDir1', three_letter_country_identifier, year)

    tree = ET.parse(settings_xml)
    root = tree.getroot()

    x_pth_lyrs = './/TASK[@id="GeographyLevelCategory.StandartGeographyTA"]' \
                 '//ARRAY_PROPERTY[@name="GeographyLevelLayers"]'

    # get the parent element for all layers
    e_lyrs = root.find(x_pth_lyrs)

    # get the parent element for each respective layer
    e_lst_lyrs = e_lyrs.findall('COMPLEX_PROPERTY')

    row_lst = [_get_lyr(bds_dir, e) for e in e_lst_lyrs]

    df = pd.DataFrame(row_lst, columns=['geo_name', 'geo_alias', 'col_id', 'col_name', 'feature_class_path']).dropna()

    return df.iloc[::-1].reset_index(drop=True)


def _is_hidden(field_element):
    """
    Helper function to determine if an xml element is hidden or not.
    Args:
        field_element: ElementTree field element from an enrich collection file.
    Returns: Boolean
    """
    if 'HideInDataBrowser' in field_element.attrib and field_element.attrib['HideInDataBrowser'] == 'True':
        return True
    else:
        return False


def _get_out_field_name(geoenrichment_field):
    """
    Helper function to crate field names used by Business Analyst - useful for reverse lookups.
    Args:
        geoenrichment_field:
    Returns: String

    """
    out_field_name = geoenrichment_field.replace(".", "_")

    # if string starts with a set of digits, replace them with Fdigits
    out_field_name = re.sub(r"(^\d+)", r"F\1", out_field_name)

    # cut to first 64 characters
    return out_field_name[:64]


def _get_collection_dataframe(collection_file: Path) -> pd.DataFrame:
    """
    Helper function to parse a collection file and return a dataframe of enrichment properties.
    Args:
        collection_file: Full path object to collection xml file.
    Returns: pd.DataFrame of enrichment variables.
    """
    # start by getting access to the file through ElementTree root
    coll_tree = ET.parse(collection_file)
    coll_root = coll_tree.getroot()

    # create a list object to populate with property values
    fld_lst = []

    # collect any raw scalar fields
    uncalc_ele_fields = coll_root.find('./Calculators/Demographic/Fields')
    if uncalc_ele_fields is not None:
        fld_lst.append([(field_ele.attrib['Name'], field_ele.attrib['Alias'], field_ele.attrib['Units'], field_ele.attrib['Vintage'])
                        for field_ele in uncalc_ele_fields.findall('Field')
                        if not _is_hidden(field_ele)])

    # collect any calculated field types
    calc_ele_fields = coll_root.find('./Calculators/Demographic/CalculatedFields')
    if calc_ele_fields is not None:

        # since there are two types of calculated fields, account for this
        for field_type in ['PercentCalc', 'Script']:
            single_fld_lst = [(field_ele.attrib['Name'], field_ele.attrib['Alias'], 'CALCULATED', field_ele.attrib['Vintage'])
                              for field_ele in calc_ele_fields.findall(field_type)
                              if not _is_hidden(field_ele)]
            fld_lst.append(single_fld_lst)

    # combine the results of both uncalculated and calculated fields located into single result
    field_lst = list(itertools.chain.from_iterable(fld_lst))

    if len(field_lst):
        # create a dataframe with the field information
        coll_df = pd.DataFrame(field_lst, columns=['name', 'alias', 'type', 'vintage'])

        # using the collected information, create the really valuable fields
        coll_df['data_collection'] = collection_file.stem
        coll_df['enrich_str'] = coll_df.apply(lambda row: f"{row['data_collection']}.{row['name']}", axis='columns')
        coll_df['enrich_field_name'] = coll_df['enrich_str'].apply(lambda val: _get_out_field_name(val))

    else:
        coll_df = None

    return coll_df


def get_enrich_variables_dataframe(three_letter_country_identifier: str = 'USA') -> pd.DataFrame:
    """
    Retrieve a listing of all available enrichment variables for local analysis.
    Args:
        three_letter_country_identifier: Just like it sounds, the three letter country identifier. Defaults to 'USA'.
    Returns: pd.DataFrame with variable information.
    """
    # get a complete list of enrichment collection files, which does not include the enrichment packs (ugh!)
    coll_dir = Path(get_ba_key_value('DataCollectionsDir', three_letter_country_identifier))
    coll_xml_lst = [f for f in coll_dir.glob('*') if f.name != 'EnrichmentPacksList.xml']

    # get and combine all the results from the data collection files
    coll_df_lst = [_get_collection_dataframe(coll_file) for coll_file in coll_xml_lst]
    coll_df = pd.concat([df for df in coll_df_lst if df is not None])

    # organize the results logically - cleanup
    coll_df.sort_values(['data_collection', 'name'])
    coll_df.reset_index(drop=True, inplace=True)

    return coll_df


def get_business_points_data_path(three_letter_country_identifier: str = 'USA') -> str:
    """
    Retrieve the path to the installed business listings feature class.
    Args:
        three_letter_country_identifier: Three letter country identifier. Defaults to 'USA'.
    Returns: String path to the business listings feature class.
    """
    # get the path to the directory where business data is stored
    biz_dir = Path(get_ba_key_value('DemographyDataDir2', three_letter_country_identifier))
    biz_settings_pth = biz_dir / 'busmetadata.xml'

    # pull the information out of the xml
    biz_tree = ET.parse(biz_settings_pth)
    biz_root = biz_tree.getroot()
    datasource_ele = biz_root.find('.//DataSource')

    # pull out the geodatabase and feature class name
    gdb = datasource_ele.find('./Workspace').attrib['PathName'].replace('.\\', '')
    fc = datasource_ele.find('./Dataset').attrib['name']

    # combine the retrieved elements for the full path to the business listings data
    biz_fc_pth = biz_dir / gdb / fc

    return str(biz_fc_pth)
