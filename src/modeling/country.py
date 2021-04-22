"""
Functions for countries introspection and Country object providing single interface for data preparation for modeling.
"""
import math
from pathlib import Path
import re
from typing import Union, AnyStr, Tuple
from warnings import warn

from arcgis.gis import GIS
from arcgis.features import GeoAccessor, FeatureSet
from arcgis.geometry import Geometry
import numpy as np
import pandas as pd

from .utils import avail_arcpy, local_vs_gis, set_source, geography_iterable_to_arcpy_geometry_list, can_enrich_gis, \
    has_networkanalysis_gis, get_sanitized_names

if avail_arcpy:
    import arcpy

    arcpy.env.overwriteOutput = True


def _get_countries_local():
    """Local introspection to discover what countries are available."""
    # get a generator of dataset objects
    ds_lst = list(arcpy._ba.ListDatasets())

    # throw error if no local datasets are available
    assert len(ds_lst), 'No datasets are available locally. If you want to locate available countries on a Web GIS, ' \
                        'please provide a GIS object instance as for the source parameter.'

    # organize all the country dataset properites
    cntry_lst = [
        (ds.CountryInfo.Name, ds.Version, ds.CountryInfo.ISO2, ds.CountryInfo.ISO3, ds.Caption, ds.DataSourceID, ds.ID)
        for ds in ds_lst]

    # create a dataframe of the country properties
    cntry_df = pd.DataFrame(cntry_lst,
                            columns=['country_name', 'vintage', 'iso2', 'iso3', 'description', 'data_source_id',
                                     'country_id'])

    # convert the vintage years to integer
    cntry_df['vintage'] = cntry_df['vintage'].astype('int64')

    # ensure the values are in order by country name and year
    cntry_df.sort_values(['country_name', 'vintage'], inplace=True)

    # organize the columns
    cntry_df = cntry_df[['iso2', 'iso3', 'country_name', 'vintage', 'country_id', 'data_source_id']]

    return cntry_df


def _get_countries_gis(source):
    """GIS introspection to discover what countries are available."""
    # make sure countries are available
    ge_err_msg = 'The provided GIS instance does not appear to have geoenrichment enabled and configured, so no ' \
                 'countries are available.'
    assert 'geoenrichment' in source.properties.helperServices, ge_err_msg
    assert isinstance(source.properties.helperServices.geoenrichment['url'], str), ge_err_msg

    # extract out the geoenrichment url
    ge_url = source.properties.helperServices.geoenrichment['url']

    # get a list of countries available on the Web GIS for enrichment
    url = f'{ge_url}/Geoenrichment/Countries'
    cntry_res = source._con.post(url, {'f': 'json'})
    cntry_dict = cntry_res['countries']

    # convert the dictionary to a dataframe
    cntry_df = pd.DataFrame(cntry_dict)

    # clean up some column names for consistency
    cntry_df.rename({'id': 'iso2', 'abbr3': 'iso3', 'name': 'country_name', 'altName': 'alt_name',
                     'defaultDatasetID': 'country_id'}, inplace=True, axis=1)
    cntry_df.drop(columns=['distanceUnits', 'esriUnits', 'hierarchies', 'currencySymbol', 'currencyFormat',
                           'defaultDataCollection', 'dataCollections', 'defaultReportTemplate',
                           'datasets', 'defaultExtent'], inplace=True)
    cntry_df = cntry_df[['iso2', 'iso3', 'country_name', 'country_id', 'alt_name', 'continent']]

    return cntry_df


def get_countries(source: Union[GIS, AnyStr] = 'local') -> pd.DataFrame:
    """
    Get the available countries based on the enrichment source.
    """

    # ensure the source is as expected
    assert isinstance(source, (str, GIS)), f'Please ensure the "source" is either the "local" keyword, or a valid ' \
                                           f'GIS object instance instead of "{type(source)}".'

    # if the source is local
    if isinstance(source, str):

        # make sure the correct keyword is being used to avoid confusion
        assert source.lower() == 'local', f'If you are interested in countries available locally, please use the ' \
                                          f'"local" keyword for the "source" parameter. You provided "{source}".'

        # arcpy must be available to perform enrichment locally
        assert avail_arcpy, 'Local enrichment can only be performed in an environment with the arcpy Python ' \
                            'package installed and available.'

        # get the local country dataframe
        cntry_df = _get_countries_local()

    # if we are using accessing a Web GIS object instance
    elif isinstance(source, GIS):

        # get gis enrichment countries available
        cntry_df = _get_countries_gis(source)

    return cntry_df


def _standardize_geographic_level_input(geo_df, geo_in):
    """Helper function to check and standardize inputs."""
    if isinstance(geo_in, str):
        if geo_in not in geo_df.name.values:
            names = ', '.join(geo_df.names.values)
            raise Exception(
                f'Your selector, "{geo_in}," is not an available selector. Please choose from {names}.')
        return geo_in

    elif isinstance(geo_in, int) or isinstance(geo_in, float):
        if geo_in > len(geo_df.index):
            raise Exception(
                f'Your selector, "{geo_in}", is beyond the maximum range of available geography_levels.')
        return geo_df.iloc[geo_in]['geo_name']

    elif geo_in is None:
        return None

    else:
        raise Exception('The geographic selector must be a string or integer.')


def _get_geographic_level_where_clause(selector=None, selection_field='NAME', query_string=None):
    """Helper function to consolidate where clauses."""
    # set up the where clause based on input
    if query_string:
        return query_string

    elif selection_field and selector:
        return f"{selection_field} LIKE '%{selector}%'"

    else:
        return None


def _get_geography_preprocessing(geo_df: pd.DataFrame, geography: Union[str, int], selector: str = None,
                                 selection_field: str = 'NAME', query_string: str = None,
                                 aoi_geography: Union[str, int] = None, aoi_selector: str = None,
                                 aoi_selection_field: str = 'NAME', aoi_query_string: str = None) -> Tuple:
    """Helper function consolidating input parameters for later steps."""
    # standardize the geography_level input
    geo = _standardize_geographic_level_input(geo_df, geography)
    aoi_geo = _standardize_geographic_level_input(geo_df, aoi_geography)

    # consolidate selection
    where_clause = _get_geographic_level_where_clause(selector, selection_field, query_string)
    aoi_where_clause = _get_geographic_level_where_clause(aoi_selector, aoi_selection_field, aoi_query_string)

    return geo, aoi_geo, where_clause, aoi_where_clause


class Country:
    """Country objects are instantiated by providing the three letter country identifier
    and optionally also specifying the source. If the source is not explicitly specified
    Country will use local resources if the environment is part of an ArcGIS Pro
    installation with Business Analyst enabled and local data installed. If this is not
    the case, Country will then attempt to use the active GIS object instance if available.
    Also, if a GIS object is explicitly passed in, this will be used.

    Args:
        name: Three letter country identifier.
        source: Optional 'local' or a GIS object instance referencing an ArcGIS Enterprise
            instance with enrichment configured or ArcGIS Online. If not explicitly
            specified, will attempt to use locally installed data with Pro and Business
            analyst first. If this is not available, will look for an active GIS. If
            no active GIS, a GIS object will need to be explicitly provided with
            permissions to perform enrichment.
        year: Optional and only applicable if using local data. In cases where models
            have been developed against a specific vintage (year) of data, this affords the
            ability to enrich data for this specific year to support these models.

    .. code-block:: python

        from arcgis.modeling import Country

        # instantiate a country
        usa = Country('USA', source='local')

        # get the seattle CBSA as a study area
        aoi_df = usa.cbsas.get('seattle')

        # use the Modeling DataFrame accessor to retrieve the block groups in seattle
        bg_df = aoi_df.mdl.block_groups.get()

        # get the available enrich variables as as DataFrame
        e_vars = usa.enrich_variables

        # filter the variables to just the current year key variables
        key_vars = e_vars[(e_vars.data_collection.str.startswith('Key')) &
                          (e_vars.name.str.endswith('CY'))]

        # enrich the data through the Modeling DataFrame accessor
        e_df = bg_df.mdl.enrich(key_vars)

    """

    def __init__(self, name: str, source: Union[str, GIS] = None, year: int = None):

        # set the source implicitly if necessary based on what is available
        self.source = set_source(source)

        # make sure the source is either local or an GIS instance
        source_err = 'The source must either be set to "local" or a GIS object instance.'
        assert isinstance(self.source, (str, GIS)), source_err
        if isinstance(self.source, str):
            self.source = self.source.lower()  # account for possibility of entering caps for some reason
            assert self.source == 'local', source_err

        # ensure not trying to set the year for using online data
        if isinstance(self.source, GIS) and year is not None:
            raise Exception('A year can only be explicitly defined when working with locally installed resources, '
                            'not with a GIS instance.')

        # make sure the year, if provided, is an integer
        if year is not None:
            assert isinstance(year, int), f'The year parameter must be an integer, not {type(year)}'

        self.geo_name = name.upper()
        self.year = year
        self.dataset_id = None
        self._enrich_variables = None
        self._geo_id = None
        self._geography_levels = None
        self._business = None
        self._cntry_df = get_countries(self.source)

        # set the geo_name to the iso3 value if the iso2 was provided
        if self.geo_name in set(self._cntry_df['iso2'].values):
            self.geo_name = self._cntry_df[self._cntry_df.iso2 == self.geo_name].iloc[0]['iso3']

        # ensure the input country name is valid
        assert self.geo_name in self._cntry_df.iso3.drop_duplicates().sort_values().values, \
            'Please choose a valid three letter country identifier (ISO3). You can get a list of valid values ' \
            'using the "modeling.get_countries" method.'

        # if the data source is local, but no year was provided, get the year
        if self.source == 'local' and self.year is None:
            self.year = self._cntry_df[self._cntry_df['iso3'] == self.geo_name].vintage.max()

        # if the year is provided, validate
        elif self.source == 'local' and self.year is not None:
            lcl_yr_vals = (self._cntry_df[self._cntry_df['iso3'] == self.geo_name]['vintage']).values
            assert self.year in lcl_yr_vals, f'The year you are provided, {self.year} is not among the available ' \
                                             f'years ({", ".join([str(v) for v in lcl_yr_vals])}) for {self.geo_name}'

        # get the geo_id, the identifier BA uses to know what dataset to use when using local data
        if self.source == 'local':
            self._geo_id = self._cntry_df[
                (self._cntry_df['iso3'] == self.geo_name)
                & (self._cntry_df['vintage'] == self.year)
                ].iloc[0]['country_id']

        # set the iso2 property
        self.iso2 = self._cntry_df[self._cntry_df.iso3 == self.geo_name].iloc[0]['iso2']

        # grab the default dataset id, needed for working with online data
        self.dataset_id = self._cntry_df[self._cntry_df.iso3 == self.geo_name].iloc[0]['country_id']

        # add on all the geographic resolution levels as properties
        for nm in self.geography_levels['geo_name']:
            setattr(self, nm, GeographyLevel(nm, self))

    def __repr__(self):
        if self.source == 'local':
            repr_str = f'<modeling.Country - {self.geo_name} ({self.source} {self.year})>'
        elif isinstance(self.source, GIS) and self.source.users.me is None:
            repr_str = f'<modeling.Country - {self.geo_name} (GIS at {self.source.url} )>'
        else:
            gis = self.source
            repr_str = f'<modeling.Country - {self.geo_name} (GIS at {gis.url} logged in as {gis.users.me.username})>'
        return repr_str

    def _set_arcpy_ba_country(self):
        """Helper function to set the country in ArcPy."""
        arcpy.env.baDataSource = f'LOCAL;;{self.dataset_id}'
        return

    @property
    def _enrich_variables_local(self):
        """Local implementation getting enrichment variables."""
        # retrieve variable objects
        var_gen = arcpy._ba.ListVariables(self._geo_id)

        # use a list comprehension to unpack the properties of the variables into a dataframe
        var_df = pd.DataFrame(
            ((v.Name, v.Alias, v.DataCollectionID, v.FullName, v.OutputFieldName) for v in var_gen),
            columns=['name', 'alias', 'data_collection', 'enrich_name', 'enrich_field_name']
        )

        return var_df

    @property
    def _enrich_variables_gis(self):
        """GIS implementation getting enrichment variables."""
        # get the data collections from the GIS enrichment REST endpiont
        params = {'f': 'json'}
        res = self.source._con.get(
            f'{self.source.properties.helperServices("geoenrichment").url}/Geoenrichment/DataCollections/{self.iso2}',
            params=params)
        assert 'DataCollections' in res.keys(), 'Could not retrieve enrichment variables (DataCollections) from ' \
                                                'the GIS instance.'

        # list to store all the dataframes as they are created for each data collection
        mstr_lst = []

        # iterate the data collections
        for col in res['DataCollections']:
            # create a dataframe of the variables, keep only needed columns, and add the data collection name
            coll_df = pd.json_normalize(col['data'])[['id', 'alias', 'description', 'vintage', 'units']]
            coll_df['data_collection'] = col['dataCollectionID']

            # schema cleanup
            coll_df.rename(columns={'id': 'name'}, inplace=True)
            coll_df = coll_df[['name', 'alias', 'data_collection', 'description', 'vintage', 'units']]

            # append the list
            mstr_lst.append(coll_df)

        # combine all the dataframes into a single master dataframe
        mstr_df = pd.concat(mstr_lst)

        # create the column for enrichment
        mstr_df.insert(3, 'enrich_name', mstr_df.data_collection + '.' + mstr_df.name)

        # create column for matching to previously enriched column names
        regex = re.compile(r"(^\d+)")
        fld_vals = mstr_df.enrich_name.apply(lambda val: regex.sub(r"F\1", val.replace(".", "_")))
        mstr_df.insert(4, 'enrich_field_name', fld_vals)

        return mstr_df

    def add_enrich_aliases(self, feature_class: Union[Path, str]) -> Path:
        """
        Add human readable aliases to an enriched feature class.

        .. note::

            This function requires ArcPy to be available in the environment, an environment created on a Windows
            operating system with ArcGIS Pro installed and ArcPy installed in the environment.

        Args:
            feature_class: Path to the enriched feature class.

        Returns:
            Path to feature class with aliases added.

        .. code-block:: python

            from modeling import Country

            out_fc_pth = Path(r'C:/path/to/geodatabase.gdb/block_groups')

            # create a country object
            cntry =  Country('USA')

            # get the current year key variables
            evars = cntry.enrich_variables
            key_vars = e_vars[
                (e_vars.data_collection.str.startswith('Key'))
                & (e_vars.name.str.endswith('CY'))
            ]

            # get the block groups for the area of interest
            bg_df = cntry.cbsas.get('seattle').mdl.block_groups.get()

            # enrich the block groups with the key variables
            enrich_df = bg_df.mdl.enrich(key_vars)

            # save to a feature class
            enrich_fc = enrich_df.spatial.to_featureclass(out_fc_pth)

            # finally, add enrich aliases to the output feature class
            cntry.add_aliases(enrich_fc)

        """
        # make sure arcpy is available because we need it
        assert avail_arcpy, 'add_enrich_aliases requires arcpy to be available since working with ArcGIS Feature ' \
                            'Classes'

        # since arcpy tools cannot handle Path objects, convert to string
        feature_class = str(feature_class) if isinstance(feature_class, Path) else feature_class

        # if, at this point, the path is not a string, something is wrong
        if not isinstance(feature_class, str):
            raise Exception(f'The feature_class must be either a Path or string, not {type(feature_class)}.')

        # start by getting a list of all the field names
        fld_lst = [f.name for f in arcpy.ListFields(feature_class)]

        # iterate through the field names and if the field is an enrich field, add the alias
        for fld_nm in fld_lst:

            # pop out the dataframe of enrich variables for readability
            e_var = self.enrich_variables

            # pull out the enrich field name series since we're going to use it a lot
            e_nm = e_var.enrich_field_name

            # get a dataframe, a single or no row dataframe, correlating to the field name
            fld_df = e_var[e_nm.str.replace('_', '').str.contains(fld_nm.replace('_', ''), case=False)]

            # if no field was found, try pattern for non-modified fields - provides pre-ArcGIS Python API 1.8.3 support
            if len(fld_df.index) == 0:
                fld_df = e_var[e_nm.str.contains(fld_nm, case=False)]

            # if the field name was found, add the alias
            if len(fld_df.index):
                arcpy.management.AlterField(
                    in_table=feature_class,
                    field=fld_nm,
                    new_field_alias=fld_df.iloc[0]['alias']
                )

        # convert path back to string for output
        feature_class = Path(feature_class)

        return feature_class

    def verify_can_enrich(self):
        """If the country enrich instance can enrich based on the permissions. Only relevant if source is a GIS
        instance."""
        if isinstance(self.source, GIS):
            can_e = can_enrich_gis(self.source.users.me) if self.source.users.me is not None else False
        else:
            can_e = True
        return can_e

    def verify_can_perform_network_analysis(self, network_function: str = None) -> bool:
        """If the country enrich instance can perform transportation network analysis based on permissions. Only
        relevant if the source is a GIS instance.

        Args:
            network_function: Optional string describing specific network function to check for. Valid values include
                'closestfacility', 'locationallocation', 'optimizedrouting', 'origindestinationcostmatrix', 'routing',
                'servicearea', or 'vehiclerouting'.

        Returns:
            Boolean indicating if the country instance, based on permissions, has network analysis privileges.
        """
        if isinstance(self.source, GIS):
            if self.source.users.me is not None:
                can_net = has_networkanalysis_gis(self.source.users.me, network_function)
            else:
                can_net = False
        else:
            can_net = True
        return can_net

    @property
    def geography_levels(self):
        """DataFrame of available geography levels."""
        if self._geography_levels is None and self.source == 'local':
            from ._xml_interrogation import get_heirarchial_geography_dataframe
            self._geography_levels = get_heirarchial_geography_dataframe(self.geo_name, self.year)

        # if source is a GIS instance
        elif self._geography_levels is None and isinstance(self.source, GIS):

            # unpack the geoenrichment url from the properties
            enrich_url = self.source.properties.helperServices.geoenrichment.url

            # construct the url to the standard geography levels
            url = f'{enrich_url}/Geoenrichment/standardgeographylevels'

            # get the geography levels from the enrichment server
            res_json = self.source._con.post(url, {'f': 'json'})

            # unpack the geography levels from the json
            geog_lvls = res_json['geographyLevels']

            # get matching geography levels out of the list of countries
            for lvl in geog_lvls:
                if lvl['countryID'] == self.iso2:
                    geog = lvl
                    break

            # get the hierarchical levels out as a dataframe
            self._geography_levels = pd.DataFrame(geog['hierarchies'][0]['levels'])

            # create the geo_name to use for identifying the levels
            self._geography_levels['geo_name'] = self._geography_levels['name'].str.lower(). \
                str.replace(' ', '_', regex=False).str.replace('(', '', regex=False).str.replace(')', '', regex=False)

            # reverse the sorting so the smallest is at the top
            self._geography_levels = self._geography_levels.iloc[::-1].reset_index(drop=True)

            # clean up the field names so they follow more pythonic conventions
            self._geography_levels = self._geography_levels[['geo_name', 'name', 'adminLevel', 'singularName',
                                                             'pluralName', 'id']].copy()
            self._geography_levels.columns = ['geo_name', 'geo_alias', 'admin_level', 'singular_name', 'plural_name',
                                              'id']

        return self._geography_levels

    @property
    def levels(self):
        """Dataframe of available geography levels. (Alias of geography_levels.)"""
        return self.geography_levels

    @property
    def enrich_variables(self):
        """
        DataFrame of all the available enrichment variables.

        .. code-block:: python

            from arcgis import GIS
            from modeling import Country

            # connect to an Enterprise GIS instance with Business Analyst installed
            gis = GIS('https://mydomain.com/portal', username='geowizard', password='Y3ll0wBr!ck$')

            # create a country to work with
            cntry = Country('USA')

            # get the available enrich variables as as DataFrame
            e_vars = usa.enrich_variables

            # filter the variables to just the current year key variables
            key_vars = e_vars[(e_vars.data_collection.str.startswith('Key')) &
                              (e_vars.name.str.endswith('CY'))]

        The ``key_vars`` table retrieved using the code sample above will look similar to the following.

        ====  ==========  =================================  =================  =====================  =====================  ===============================================  =========  ========
          ..  name        alias                              data_collection    enrich_name            enrich_field_name      description                                        vintage  units
        ====  ==========  =================================  =================  =====================  =====================  ===============================================  =========  ========
           0  TOTPOP_CY   2020 Total Population              KeyUSFacts         KeyUSFacts.TOTPOP_CY   KeyUSFacts_TOTPOP_CY   2020 Total Population (Esri)                          2020  count
           1  GQPOP_CY    2020 Group Quarters Population     KeyUSFacts         KeyUSFacts.GQPOP_CY    KeyUSFacts_GQPOP_CY    2020 Group Quarters Population (Esri)                 2020  count
           2  DIVINDX_CY  2020 Diversity Index               KeyUSFacts         KeyUSFacts.DIVINDX_CY  KeyUSFacts_DIVINDX_CY  2020 Diversity Index (Esri)                           2020  count
           3  TOTHH_CY    2020 Total Households              KeyUSFacts         KeyUSFacts.TOTHH_CY    KeyUSFacts_TOTHH_CY    2020 Total Households (Esri)                          2020  count
           4  AVGHHSZ_CY  2020 Average Household Size        KeyUSFacts         KeyUSFacts.AVGHHSZ_CY  KeyUSFacts_AVGHHSZ_CY  2020 Average Household Size (Esri)                    2020  count
           5  MEDHINC_CY  2020 Median Household Income       KeyUSFacts         KeyUSFacts.MEDHINC_CY  KeyUSFacts_MEDHINC_CY  2020 Median Household Income (Esri)                   2020  currency
           6  AVGHINC_CY  2020 Average Household Income      KeyUSFacts         KeyUSFacts.AVGHINC_CY  KeyUSFacts_AVGHINC_CY  2020 Average Household Income (Esri)                  2020  currency
           7  PCI_CY      2020 Per Capita Income             KeyUSFacts         KeyUSFacts.PCI_CY      KeyUSFacts_PCI_CY      2020 Per Capita Income (Esri)                         2020  currency
           8  TOTHU_CY    2020 Total Housing Units           KeyUSFacts         KeyUSFacts.TOTHU_CY    KeyUSFacts_TOTHU_CY    2020 Total Housing Units (Esri)                       2020  count
           9  OWNER_CY    2020 Owner Occupied HUs            KeyUSFacts         KeyUSFacts.OWNER_CY    KeyUSFacts_OWNER_CY    2020 Owner Occupied Housing Units (Esri)              2020  count
          10  RENTER_CY   2020 Renter Occupied HUs           KeyUSFacts         KeyUSFacts.RENTER_CY   KeyUSFacts_RENTER_CY   2020 Renter Occupied Housing Units (Esri)             2020  count
          11  VACANT_CY   2020 Vacant Housing Units          KeyUSFacts         KeyUSFacts.VACANT_CY   KeyUSFacts_VACANT_CY   2020 Vacant Housing Units (Esri)                      2020  count
          12  MEDVAL_CY   2020 Median Home Value             KeyUSFacts         KeyUSFacts.MEDVAL_CY   KeyUSFacts_MEDVAL_CY   2020 Median Home Value (Esri)                         2020  currency
          13  AVGVAL_CY   2020 Average Home Value            KeyUSFacts         KeyUSFacts.AVGVAL_CY   KeyUSFacts_AVGVAL_CY   2020 Average Home Value (Esri)                        2020  currency
          14  POPGRW10CY  2010-2020 Growth Rate: Population  KeyUSFacts         KeyUSFacts.POPGRW10CY  KeyUSFacts_POPGRW10CY  2010-2020 Population: Annual Growth Rate (Esri)       2020  pct
          15  HHGRW10CY   2010-2020 Growth Rate: Households  KeyUSFacts         KeyUSFacts.HHGRW10CY   KeyUSFacts_HHGRW10CY   2010-2020 Households: Annual Growth Rate (Esri)       2020  pct
          16  FAMGRW10CY  2010-2020 Growth Rate: Families    KeyUSFacts         KeyUSFacts.FAMGRW10CY  KeyUSFacts_FAMGRW10CY  2010-2020 Families: Annual Growth Rate (Esri)         2020  pct
          17  DPOP_CY     2020 Total Daytime Population      KeyUSFacts         KeyUSFacts.DPOP_CY     KeyUSFacts_DPOP_CY     2020 Total Daytime Population (Esri)                  2020  count
          18  DPOPWRK_CY  2020 Daytime Pop: Workers          KeyUSFacts         KeyUSFacts.DPOPWRK_CY  KeyUSFacts_DPOPWRK_CY  2020 Daytime Population: Workers (Esri)               2020  count
          19  DPOPRES_CY  2020 Daytime Pop: Residents        KeyUSFacts         KeyUSFacts.DPOPRES_CY  KeyUSFacts_DPOPRES_CY  2020 Daytime Population: Residents (Esri)             2020  count
        ====  ==========  =================================  =================  =====================  =====================  ===============================================  =========  ========

        """
        if self._enrich_variables is None and self.source == 'local':
            self._enrich_variables = self._enrich_variables_local

        elif self._enrich_variables is None and isinstance(self.source, GIS):
            self._enrich_variables = self._enrich_variables_gis

        # add on the country for subsequent analysis in the dataframe metatdata
        self._enrich_variables.attrs['_cntry'] = self

        return self._enrich_variables

    @local_vs_gis
    def level(self, geography_index: (str, int)) -> pd.DataFrame:
        """
        Get an available ``geography_level`` in the country.

        Args:
            geography_index:
                Either the geographic_level geo_name or the
                index of the geography_level level. This can be discovered
                using the ``Country.geography_levels`` method.

        Returns:
            Spatially Enabled DataFrame of the requested geography_level with
            the Modeling accessor properties initialized.
        """
        pass

    def _level_local(self, geography_index: Union[str, int]) -> pd.DataFrame:
        """Local implementation of level."""
        # get the geo_name of the geography
        if isinstance(geography_index, int):
            nm = self.geography_levels.iloc[geography_index]['geo_name']

        elif isinstance(geography_index, str):
            nm = geography_index
        else:
            raise Exception(f'geography_index must be either a string or integer, not {type(geography_index)}')

        # create a GeographyLevel object instance
        return GeographyLevel(nm, self)

    @local_vs_gis
    def enrich(self, data: pd.DataFrame,
               enrich_variables: Union[list, np.array, pd.Series, pd.DataFrame],
               enrich_geography_level: str = None, enrich_id_column: str = None) -> pd.DataFrame:
        """
        Enrich a spatially enabled dataframe using either a enrichment
        variables defined using a Python List, NumPy Array or Pandas
        Series of enrich names. Also, a filtered enrich variables Pandas
        DataFrame can also be used.

        Args:
            data:
                Spatially Enabled DataFrame with geography_levels to be
                enriched.
            enrich_variables:
                Optional iterable of enrich variables to use for
                enriching data. Filtered output from
                Country.enrich_variables can also be used.
            enrich_geography_level: If using a standard level of geography,
                the geography level to use. This is necessary only when
                specifying an ``enrich_id_column`` as well. This is not
                necessary if the standard geographies were created as
                part of a data preperation workflow using the
                ModelingAccesor.
            enrich_id_column: If specifying to use a standard level of
                geography, the column in the source data containing the
                identifying values correlated to the geographic areas.
                This is not necessary if the standard geographies were
                created as part of a data preperation workflow using the
                ModelingAccesor.


        Returns:
            Spatially Enabled DataFrame with enriched data added.
        """
        pass

    def get_enrich_variables_dataframe_from_variable_list(self,
                                                          enrich_variables: Union[list, tuple, np.ndarray, pd.Series],
                                                          drop_duplicates=True) -> pd.DataFrame:
        """Get a dataframe of enrich variables associated with the list of variables
        passed in. This is especially useful when needing aliases (*human readable
        names*), or are interested in enriching more data using previously enriched
        data as a template.

        Args:
            enrich_variables: Iterable (normally a list) of variables correlating to
                enrichment variables. These variable names can be simply the name, the
                name prefixed by the collection separated by a dot, or the output from
                enrichment in ArcGIS Pro with the field name modified to fit field naming
                and length constraints.
            drop_duplicates: Optional boolean (default True) indicating whether to drop
                duplicates. Since the same variables appear in multiple data collections,
                multiple instances of the same variable can be found. Dropping duplicates
                removes redundant matches.

        Returns:
            Pandas DataFrame of enrich variables with the different available aliases.

        .. code-block:: python

            from pathlib import Path

            import arcpy
            from modeling import Country

            # path to previously enriched data
            enriched_fc_pth = Path(r'C:/path/to/geodatabase.gdb/enriched_data')
            new_fc_pth = Path(r'C:/path/to/geodatabase.gdb/block_groups_pdx')

            # get a list of column names from previously enriched data
            attr_lst = [c.name for c in arcpy.ListFields(str(enriched_fc_pth))

            # get a country to work in
            cntry = Country('USA', source='local')

            # get dataframe of variables used for previously enriched data
            enrich_vars = cntry.get_enrich_variables_dataframe_from_variable_list(attr_lst)

            # enrich block groups in new area of interest using the same variables
            enrich_df = cntry.cbsas.get('portland-vancouver').mdl.block_groups.get().mdl.enrich(enrich_vars)

            # save the new data and add aliases
            enrich_df.spatial.to_featureclass(new_fc_pth)
            cntry.add_enrich_aliases(new_fc_pth)

        """
        # enrich variable dataframe column names
        enrich_nm_col, enrich_nmpro_col, enrich_str_col = 'name', 'enrich_field_name', 'enrich_name'
        col_nm_san, col_pronm_san, col_estr_san = 'nm_san', 'nmpro_san', 'estr_san'

        # get shorter version of variable name to work with and also one to modify if necessary
        ev_df = self.enrich_variables

        # get a series of submitted enrich varialbes all lowercase to account for case variations
        ev_lower = enrich_variables.str.lower()

        # default to trying to find enrich variables using the enrich string values
        enrich_vars_df = ev_df[ev_df[enrich_str_col].str.lower().isin(ev_lower)]

        # if nothing was returned, try using just the variable names (common if using previously enriched data)
        if len(enrich_vars_df.index) == 0:
            enrich_vars_df = ev_df[ev_df[enrich_nm_col].str.lower().isin(ev_lower)]

        # the possibly may exist where names are from columns in data enriched using local enrichment in Pro
        if len(enrich_vars_df.index) == 0:
            enrich_vars_df = ev_df[ev_df[enrich_nmpro_col].str.lower().isin(ev_lower)]

        # try to find enrich variables using the enrich string values sanitized (common if exporting from SeDF)
        if len(enrich_vars_df.index) == 0:
            ev_df[col_estr_san] = get_sanitized_names(ev_df[enrich_str_col])
            enrich_vars_df = ev_df[ev_df[col_estr_san].str.lower().isin(ev_lower)]

        # try columns in data enriched using local enrichment in Pro and sanitized (common if exporting from SeDF)
        if len(enrich_vars_df.index) == 0:
            ev_df[col_pronm_san] = get_sanitized_names(enrich_nmpro_col)
            enrich_vars_df = ev_df[ev_df[col_pronm_san].isin(ev_lower)]

        # if nothing was returned, try using just the variable names possibly sanitized (anticipate this to be rare)
        if len(enrich_vars_df.index) == 0:
            ev_df[col_nm_san] = get_sanitized_names(ev_df[enrich_nm_col])
            enrich_vars_df = ev_df[ev_df[col_nm_san].str.lower().isin(ev_lower)]

        # make sure something was found, but don't break the runtime
        if len(enrich_vars_df) == 0:
            warn(f'It appears none of the input enrich variables were found in the {self.source} country.', UserWarning)

        # if drop_duplicates, drop on variable name column to remove redundant matches
        if drop_duplicates:
            enrich_vars_df = enrich_vars_df.drop_duplicates(enrich_nm_col)

        # clean up the index
        enrich_vars_df.reset_index(drop=True, inplace=True)

        return enrich_vars_df

    def _enrich_variable_preprocessing(self, enrich_variables: Union[list, tuple, np.ndarray, pd.Series, pd.DataFrame]):
        """Provide flexibility for enrich variable preprocessing."""
        # enrich variable dataframe column name
        enrich_str_col = 'enrich_name'
        enrich_nm_col = 'name'

        # if just a single variable is provided pipe it into a pandas series
        if isinstance(enrich_variables, str):
            enrich_variables = pd.Series([enrich_variables])

        # toss the variables into a pandas Series if an iterable was passed in
        elif isinstance(enrich_variables, (list, tuple, np.ndarray)):
            enrich_variables = pd.Series(enrich_variables)

        # if the enrich dataframe is passed in, check to make sure it has what we need, the right columns
        if isinstance(enrich_variables, pd.DataFrame):
            assert enrich_str_col in enrich_variables.columns, f'It appears the dataframe used for enrichment does' \
                                                               f' not have the column with enrich string names ' \
                                                               f'({enrich_str_col}).'
            assert enrich_nm_col in enrich_variables.columns, f'It appears the dataframe used for enrichment does ' \
                                                              f'not have the column with the enrich variables names ' \
                                                              f'({enrich_nm_col}).'
            enrich_vars_df = enrich_variables

        # otherwise, create a enrich variables dataframe from the enrich series for a few more checks
        else:

            # get the enrich variables dataframe
            enrich_vars_df = self.get_enrich_variables_dataframe_from_variable_list(enrich_variables)

        # now, drop any duplicates so we're not getting the same variable twice from different data collections
        enrich_vars_df = enrich_vars_df.drop_duplicates('name').reset_index(drop=True)

        # note any variables submitted, but not found
        if len(enrich_variables) > len(enrich_vars_df.index):
            missing_count = len(enrich_variables) - len(enrich_vars_df.index)
            warn('Some of the variables provided are not available for enrichment '
                 f'(missing count: {missing_count:,}).', UserWarning)

        # check to make sure there are variables for enrichment
        if len(enrich_vars_df.index) == 0:
            raise Exception('There appear to be no variables selected for enrichment.')

        # get a list of the variables for enrichment
        enrich_variables = enrich_vars_df[enrich_str_col].reset_index()[enrich_str_col]

        return enrich_variables

    def _enrich_local(self, data: pd.DataFrame,
                      enrich_variables: Union[list, np.array, pd.Series, pd.DataFrame] = None,
                      enrich_geography_level: str = None, enrich_id_column: str = None) -> pd.DataFrame:
        """Implementation of enrich for local analysis."""
        # TODO: Implement - likely will require using a SQL query against the local layer
        if enrich_geography_level or enrich_id_column:
            raise NotImplemented('using enrich_geography_level and enrich_id_column is not yet implemented with local '
                                 'data.')

        # preprocess and combine all the enrichment variables into a single string for input into the enrich tool
        evars = self._enrich_variable_preprocessing(enrich_variables)
        enrich_str = ';'.join(evars)

        # convert the geometry column to a list of arcpy geometry objects
        geom_lst = list(data['SHAPE'].apply(lambda geom: geom.as_arcpy).values)

        # set the arcpy environment to the correct country
        self._set_arcpy_ba_country()

        # invoke the enrich method to get the data
        enrich_fc = arcpy.ba.EnrichLayer(
            in_features=geom_lst,
            out_feature_class='memory/enrich_tmp',
            variables=enrich_str
        )[0]

        # get the Object ID field for schema cleanup
        oid_col = arcpy.Describe(enrich_fc).OIDFieldName

        # convert the enrich feature class to a dataframe and do some schema cleanup
        enrich_df = GeoAccessor.from_featureclass(enrich_fc)
        drop_cols = [c for c in enrich_df.columns if c in [oid_col, 'HasData', 'aggregationMethod', 'SHAPE']]
        enrich_df.drop(columns=drop_cols, inplace=True)

        # combine the two dataframes for output
        out_df = pd.concat([data, enrich_df], axis=1, sort=False)

        # organize the columns so geometry is the last column
        out_df = out_df[[c for c in out_df.columns if c != 'SHAPE'] + ['SHAPE']]

        # ensure this dataframe will be recognized as spatially enabled
        out_df.spatial.set_geometry('SHAPE')

        # ensure WGS84
        out_data = out_df.mdl.project(4326)

        # rename the columns for schema consistency with REST responses
        e_vars_df = self.enrich_variables[[n in out_df.columns for n in self.enrich_variables.enrich_field_name]]
        col_nm_map = pd.Series(e_vars_df['name'].values, index=e_vars_df['enrich_field_name']).to_dict()
        out_df.rename(col_nm_map, axis=1, inplace=True)

        # add the country onto the metadata
        out_data.attrs['_cntry'] = self

        return out_data

    def _enrich_gis(self, data: pd.DataFrame,
                    enrich_variables: Union[list, np.array, pd.Series, pd.DataFrame] = None,
                    enrich_geography_level: str = None, enrich_id_column: str = None) -> pd.DataFrame:
        """Implementation of enrich for analysis using Web GIS."""
        if enrich_geography_level or enrich_id_column:
            assert enrich_geography_level and enrich_id_column, 'Both enrich_geography_level and enrich_id_column ' \
                                                                'must be specified to use standard ids.'

        assert enrich_id_column in data.columns, f'The enrich_id_column provided, "{enrich_id_column}," does not ' \
                                                 f'appear to be an available column.'

        # get the geography level and id's are explicitly provided
        if enrich_geography_level and enrich_id_column:

            # get the geography level row from the geography levels dataframe by the geo_name
            geo_lvl_df = self.geography_levels[self.geography_levels.geo_name == enrich_geography_level]

            # if, for some reason, the acutal geography level id was passed in, check for this as well
            if len(geo_lvl_df.index) == 0:
                geo_lvl_df = self.geography_levels[self.geography_levels.id == enrich_geography_level]

            # pitch a fit if no geography level was found, but try to offer helpful suggestions
            assert len(geo_lvl_df.index) > 0, f'The specified geography level, {enrich_geography_level} does not ' \
                                              f'appear to be one of the available geography levels. This must be a ' \
                                              f'value from the "geo_name" column in teh dataframe available from the ' \
                                              f'"geography_levels" property of the Country object.'

            # get the actual geography level id
            geo_lvl = geo_lvl_df.iloc[0].id

            # create the list of ids to be used for enrichment
            id_lst = list(data[enrich_id_column])

        # check if they are in the attrs from a previous ModelingAccessor function output
        elif 'parent_geo' in data.attrs.keys():
            geo_lvl = data.attrs['parent_geo']['resource']
            id_lst = data.attrs['parent_geo']['id']

        # otherwise, set to null - makes it easier to check later
        else:
            geo_lvl, id_lst = None, None

        evars = self._enrich_variable_preprocessing(enrich_variables)

        # get the maximum batch size less one just for good measure
        res = self.source._con.get(
            f'{self.source.properties.helperServices("geoenrichment").url}/Geoenrichment/ServiceLimits')
        std_batch_size = [v['value'] for v in res['serviceLimits']['value'] if v['paramName'] == 'maxRecordCount'][0]
        geom_batch_size = [v['value'] for v in res['serviceLimits']['value']
                           if v['paramName'] == 'optimalBatchStudyAreasNumber'][0]

        # not sure why, but it struggles with larger datasets
        geom_batch_size = math.floor(geom_batch_size * 0.5)

        # initialize the params for the REST call
        params = {
            'f': 'json',
            'analysisVariables': list(evars),
            'returnGeometry': False  # because we already have the geometry
        }

        # list to store inputs and results
        in_req_list, out_df_lst = [], []

        # if working with data derived from standard geographies
        if geo_lvl and id_lst:

            # use the count of features and the max batch size to iteratively enrich the input data
            for x in range(0, len(data.index), std_batch_size):

                # peel off just the id's for this batch
                batch_id_lst = id_lst[x: x + std_batch_size]

                params['studyAreas'] = [{
                    "sourceCountry": geo_lvl.split('.')[0],
                    "layer": geo_lvl,
                    "ids": batch_id_lst
                }]

                # add the payload onto the list
                in_req_list.append(params)

        # if not standard geographies, working with geometries
        else:

            # validate the spatial property
            assert data.spatial.name is not None, 'The input data does not appear to be a valid Spatially Enabled ' \
                                                  'DataFrame. Possibly try df.spatial.set_geometry("SHAPE") to rectify.'

            # use the count of features and the max batch size to iteratively enrich the input data
            for x in range(0, len(data.index), geom_batch_size):

                # get a slice of the input data to enrich for this
                in_batch_df = data.loc[x: x + geom_batch_size]

                # format the features for sending - keep it light, just the geometry
                features = in_batch_df[in_batch_df.spatial.name].to_frame().spatial.to_featureset().features
                params['studyAreas'] = [f.as_dict for f in features]

                # get the input spatial reference
                params['insr'] = data.spatial.sr

                # add the payload onto the list
                in_req_list.append(params)

        # iterate the packaged request payloads
        for idx, req_params in enumerate(in_req_list):

            # track failed request attempts, and put attempt count in a variable
            fail_cnt = 0
            attempt_cnt = 5

            # account for it sometimes just not working
            while fail_cnt <= attempt_cnt:

                # enable a little resiliency if encountering issues
                try:

                    # send the request to the server using post because if sending geometry, the message can be big
                    enrich_url = self.source.properties.helperServices("geoenrichment").url
                    r_json = self.source._con.post(f'{enrich_url}/Geoenrichment/Enrich', params=req_params)

                    # ensure a valid result is received
                    if 'error' in r_json:
                        err = r_json['error']
                        raise Exception(f'Error in enriching data using Business Analyst Enrich REST endpoint. Error '
                                        f'Code {err["code"]}: {err["message"]}')

                    # unpack the enriched results - reaching into the FeatureSet for just the attributes
                    r_df = pd.DataFrame([f['attributes'] for f in
                                         r_json['results'][0]['value']['FeatureSet'][0]['features']])

                    # if earlier errors were encountered, reset the counter
                    fail_cnt = 0

                # if something goes wrong
                except Exception as e:

                    # increment the fail count
                    fail_cnt += 1

                    # bomb out if reached max attempts, and if not, still report results with warning
                    if fail_cnt == attempt_cnt:
                        raise Exception(e)
                    else:
                        warn(f'Batch: ({idx/len(in_req_list)}) - Failed attempt: ({fail_cnt}/{attempt_cnt}) - Error: {e}')

            # get just the columns with the enrich data requested
            evar_mstr = self.enrich_variables
            evars_df = evar_mstr[[n in evars.str.lower().values for n in evar_mstr['enrich_name'].str.lower()]]
            e_col_lst = [n for n in r_df.columns if n in evars_df['name'].values]

            # filter the response dataframe to just enrich columns
            e_df = r_df[e_col_lst]

            # add the dataframe to the list
            out_df_lst.append(e_df)

        # combine all the received enriched data and add onto the original data
        enrich_df = pd.concat(out_df_lst).reset_index(drop=True)
        out_df = pd.concat([data, enrich_df], axis=1, sort=False)

        # if there is geometry, which there may not be since can enrich using just ids
        if 'SHAPE' in out_df.columns:

            # shuffle columns so geometry is at the end
            out_df = out_df[[c for c in out_df.columns if c != 'SHAPE'] + ['SHAPE']]

            # set the geometry so the GeoAccessor knows it is an SeDF
            out_df.spatial.set_geometry('SHAPE')

        # tack on metadata for potential future processing
        out_df.attrs = data.attrs

        return out_df


class GeographyLevel:

    def __init__(self, geographic_level: (str, int), country: Country, parent_data: (pd.DataFrame, pd.Series) = None):
        self._cntry = country
        self.source = country.source
        self.geo_name = self._standardize_geographic_level_input(geographic_level)
        self._resource = None
        self._parent_data = parent_data

    def __repr__(self):
        return f'<class: GeographyLevel - {self.geo_name}>'

    @staticmethod
    def _format_get_selectors(selector: Union[str, int, list, tuple]) -> Union[str, list]:
        """Helper to format get selectors"""
        if isinstance(selector, (list, tuple)):

            # make sure all values are strings
            if isinstance(selector[0], int):
                selector = [str(v) for v in selector]

            # ensure if digits, are left padded with zeros
            if all(v.isdecimal() for v in selector):
                max_len = max(len(v) for v in selector)
                selector = [v.zfill(max_len) for v in selector]

        elif isinstance(selector, int):
            selector = str(selector)

        return selector

    def _standardize_geographic_level_input(self, geo_in: Union[str, int]) -> str:
        """Helper function to check and standardize named input or integers to geographic heirarchial levels."""

        geo_df = self._cntry.geography_levels

        if isinstance(geo_in, str):
            if geo_in not in geo_df.geo_name.values:
                names = ', '.join(geo_df.geo_name.values)
                raise Exception(
                    f'Your selector, "{geo_in}," is not an available selector. Please choose from {names}.')
            geo_lvl_name = geo_in

        elif isinstance(geo_in, int) or isinstance(geo_in, float):
            if geo_in > len(geo_df.index):
                raise Exception(
                    f'Your selector, "{geo_in}", is beyond the maximum range of available geography_levels.')
            geo_lvl_name = geo_df.iloc[geo_in]['geo_name']

        else:
            raise Exception('The geographic selector must be a string or integer.')

        return geo_lvl_name

    @property
    def resource(self):
        """The resource, either a layer or Feature Layer, for accessing the data for the geographic layer."""
        # get the geography levels dataframe to work with
        lvl_df = self._cntry.geography_levels

        # get the feature class path if local
        if self._resource is None and self._cntry.source == 'local':
            self._resource = lvl_df[lvl_df['geo_name'] == self.geo_name].iloc[0]['feature_class_path']

        elif self._resource is None and isinstance(self._cntry.source, GIS):
            self._resource = lvl_df[lvl_df['geo_name'] == self.geo_name].iloc[0]['id']

        return self._resource

    @local_vs_gis
    def get(self, selector: str = None, selection_field: str = 'NAME',
            query_string: str = None, return_geometry: bool = True) -> pd.DataFrame:
        """ Get a DataFrame at an available geography_level level. Since frequently
        working within an area of interest defined by a higher level of
        geography_level, typically a CBSA or DMA, the ability to specify this
        area using input parameters is also included. This dramatically speeds
        up the process of creating the output.

        Args:
            selector: If a specific value can be identified using a string, even if
                just part of the field value, you can insert it here.
            selection_field: This is the field to be searched for the string values
                input into selector.
            query_string: If a more custom query is desired to filter the output, please
                use SQL here to specify the query. The normal query is "UPPER(NAME) LIKE
                UPPER('%<selector>%')". However, if a more specific query is needed, this
                can be used as the starting point to get more specific.
            return_geometry: Boolean indicating if geometry should be returned. While
                typically the case, there are instances where it is useful to not
                retrieve the geometries. This includes when getting a query right to only
                retrieve one area of interest. It also can be useful for only getting the
                block group id's within an area of interest.

        Returns:
            Spatially Enabled pd.DataFrame.
        """
        pass

    def _get_local(self, selector: (str, list) = None, selection_field: str = 'NAME',
                   query_string: str = None, return_geometry: bool = True) -> pd.DataFrame:

        # if not returning the geometry, use a search cursor - MUCH faster
        if not return_geometry:

            # create or use the input query parameters
            sql = self._get_sql_helper(selector, selection_field, query_string)

            # create an output dataframe of names filtered using the query
            col_nms = ['ID', 'NAME']
            val_lst = (r for r in arcpy.da.SearchCursor(self.resource, field_names=col_nms, where_clause=sql))
            out_df = pd.DataFrame(val_lst, columns=col_nms)

        # otherwise, go the SeDF route
        else:
            out_df = self._get_local_df(selector, selection_field, query_string, self._parent_data)

        return out_df

    def _get_gis(self, selector: (str, list) = None, selection_field: str = 'NAME',
                 query_string: str = None, return_geometry: bool = True) -> pd.DataFrame:
        """Web GIS implementation of 'get'."""

        # TODO: build ability to use selection_field and query_string parameters with Web GIS
        assert selection_field == 'NAME' and query_string is None, 'Neither the "selection_field" nor "query_string" ' \
                                                                   'parameters are implemented to be used with a Web ' \
                                                                   'GIS.'

        # start building the payload to send with the request
        params = {
            'f': 'json',
            'returnGeometry': return_geometry,
            'outsr': 4326
        }

        # format the selector - ensure everything strings or list of strings
        selector = self._format_get_selectors(selector)

        # if there is a parent standard geography
        if self._parent_data is not None:
            params['geographyIDs'] = self._parent_data.attrs['parent_geo']['id']
            params['geographyLayers'] = self._parent_data.attrs['parent_geo']['resource']
            params['returnsubgeographylayer'] = True
            params['subgeographylayer'] = self.resource
            params['subgeographyquery'] = selector

        # otherwise, if a fresh query (freshie!)
        else:
            params['geographyLayers'] = self.resource

            # determine what type of selector is being passed in, and set the correct request parameter
            if isinstance(selector, str):
                params['geographyQuery'] = selector
            elif all(v.isdigit() for v in selector):
                params['geographyIDs'] = selector

        # make the request using the gis connection to handle authentication
        r_json = self.source._con.post(
            f'{self.source.properties.helperServices("geoenrichment").url}/StandardGeographyQuery',
            params=params
        )

        # handle any errors returned
        if 'error' in r_json:
            err = r_json['error']
            raise Exception(f'Error in retrieving geographies from Business Analyst StandardGeographyQuery REST '
                            f'endpoint. Error Code {err["code"]}: {err["message"]}')

        else:
            # unpack the feature set from the response and convert it to a SeDF
            df = FeatureSet.from_dict(r_json['results'][0]['value']).sdf

            assert len(df.index), 'Your selection did not return any results. No records were returned.'

            # clean up the schema for consistency
            if return_geometry:
                df = df[['AreaID', 'AreaName', 'SHAPE']].copy()
                df.columns = ['ID', 'NAME', 'SHAPE']
                df.spatial.set_geometry('SHAPE')
            else:
                df = df[['AreaID', 'AreaName']].copy()
                df.columns = ['ID', 'NAME']

            # tack on the country and geographic level name for potential use later
            df.attrs['_cntry'] = self._cntry
            df.attrs['geo_name'] = self.geo_name
            df.attrs['parent_geo'] = {'resource': self.resource, 'id': list(df['ID'])}

        return df

    @local_vs_gis
    def within(self, selecting_geography: (pd.DataFrame, Geometry, list)) -> pd.DataFrame:
        """
        Get a input_dataframe at an available geography_level level falling within
        a defined selecting geography.

        Args:
            selecting_geography: Either a Spatially Enabled DataFrame, arcgis.Geometry object instance, or list of
                arcgis.Geometry objects delineating an area of interest to use for selecting geography_levels for
                analysis.

        Returns: pd.DataFrame as Geography object instance with the requested geography_levels.
        """
        pass

    def _within_local(self, selecting_geography: (pd.DataFrame, Geometry, list)) -> pd.DataFrame:
        """Local implementation of within."""
        return self._get_local_df(selecting_geography=selecting_geography)

    def _get_sql_helper(self, selector: (str, list) = None, selection_field: str = 'NAME',
                        query_string: str = None):
        """Helper to handle creation of sql queries for get functions."""
        if query_string:
            sql = query_string
        elif selection_field and isinstance(selector, list):
            sql_lst = [f"UPPER({selection_field}) LIKE UPPER('%{sel}%')" for sel in selector]
            sql = ' OR '.join(sql_lst)
        elif selection_field and isinstance(selector, str):
            sql = f"UPPER({selection_field}) LIKE UPPER('%{selector}%')"
        else:
            sql = None

        return sql

    def _get_local_df(self, selector: (str, list) = None, selection_field: str = 'NAME',
                      query_string: str = None,
                      selecting_geography: (pd.DataFrame, pd.Series, Geometry, list) = None) -> pd.DataFrame:
        """Single function handling business logic for both _get_local and _within_local."""
        # set up the where clause based on input enabling overriding using a custom query if desired
        sql = self._get_sql_helper(selector, selection_field, query_string)

        # get the relevant geography_level row from the data
        row = self._cntry.geography_levels[self._cntry.geography_levels['geo_name'] == self.geo_name].iloc[0]

        # get the id and geographic_level fields along with the path to the data from the row
        fld_lst = [row['col_id'], row['col_name']]
        pth = row['feature_class_path']

        # use the query string, if provided, to create and return a layer with the output fields
        if sql is None:
            lyr = arcpy.management.MakeFeatureLayer(pth)[0]
        else:
            lyr = arcpy.management.MakeFeatureLayer(pth, where_clause=sql)[0]

        # if there is selection data, convert to a layer and use this layer to select features from the above layer.
        if selecting_geography is not None:

            # convert all the selecting geography_levels to a list of ArcPy Geometries
            arcpy_geom_lst = geography_iterable_to_arcpy_geometry_list(selecting_geography, 'polygon')

            # create an feature class in memory
            tmp_fc = arcpy.management.CopyFeatures(arcpy_geom_lst, 'memory/tmp_poly')[0]

            # create a layer using the temporary feature class
            sel_lyr = arcpy.management.MakeFeatureLayer(tmp_fc)[0]

            # select local features using the temporary selection layer
            arcpy.management.SelectLayerByLocation(in_layer=lyr, overlap_type='HAVE_THEIR_CENTER_IN',
                                                   select_features=sel_lyr)

            # clean up arcpy litter
            for arcpy_resource in [tmp_fc, sel_lyr]:
                arcpy.management.Delete(arcpy_resource)

        # ensure something was actually selected
        assert int(arcpy.management.GetCount(lyr)[0]), 'It appears there are not any features in your selection to ' \
                                                       '"get".'

        # create a spatially enabled dataframe from the data in WGS84
        out_data = GeoAccessor.from_featureclass(lyr, fields=fld_lst).mdl.project(4326)

        # tack on the country and geographic level name for potential use later
        out_data.attrs['_cntry'] = self._cntry
        out_data.attrs['geo_name'] = self.geo_name

        return out_data
