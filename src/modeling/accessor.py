"""Provide modeling accessor object namespace and methods."""
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Union

from arcgis.features.geo._internals import register_dataframe_accessor
from arcgis.gis import GIS
from arcgis.geometry import SpatialReference

from .country import Country, GeographyLevel


@register_dataframe_accessor('mdl')
class Modeling:
    """
    Modeling is a Pandas DataFrame accessor, a standalone namespace for
    accessing geographic modeling functionality. If the DataFrame was created using
    a Country object, then the Modeling (``modeling``) namespace will automatically
    be available. However, if you want to use this functionality, and have not created
    the DataFrame using the Country object, you must import arcgis.modeling.Modeling
    to have this functionality available.
    """

    def __init__(self, obj):
        self._data = obj
        self._index = obj.index

        # save the country if it is passed from the invoking parent
        if '_cntry' in obj.attrs.keys():
            self._cntry = obj.attrs['_cntry']
        elif hasattr(obj, '_cntry'):
            self._cntry = obj._cntry
        else:
            self._cntry = None

        # if geo_name is a property of the dataframe, is the output of a chained function, and we can add capability
        if 'geo_name' in obj.attrs.keys():

            # get the geographic level index
            self._geo_idx = self._cntry.levels[self._cntry.levels['geo_name'] == self._data.attrs['geo_name']].index[0]

            # add all the geographic levels below the current geographic level as properties
            for idx in self._cntry.levels.index:
                if idx < self._geo_idx:
                    geo_name = self._cntry.levels.iloc[idx]['geo_name']
                    setattr(self, geo_name, GeographyLevel(geo_name, self._cntry, obj))

    def level(self, geographic_level: int) -> GeographyLevel:
        """
        Retrieve a Spatially Enabled DataFrame of geometries corresponding
        to the index returned by the Country.geography_levels property. This is
        most useful when retrieving the lowest, most granular, level of
        geography within a country.

        Args:
            geographic_level:
                Integer referencing the index of the geographic level desired.

        Returns:
            GeographyLevel object instance

        .. code-block:: python

            from dm import Country

            # create an instance of the country object
            cntry = Country('USA')

            # the get function returns a dataframe with the 'dm' property
            metro_df = cntry.cbsas('seattle')

            # level returns a CountryLevel object enabling getting all geography_levels
            # falling within the parent dataframe
            lvl_df = metro_df.dm.level(0).get()

        """
        assert self._cntry is not None, "The 'dm.level' method requires the parent dataframe be created by the" \
                                        "Country object."

        assert geographic_level <= self._geo_idx, 'The index for the sub-geography level must be less than the ' \
                                                  f'parent. You provided an index of {geographic_level}, ' \
                                                  f'which is greater than the parent index of {self._geo_idx}. '

        # get the name of the geographic level corresponding to the provided index
        geo_nm = self._cntry.geography_levels.iloc[geographic_level]['geo_name']

        # create a geographic level object
        geo_lvl = GeographyLevel(geo_nm, self._cntry, self._data)

        return geo_lvl

    def enrich(self, enrich_variables: Union[list, np.array, pd.Series, pd.DataFrame] = None,
               data_collections: Union[str, list, np.array, pd.Series] = None, country: Country = None) -> pd.DataFrame:
        """
        Enrich the DataFrame using the provided enrich variable list or data
        collections list. Either a variable list or list of data
        collections can be provided, but not both.

        Args:
            enrich_variables:
                List of data variables for enrichment. This can optionally
                be a filtered subset of the dataframe property of an instance
                of the Country object.
            data_collections:
                List of data collections for enrichment.
            country: Optional
                Country object instance. This must be included if the parent
                dataframe was not created using this package's standard
                geography methods, or if the enrichment variables are not
                defined by passing in an enrich variables dataframe created
                using this package's introspection methods.

        Returns:
            pd.DataFrame with enriched data.

        .. code-block:: python

            from pathlib import Path

            from arcgis import GeoAccessor
            from dm import Country, DemographicModeling
            import pandas as pd

            # get a path to the trade area data
            prj_pth = Path(__file__).parent
            gdb_pth = dir_data/'data.gdb'
            fc_pth = gdb/'trade_areas'

            # load the trade areas into a Spatially Enabled DataFrame
            ta_df = pd.DataFrame.spatial.from_featureclass(fc_pth)

            # create a country object instance
            usa = Country('USA', source='local')

            # get all the available enrichment variables
            e_vars = usa.enrich_variables

            # filter to just the current year key variables
            key_vars = e_vars[(e_vars.data_collection.str.startswith('Key')) &
                              (e_vars.name.str.endswith('CY'))]

            # enrich the Spatially Enabled DataFrame
            tae_df = ta_df.dm.enrich(key_vars)

        """
        # prioirtize the country parameter
        if country is not None:
            cntry = country

        # next, if the enrich variables has the country defined
        elif '_cntry' in enrich_variables.attrs.keys():
            cntry = enrich_variables.attrs['_cntry']

        # now, see if the parent dataframe has a country property
        elif hasattr(self, '_cntry'):
            cntry = self._cntry

        # otherwise, we don't know what to do
        else:
            cntry = None

        assert isinstance(cntry, Country), "The 'modeling.enrich' method requires the parent dataframe be created by " \
                                           "the Country object, the enrich variables to be provided as a dataframe " \
                                           "retrieved from a Country object, or a valid Country object must be " \
                                           "explicitly provided as input into the country parameter."

        # get the data from the GeoAccessor _data property
        data = self._data

        # invoke the enrich method from the country
        out_df = cntry.enrich(data, enrich_variables)

        return out_df

    def project(self, output_spatial_reference: Union[SpatialReference, int] = 4326):
        """
        Project to a new spatial reference, applying an applicable transformation if necessary.

        Args:
            output_spatial_reference:
                Optional - The output spatial reference. Default is 4326 (WGS84).

        Returns:
            Spatially Enabled DataFrame projected to the new spatial reference.
        """
        # import needed resources
        from .spatial import project_as

        # perform the projection
        return project_as(self._data, output_spatial_reference)

    def get_nearest(self, destination_dataframe: pd.DataFrame, source: Union[str, Path, Country, GIS] = None,
                    single_row_per_origin: bool = True, origin_id_column: str = 'LOCNUM',
                    destination_id_column: str = 'LOCNUM', destination_count: int = 4, near_prefix: str = None,
                    destination_columns_to_keep: Union[str, list] = None) -> pd.DataFrame:
        """
        Create a closest destination dataframe using a destination Spatially Enabled
        Dataframe relative to the parent Spatially enabled DataFrame, but keep each
        origin and destination still in a discrete row instead of collapsing to a
        single row per origin. The main reason to use this is if needing the geometry
        for visualization.

        Args:
            destination_dataframe:
                Destination points in one of the supported input formats.
            source:
                Optional - Either the path to the network dataset, the Country object
                associated with the Business Analyst source being used, or a GIS object
                instance. If invoked from a dataframe created for a country's standard
                geography levels using the dm accessor, get_nearest will use the parent
                country properties to ascertain how to perform the networks solve.
            single_row_per_origin:
                Optional - Whether or not to pivot the results to return
                only one row for each origin location. Default is True.
            origin_id_column:
                Optional - Column in the origin points Spatially Enabled Dataframe
                uniquely identifying each feature. Default is 'LOCNUM'.
            destination_id_column:
                Column in the destination points Spatially Enabled Dataframe
                uniquely identifying each feature
            destination_count:
                Integer number of destinations to search for from every origin
                point.
            near_prefix:
                String prefix to prepend onto near column names in the output.
            destination_columns_to_keep:
                List of columns to keep in the output. Commonly, if
                businesses, this includes the column with the business names.

        Returns:
            Spatially Enabled Dataframe with a row for each origin id, and metrics for
            each nth destinations.
        """
        # retrieve resources needed
        from .proximity import get_nearest

        # if the source is provided,
        source = self._cntry if source is None else source

        # solve get nearest
        near_df = get_nearest(self._data, destination_dataframe, source, single_row_per_origin, origin_id_column,
                              destination_id_column, destination_count, near_prefix, destination_columns_to_keep)

        # if the source is a country, tack it on for any follow-on analysis
        if isinstance(source, Country):
            setattr(near_df, '_cntry', source)

        return near_df
