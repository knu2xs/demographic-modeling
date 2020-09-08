from functools import wraps
import os.path

import arcgis.gis
import arcgis.features
from arcgis.geometry import Geometry
import pandas as pd

from . import util
from ._xml_interrogation import get_heirarchial_geography_dataframe
from ._registry import get_ba_demographic_gdb_path
from ._modify_geoaccessor import GeoAccessorIO as GeoAccessor

if util.arcpy_avail:
    import arcpy


def local_vs_gis(fn):
    # get the method geographic_level - this will be used to redirect the function call
    method_name = fn.__name__

    @wraps(fn)
    def wrapped(self, *args, **kwargs):

        # if performing analysis locally, try to access the function locally, but if not implemented, catch the error
        if self.source == 'local':
            try:
                fn_to_call = getattr(self, f'_{method_name}_local')
            except AttributeError:
                raise AttributeError(f"'{method_name}' not available using 'local' as the source.")

        # now, if performing analysis using a Web GIS, then access the function referencing remote resources
        elif isinstance(self.source, arcgis.gis.GIS):
            try:
                fn_to_call = getattr(self, f'_{method_name}_gis')
            except AttributeError:
                raise AttributeError(f"'{method_name}' not available using a Web GIS as the source.")

        # if another source, we don't plan on doing that any time soon
        else:
            raise AttributeError(f"'{self.source}' is not a recognized demographic modeling source.")

        return fn_to_call(*args, **kwargs)

    return wrapped


class Country:

    def __init__(self, name: str, source: [str, arcgis.gis.GIS] = None):
        self.name = name
        self.source = util.set_source(source)
        self._geographies = None

        # add on all the geographic resolution levels as properties
        for nm in self.geographies.name:
            setattr(self, nm, GeographyLevel(nm, self))

    def __repr__(self):
        return self.name

    @property
    def geographies(self):
        """DataFrame of available geographies."""

        if self._geographies is None and self.source is 'local':
            self._geographies = get_heirarchial_geography_dataframe(self.name)

        elif self._geographies is None and isinstance(self.source, arcgis.gis.GIS):
            raise Exception('Using a GIS instance is not yet implemented.')

        return self._geographies

    @local_vs_gis
    def level(self, geography_index: [str, int]) -> pd.DataFrame:
        """
        Get a GeographyLevel at an available geography_level level in the country.

        Args:
            geography_index: Either the geographic_level name or the index of the geography_level level. This can be
                discovered using the Country.geographies method.

        Returns: pd.DataFrame as Geography object instance with the requested geographies.
        """
        pass

    def _level_local(self, geography_index: [str, int]) -> pd.DataFrame:
        """Local implementation of level."""
        # get the name of the geography
        if isinstance(geography_index, int):
            nm = self.geographies.iloc[geography_index]['name']
        elif isinstance(geography_index, str):
            nm = geography_index
        else:
            raise Exception(f'geography_index must be either a string or integer, not {type(geography_index)}')

        # create a GeographyLevel object instance
        return GeographyLevel(nm, self)


@pd.api.extensions.register_dataframe_accessor('geo_level')
class GeographyLevel:

    def __init__(self, geographic_level: [str, int], country: Country, parent_data: pd.DataFrame = None):
        self._cntry = country
        self.source = country.source
        self.name = self._standardize_geographic_level_input(geographic_level)
        self._resource = None

    def __repr__(self):
        return f'GeographyLevel: {self.name}'

    def _standardize_geographic_level_input(self, geo_in:[str, int])->str:
        """Helper function to check and standardize named input."""

        geo_df = self._cntry.geographies

        if isinstance(geo_in, str):
            if geo_in not in geo_df.name.values:
                names = ', '.join(geo_df.names.values)
                raise Exception(
                    f'Your selector, "{geo_in}," is not an available selector. Please choose from {names}.')
            geo_lvl_name = geo_in

        elif isinstance(geo_in, int) or isinstance(geo_in, float):
            if geo_in > len(geo_df.index):
                raise Exception(
                    f'Your selector, "{geo_in}", is beyond the maximum range of available geographies.')
            geo_lvl_name = geo_df.iloc[geo_in]['name']

        else:
            raise Exception('The geographic selector must be a string or integer.')

        return geo_lvl_name

    @property
    def resource(self):
        """The resource, either a layer or Feature Layer, for accessing the data for the geographic layer."""
        if self._resource is None and self._cntry.source is 'local':
            self._resource = self._cntry.geographies[self._cntry.geographies['name'] == self.name].iloc[0]['feature_class_path']

        elif self._resource is None and isinstance(self._cntry.source, arcgis.gis.GIS):
            raise Exception('Using a GIS instance not yet implemented.')

        return self._resource


    @local_vs_gis
    def get(self, geography: [str, int], selector: str = None, selection_field: str = 'NAME',
            query_string: str = None) -> pd.DataFrame:
        """
        Get a df at an available geography_level level. Since frequently working within an area of interest defined
        by a higher level of geography_level, typically a CBSA or DMA, the ability to specify this area using input
        parameters is also included. This dramatically speeds up the process of creating the output.

        Args:
            geography: Either the geographic_level or the index of the geography_level level. This can be discovered using the
                Country.geographies method.
            selector: If a specific value can be identified using a string, even if just part of the field value,
                you can insert it here.
            selection_field: This is the field to be searched for the string values input into selector.
            query_string: If a more custom query is desired to filter the output, please use SQL here to specify the
                query.

        Returns: pd.DataFrame as Geography object instance with the requested geographies.
        """
        pass

    def _get_local(self, selector: [str, list] = None, selection_field: str = 'NAME',
                   query_string: str = None) -> pd.DataFrame:

        # set up the where clause based on input enabling overriding using a custom query if desired
        if query_string:
            sql = query_string
        elif selection_field and isinstance(selector, list):
            sql_lst = [f"UPPER({selection_field}) LIKE UPPER('%{sel}%')" for sel in selector]
            sql = ' OR '.join(sql_lst)
        elif selection_field and isinstance(selector, str):
            sql = f"UPPER({selection_field}) LIKE UPPER('%{selector}%')"
        else:
            sql = None

        # simplify by setting local variable to the geography_index levels dataframe
        df_geo = self._cntry.geographies

        # get the relevant geography_level row from the data
        row = df_geo[df_geo['name'] == self.name].iloc[0]

        # get the id and geographic_level fields along with the path to the data from the row
        fld_lst = [row['col_id'], row['col_name']]
        pth = row['feature_class_path']

        # use the query string, if provided, to create and return a layer with the output fields
        if sql is None:
            lyr = arcpy.management.MakeFeatureLayer(pth)[0]
        else:
            lyr = arcpy.management.MakeFeatureLayer(pth, where_clause=sql)[0]

        out_df = GeoAccessor.from_featureclass(lyr, fields=fld_lst)

        return out_df
