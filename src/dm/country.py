from functools import wraps
import os.path

import arcgis.gis
import arcgis.features
from arcgis.features import GeoAccessor
from arcgis.geometry import Geometry
from ba_tools import data
import pandas as pd

from . import util
from ._registry_nav import get_ba_demographic_gdb_path

if util.arcpy_avail:
    import arcpy


class GeographyDataFrame(pd.DataFrame):
    """Class to enable function chaining from the get_geography method below."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


def dataframe_wrapper(fn):
    @wraps
    def wrapper(*args, **kwds):
        return fn(*args, **kwds)

    return wrapper


class Country(GeographyDataFrame):

    def __init__(self, name: str, source: [str, arcgis.gis.GIS], **kwargs):

        super().__init__(**kwargs)

        self.name = name
        self.source = util._set_source(source)

    @property
    def geographies(self) -> pd.DataFrame:
        """DataFrame of available geographies."""
        # TODO: How to populate by reading local resources?
        geo_lst = [
            ('block_group', 'Block Group', 'US Census designated block groups', None),
            ('tract', 'Tract', 'US Census designated tracts', None),
            ('place', 'Place', 'US Census designated places, typically municipal boundaries', 3),
            ('csd', 'County Sub-Divisions', 'Sub divisions within counties.', None),
            ('county', 'County', 'County boundaries', int(2)),
            ('cbsa', 'Core-Based Statistical Area', 'Metro and mircopolitan areas.', None),
            ('dma', 'Designated Market Area', 'Market areas defined by media reach comprised of counties.', None),
            ('state', 'State', 'US states', int(1)),
            ('country', 'Entire Country', 'The United States', int(0))
        ]
        df = pd.DataFrame(geo_lst, columns=['name', 'alias', 'description', 'admin_level'], )
        return df

    def get_geography(self, geography: [str, int], selector: str = None, selection_field: str = 'NAME',
                      query_string=None) -> GeographyDataFrame:
        """
        Get a dataframe at an available geography level.
        :param geography: Either the name or the index of the geography level. This can be discovered using the
            Country.geographies method.
        :param selector: If a specific value can be identified using a string, even if just part of the field value,
            you can insert it here.
        :param selection_field: This is the field to be searched for the string values input into selector.
        :param query_string: If a more custom query is desired to filter the output, please use SQL here to specify the
            query.
        :return: GeographyDataFrame with the requested geographies.
        """

        if self.source is 'local':
            gdb = get_ba_demographic_gdb_path()

        # set up the where clause based on input
        if query_string:
            where_clause = query_string

        elif selection_field and selector:
            where_clause = f"{selection_field} LIKE '%{selector}%'"

        else:
            where_clause = None

        def _read_geo(feature_class_name, fld_lst=['ID', 'NAME']):
            """Helper function to speed things up."""
            if 'SHAPE' in fld_lst:
                fld_lst.remove('SHAPE')

            fc_pth = os.path.join(gdb, feature_class_name)

            # df = pd.DataFrame(
            #     data=[list(r[:-1]) + [Geometry(r[-1].JSON)] for r in
            #           arcpy.da.SearchCursor(fc_pth, fld_lst + ['SHAPE@'])],
            #     columns=fld_lst + ['SHAPE']
            # )
            # df.spatial.set_geometry = 'SHAPE'
            df = GeoAccessor.from_featureclass(fc_pth, fields=fld_lst, where_clause=where_clause)

            return self.spatial.from_df(df, geometry_column='SHAPE')

        # if string, ensure is one of the geographc areas
        if isinstance(geography, str):

            if geography not in self.geographies.name.values:
                raise Exception(
                    f'Your selector, "{geography}," is not an available selector. To view a list of available'
                    f' selectors, please view the {self.name}.geographies property.')

            if geography == 'block_group':
                return _read_geo('BlockGroups_bg')

            elif geography == 'cbsa':
                return _read_geo('CBSAs_cb')

        # if integer, ensure in not beyond the range
        if isinstance(geography, (int, float)):

            if geography > len(self.geographies.index):
                raise Exception(f'Your selector, "{geography}", is beyond the maximum range of available geographies.')

            if geography == 0:
                return _read_geo('BlockGroups_bg')

            elif geography == 5:
                return _read_geo('CBSAs_cb')

    def within(self, selecting_area: [pd.DataFrame, arcgis.geometry.Geometry],
               index_features: bool = False) -> GeographyDataFrame:
        """
        Get only features contained within the selecting area.
        """
        if isinstance(selecting_area, pd.DataFrame):

            if selecting_area.spatial.validate() is False:
                raise Exception('The selecting areas dataframe provided does not appear to be a valid spatially'
                                ' enabled dataframe.')

            # copy the dataframe with everything except the geometry
            tmp_df = self[[col for col in self.columns if col != 'SHAPE']].copy()

            # convert the geometries to points
            tmp_df['SHAPE'] = self.SHAPE.apply(lambda geom: Geometry(
                {'x': geom.centroid[0], 'y': geom.centroid[1], 'spatialReference': geom.spatial_reference}))

            # select records falling within the area of interest
            return tmp_df.spatial.select(self)
