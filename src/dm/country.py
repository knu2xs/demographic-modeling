from functools import wraps
import os.path

import arcgis.gis
import arcgis.features
from arcgis.features import GeoAccessor
from arcgis.geometry import Geometry
import pandas as pd

from . import util
from ._xml_interrogation import get_heirarchial_geography_dataframe
from ._registry import get_ba_demographic_gdb_path

if util.arcpy_avail:
    import arcpy


def use_local_vs_gis(fn):
    # get the method name, used to redirect call
    method_name = fn.__name__

    @wraps(fn)
    def wrapped(*args, **kwargs):
        # This is a class method decorator, first arg is self
        self = args[0]

        # if performing analysis locally, try to access the function locally, but if not implemented, catch the error
        if self.source == 'local':
            try:
                f_to_call = getattr(self, method_name + "_local")
            except AttributeError:
                raise AttributeError(f"'{method_name}' not available using 'local' as the source.")

        elif isinstance(self.source, arcgis.gis.GIS):

            try:
                f_to_call = getattr(self, method_name + "_gis")
            except AttributeError:
                raise AttributeError(f"'{method_name}' not available using a Web GIS as the source.")
        else:
            raise AttributeError(f"Source '{self.source}' does not implement ")

        return f_to_call(*args, **kwargs)

    return wrapped


class Country(pd.DataFrame):

    def __init__(self, name: str, source: [str, arcgis.gis.GIS] = None, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.name = name
        self.source = util._set_source(source)

    @property
    def geographies(self) -> pd.DataFrame:
        """DataFrame of available geographies."""
        return get_heirarchial_geography_dataframe(self.name)

    def _get_geography_preprocessing(self, geography: [str, int], selector: str = None,
                                     selection_field: str = 'NAME', query_string: str = None,
                                     aoi_geography: [str, int] = None,
                                     aoi_selector: str = None, aoi_selection_field: str = 'NAME',
                                     aoi_query_string: str = None) -> tuple:
        """Helper function consolidating input parameters for later steps."""

        def _check_geo_name(nm):
            """Helper function to check if the geography name is valid."""
            if geography not in self.geographies.name.values:
                raise Exception(
                    f'Your selector, "{nm}," is not an available selector. To view a list of available'
                    f' selectors, please view the {self.name}.geographies property.')
            else:
                return True

        def _check_geo_index(idx):
            """Helper function to check if the index is in range."""
            if idx > len(self.geographies.index):
                raise Exception(f'Your selector, "{geography}", is beyond the maximum range of available geographies.')
            else:
                return True

        def _standardize_geo_input(geo_in):
            """Helper function to check and standardize inputs."""
            if isinstance(geo_in, str):
                _check_geo_name(geo_in)
                return geo_in
            elif isinstance(geo_in, int):
                _check_geo_index(geo_in)
                return self.geographies.iloc[geo_in]

        def _get_where_clause(selector, selection_field, query_string):
            """Helper function to consolidate where clauses."""
            # set up the where clause based on input
            if query_string:
                return query_string

            elif selection_field and selector:
                return f"{selection_field} LIKE '%{selector}%'"

            else:
                return None

        # standardize the geography input
        geo = _standardize_geo_input(geography)
        aoi_geo = _standardize_geo_input(aoi_geography)

        # consolidate selection
        where_clause = _get_where_clause(selector, selection_field, query_string)
        aoi_where_clause = _get_where_clause(aoi_selector, aoi_selection_field, aoi_query_string)

        return geo, aoi_geo, where_clause, aoi_where_clause

    @use_local_vs_gis
    def get_geography(self, geography: [str, int], selector: str = None,
                      selection_field: str = 'NAME', query_string: str = None, aoi_geography: [str, int] = None,
                      aoi_selector: str = None, aoi_selection_field: str = 'NAME',
                      aoi_query_string: str = None) -> pd.DataFrame:
        """
        Get a dataframe at an available geography level. Since frequently working within an area of interest defined
        by a higher level of geography, typically a CBSA or DMA, the ability to specify this area using input
        parameters is also included. This dramatically speeds up the process of creating the output.

        Args:
            geography: Either the name or the index of the geography level. This can be discovered using the
                Country.geographies method.
            selector: If a specific value can be identified using a string, even if just part of the field value,
                you can insert it here.
            selection_field: This is the field to be searched for the string values input into selector.
            query_string: If a more custom query is desired to filter the output, please use SQL here to specify the
                query.
            aoi_geography: Similar to the geography parameter above, the
            aoi_selector:
            aoi_selection_field:
            aoi_query_string:

        Returns: pd.DataFrame as Geography object instance with the requested geographies..
        """
        pass

    def get_geography_local(self, geography: [str, int], selector: str = None,
                            selection_field: str = 'NAME', query_string: str = None, aoi_geography: [str, int] = None,
                            aoi_selector: str = None, aoi_selection_field: str = 'NAME',
                            aoi_query_string: str = None) -> pd.DataFrame:

        # preprocess the inputs
        geo, aoi_geo, where_clause, aoi_where_clause = self._get_geography_preprocessing(geography, selector,
                                                                                         selection_field,
                                                                                         query_string, aoi_geography,
                                                                                         aoi_selector,
                                                                                         aoi_selection_field,
                                                                                         aoi_query_string)

        def _fc_to_lyr(geo, query_str):
            """Helper function to create a feature layer for working with."""
            df_geo = self.geographies
            row = df_geo[df_geo['name'] == geo].iloc[0]

            fld_lst = [row['col_id'], row['col_name']]
            pth = row['feature_class_path']

            if query_string:
                lyr = arcpy.management.MakeFeatureLayer(str(pth), where_clause=query_str)[0]
            else:
                lyr = arcpy.management.MakeFeatureLayer(str(pth))[0]

            return lyr, fld_lst

        # singular geographic retrieval
        if geography and not aoi_geography:
            lyr, fld_lst = _fc_to_lyr(geo, where_clause)
            return self.spatial.from_featureclass(lyr, fields=fld_lst)

        # if using a filter
        else:

            # select by location to reduce output overhead
            sel_lyr = arcpy.management.SelectLayerByLocation(
                in_layer=_fc_to_lyr(geo, where_clause),
                overlap_type='HAVE_THEIR_CENTER_IN',
                select_features=_fc_to_lyr(aoi_geo, aoi_where_clause)
            )[0]

            # convert to spatially enabled dataframe and return results
            return self.spatial.from_featureclass(sel_lyr, fields=out_fld_lst)

    def within(self, selecting_area: [pd.DataFrame, arcgis.geometry.Geometry],
               index_features: bool = False) -> pd.DataFrame:
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
