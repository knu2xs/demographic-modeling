from functools import wraps
import os.path

import arcgis.gis
import arcgis.features
from arcgis.geometry import Geometry
import pandas as pd

from . import util
from ._xml_interrogation import get_heirarchial_geography_dataframe
from ._registry import get_ba_demographic_gdb_path
from ._modify_geoaccessor import GeoAccessorIO

if util.arcpy_avail:
    import arcpy


def local_vs_gis(fn):
    # get the method name - this will be used to redirect the function call
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


class Country(pd.DataFrame):

    def __init__(self, name: str, source: [str, arcgis.gis.GIS] = None, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.name = name
        self.source = util._set_source(source)

    @property
    def geographies(self) -> pd.DataFrame:
        """DataFrame of available geographies."""
        return get_heirarchial_geography_dataframe(self.name)

    @local_vs_gis
    def get_geography(self, geography: [str, int], selector: str = None,
                      selection_field: str = 'NAME', query_string: str = None, aoi_geography: [str, int] = None,
                      aoi_selector: str = None, aoi_selection_field: str = 'NAME',
                      aoi_query_string: str = None) -> pd.DataFrame:
        """
        Get a df at an available geography level. Since frequently working within an area of interest defined
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

    def _get_geography_local(self, geography: [str, int], selector: str = None,
                            selection_field: str = 'NAME', query_string: str = None, aoi_geography: [str, int] = None,
                            aoi_selector: str = None, aoi_selection_field: str = 'NAME',
                            aoi_query_string: str = None) -> pd.DataFrame:
        """Local analysis implementation of get_geography function."""
        # preprocess the inputs
        geo, aoi_geo, sql, aoi_sql = util.get_geography_preprocessing(self.geographies, geography, selector,
                                                                      selection_field, query_string, aoi_geography,
                                                                      aoi_selector, aoi_selection_field,
                                                                      aoi_query_string)
        # get a single instance of the geographies to avoid recreating multiple times
        df_geo = self.geographies

        # singular geographic retrieval
        if geo and aoi_geo is None:
            lyr, fld_lst = util.get_lyr_flds_from_geo_df(df_geo, geo, sql)
            out_df = self.spatial.from_featureclass(lyr, fields=fld_lst)

        # if using a spatial filter
        else:

            # get the relevant layer information for both the
            lyr, fld_lst = util.get_lyr_flds_from_geo_df(df_geo, geo, sql)
            aoi_lyr = util.get_lyr_flds_from_geo_df(df_geo, aoi_geo, aoi_sql)[0]

            # select by location to reduce output overhead
            sel_lyr = arcpy.management.SelectLayerByLocation(
                in_layer=lyr,
                overlap_type='HAVE_THEIR_CENTER_IN',
                select_features=aoi_lyr
            )[0]

            # convert to spatially enabled df and return results
            out_df = self.spatial.from_featureclass(sel_lyr, fields=fld_lst)

        # to ensure all properties are retained, set the data and return a copy of this instance
        self.spatial._data = out_df
        return self.copy()

    def within(self, selecting_area: [pd.DataFrame, arcgis.geometry.Geometry],
               index_features: bool = False) -> pd.DataFrame:
        """
        Get only features contained within the selecting area.

        Args:
            selecting_area: Either a Spatially Enabled Pandas DataFame or a Geometry
                object describing the area of interest for extracting features.
            index_features: Optional: Boolean describing whether or not to take the
                time to create spatial indices before extracting features.
                Default - False

        Returns: Spatially Enabled Pandas DataFrame
        """
        if isinstance(selecting_area, arcgis.geometry.Geometry):
            selecting_df = pd.DataFrame([selecting_area], columns=['SHAPE'])
            selecting_df.spatial.set_geometry('SHAPE')

        elif isinstance(selecting_area, pd.DataFrame):
            selecting_df = selecting_area

        else:
            raise Exception('Selecting area must be either a Spatially Enabled DataFrame or a Geometry object, not'
                            f' {type(selecting_area)}.')

        if selecting_df.spatial.validate() is False:
            raise Exception('The selecting areas DataFrame provided does not appear to be a valid spatially'
                            ' Enabled DataFrame.')

        # convert the geometries to points
        pts = self.SHAPE.apply(lambda geom: Geometry(
            {'x': geom.centroid[0], 'y': geom.centroid[1], 'spatialReference': geom.spatial_reference}))
        pt_df = pts.to_frame().spatial.set_index('SHAPE')

        # index the features being selected if desired
        if index_features:
            pt_df.spatial.sindex(stype='rtree')

        # select records falling within the area of interest
        pt_sel_df = selecting_df.spatial.select(pt_df)

        # get the indices of the selected records, and use them to select the records in the original dataframe
        return self.iloc[pt_sel_df.index]
