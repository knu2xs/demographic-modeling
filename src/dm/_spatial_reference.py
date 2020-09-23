import importlib.util

from arcgis.gis import GIS
from arcgis.env import active_gis
from arcgis.geometry import find_transformation, SpatialReference
import pandas as pd

from . import util

if util.arcpy_avail:
    import arcpy


def reproject(input_dataframe: pd.DataFrame, output_spatial_reference: [int, SpatialReference] = 4326) -> pd.DataFrame:
    """
    Reproject input Spatially Enabled Dataframe to output spatial reference and applying a transformation if needed
        due to a changed geographic coordinate system.
    Args:
        input_dataframe: Valid Spatially Enabled DataFrame
        output_spatial_reference: Optional - Desired output Spatial Reference. Default is 4326 (WGS84).
    Returns: Spatially Enabled DataFrame in the desired output spatial reference.
    """
    # ensure the geometry is set
    geom_col_lst = [c for c in input_dataframe.columns if input_dataframe[c].dtype.name.lower() == 'geometry']
    assert len(geom_col_lst) > 0, 'The input DataFrame for reprojection does not appear to have a geometry column set.'

    # save the geometry column to a variable
    geom_col = geom_col_lst[0]

    # ensure the input spatially enabled dataframe validates
    assert input_dataframe.spatial.validate(), 'The input DataFrame for reprojection does not appear to be valid.'

    # get the input spatial reference for the dataframe
    in_sr = input_dataframe.spatial.sr

    # ensure the output spatial reference is a SpatialReference object instance
    if isinstance(output_spatial_reference, SpatialReference):
        out_sr = output_spatial_reference
    else:
        out_sr = SpatialReference(output_spatial_reference)

    # copy the input spatial dataframe since the project function changes the dataframe in place
    out_df = input_dataframe.copy()
    out_df.spatial.set_geometry(geom_col)

    # if arcpy is available, use it to find the transformation
    if util.arcpy_avail:

        # get any necessary transformations using arcpy, which returns only a list of transformation names
        trns_lst = arcpy.ListTransformations(in_sr.as_arcpy, out_sr.as_arcpy)

    # otherwise we will have to use the geometry rest endpoint to find transformations
    else:

        # explicitly ensure find_transformations has a gis instance
        gis = active_gis if active_gis else GIS()

        # get any transformations, if needed due to changing geographic spatial reference, as a list of dicts
        trns_lst = find_transformation(in_sr, out_sr, gis=gis)['transformations']

    # apply across the geometries using apply since it recognizes the transformation correctly if transformation
    # is necessary and also tries arcpy first, and if not available, rolls back to rest resources elegantly
    if len(trns_lst):
        out_df[geom_col] = out_df[geom_col].apply(lambda geom: geom.project_as(out_sr, trns_lst[0]))

    # otherwise, do the same thing using the apply method since the geoaccessor project method is not working reliably
    else:
        out_df[geom_col] = out_df[geom_col].apply(lambda geom: geom.project_as(out_sr))

    # tell the dataframe to recognize the new spatial reference
    out_df.spatial.set_geometry(geom_col_lst[0])

    return out_df
