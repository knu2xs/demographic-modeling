import importlib.util

from arcgis.env import active_gis
from arcgis.geometry import find_transformation, SpatialReference
from arcgis.gis import GIS
import pandas as pd

arcpy_avail = True if importlib.util.find_spec("arcpy") else False

if arcpy_avail:
    import arcpy


def reproject(input_dataframe: pd.DataFrame, output_spatial_reference: [int, SpatialReference] = 4326,
              input_spatial_reference: [int, SpatialReference] = None,
              transformation_name: str = None) -> pd.DataFrame:
    """
    Project input Spatially Enabled Dataframe to a desired output spatial reference, applying a
        transformation if needed due to the geographic coordinate system changing.
    Args:
        input_dataframe: Valid Spatially Enabled DataFrame
        output_spatial_reference: Optional - Desired output Spatial Reference. Default is
            4326 (WGS84).
        input_spatial_reference: Optional - Only necessary if the Spatial Reference is not
            properly defined for the input data geometry.
        transformation_name: Optional - Transformation name to be used, if needed, to
            convert between spatial references. If not explicitly provided, this will be
            inferred based on the spatial reference of the input data and desired output
            spatial reference.
    Returns: Spatially Enabled DataFrame in the desired output spatial reference.
    """
    # ensure the geometry is set
    geom_col_lst = [c for c in input_dataframe.columns if input_dataframe[c].dtype.name.lower() == 'geometry']
    assert len(geom_col_lst) > 0, 'The DataFrame does not appear to have a geometry column defined. This can be ' \
                                  'accomplished using the "df.spatial.set_geometry" method.'

    # save the geometry column to a variable
    geom_col = geom_col_lst[0]

    # ensure the input spatially enabled dataframe validates
    assert input_dataframe.spatial.validate(), 'The DataFrame does not appear to be valid.'

    # if a spatial reference is set for the dataframe, just use it
    if input_dataframe.spatial.sr is not None:
        in_sr = input_dataframe.spatial.sr

    # if a spatial reference is explicitly provided, but the data does not have one set, use the one provided
    elif input_spatial_reference is not None:

        # check the input
        assert isinstance(input_spatial_reference, int) or isinstance(input_spatial_reference, SpatialReference), \
            f'input_spatial_reference must be either an int referencing a wkid or a SpatialReference object, ' \
            f'not {type(input_spatial_reference)}.'

        if isinstance(input_spatial_reference, int):
            in_sr = SpatialReference(input_spatial_reference)
        else:
            in_sr = input_spatial_reference

    # if the spatial reference is not set, common for data coming from geojson, check if values are in lat/lon
    # range, and if so, go with WGS84, as this is likely the case if in this range
    else:

        # get the bounding values for the data
        x_min, y_min, x_max, y_max = input_dataframe.spatial.full_extent

        # check the range of the values, if in longitude and latitude range
        wgs_range = True if (x_min > -181 and y_min > -91 and x_max < 181 and y_max < 91) else False
        assert wgs_range, 'Input data for projection data must have a spatial reference, or one must be provided.'

        # if the values are in range, run with it
        in_sr = SpatialReference(4326)

    # ensure the output spatial reference is a SpatialReference object instance
    if isinstance(output_spatial_reference, SpatialReference):
        out_sr = output_spatial_reference
    else:
        out_sr = SpatialReference(output_spatial_reference)

    # copy the input spatial dataframe since the project function changes the dataframe in place
    out_df = input_dataframe.copy()
    out_df.spatial.set_geometry(geom_col)

    # if arcpy is available, use it to find the transformation
    if arcpy_avail and transformation_name is None:

        # get any necessary transformations using arcpy, which returns only a list of transformation names
        trns_lst = arcpy.ListTransformations(in_sr.as_arcpy, out_sr.as_arcpy)

    # otherwise we will have to use the geometry rest endpoint to find transformations
    elif transformation_name is None:

        # explicitly ensure find_transformations has a gis instance
        gis = active_gis if active_gis else GIS()

        # get any transformations, if needed due to changing geographic spatial reference, as a list of dicts
        trns_lst = find_transformation(in_sr, out_sr, gis=gis)['transformations']

    # apply across the geometries using apply since it recognizes the transformation correctly if transformation
    # is necessary and also tries arcpy first, and if not available, rolls back to rest resources elegantly
    if len(trns_lst) or transformation_name is not None:
        trns = transformation_name if transformation_name is not None else trns_lst[0]
        out_df[geom_col] = out_df[geom_col].apply(lambda geom: geom.project_as(out_sr, trns))

    # otherwise, do the same thing using the apply method since the geoaccessor project method is not working reliably
    # and only if necessary if the spatial reference is being changed
    elif in_sr.wkid != out_sr.wkid:
        out_df[geom_col] = out_df[geom_col].apply(lambda geom: geom.project_as(out_sr))

    # ensure the dataframe recognizes the new spatial reference
    if not out_df.spatial.validate():
        out_df.spatial.set_geometry(geom_col)

    return out_df
