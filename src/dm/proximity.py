import math
import os
import tempfile
import uuid

from arcgis.features import GeoAccessor
from arcgis.geometry import Geometry
import arcpy
import pandas as pd

from . import util
from .util import local_vs_gis
from ._registry import get_ba_key_value

# location to store temp files if necessary
csv_file_prefix = 'temp_closest'
temp_file_root = os.path.join(tempfile.gettempdir(), csv_file_prefix)

# ensure previous runs do not interfere
arcpy.env.overwriteOutput = True


def prep_sdf_for_nearest(df, id_fld):
    """
    Given an input Spatially Enabled Dataframe, prepare it to work well with the nearest solver.
    :param df: Spatially Enabled Dataframe with really any geometry.
    :param id_fld: Field uniquely identifying each of location to be used for routing to nearest.
    :return: Spatially Enabled Dataframe of points with correct columns for routing to nearest.
    """
    # par down the input dataframe to just the columns needed
    df = df[[id_fld, 'SHAPE']].copy()

    # rename the columns to follow the schema needed for routing
    df.columns = ['ID', 'SHAPE']

    # otherwise, if the geometry is not points, we still need points, so just get the geometric centroids
    # TODO: Account for polygons NOT always being in WGS 84
    if df.spatial.geometry_type != ['point']:
        df['SHAPE'] = df['SHAPE'].apply(
            lambda geom: Geometry({'x': geom.centroid[0], 'y': geom.centroid[1], 'spatialReference': {'wkid': 4326}}))

    # add a second column for the ID as Name
    df['Name'] = df['ID']

    # ensure the geometry is correctly being recognized
    df.spatial.set_geometry('SHAPE')

    # set the order of the columns and return
    return df[['ID', 'Name', 'SHAPE']].copy()


def _get_max_near_dist_arcpy(origin_lyr):
    """Get the maximum geodesic distance between stores."""
    # create a location for temporary data
    temp_table = r'in_memory\near_table_{}'.format(uuid.uuid4().hex)

    # if only one location, cannot generate a near table, and default to 120 miles
    if int(arcpy.management.GetCount(origin_lyr)[0]) <= 1:
        max_near_dist = 120 * 1609.34

    else:
        # use arcpy to get a table of all distances between stores
        near_tbl = arcpy.analysis.GenerateNearTable(
            in_features=origin_lyr,
            near_features=origin_lyr,
            out_table=temp_table,
            method="GEODESIC"
        )[0]

        # get the maximum near distance, which will be in meters
        meters = max([row[0] for row in arcpy.da.SearchCursor(near_tbl, 'NEAR_DIST')])

        # remove the temporty table to ensure not stuff lying around and consuming RAM
        arcpy.management.Delete(temp_table)

        # get the maximum near distance (in meters)
        max_near_dist = meters * 0.00062137

    return max_near_dist


def _get_closest_network_local(origin_df, dest_df, dest_count, network_dataset, max_dist=None):
    """
    Perform network solve using local resources with assumption of standard input.
    Args:
        origin_df: Origin points Spatially Enabled Dataframe
        dest_df: Destination points Spatially Enabled Dataframe
        dest_count: Destination points Spatially Enabled Dataframe
        network_dataset: Path to ArcGIS Network dataset for performing routing.
        max_dist: Maximum nearest routing distance in miles.
    Returns: Spatially Enabled Dataframe of solved closest facility routes.
    """
    # get the mode of travel from the network dataset - rural so gravel roads are fair game
    nd_lyr = arcpy.nax.MakeNetworkDatasetLayer(network_dataset)[0]
    trvl_mode_dict = arcpy.nax.GetTravelModes(nd_lyr)
    trvl_mode = trvl_mode_dict['Rural Driving Time']

    # create the closest solver object instance
    # https://pro.arcgis.com/en/pro-app/arcpy/network-analyst/closestfacility.htm
    closest_solver = arcpy.nax.ClosestFacility(network_dataset)

    # set parameters for the closest solver
    closest_solver.travelMode = trvl_mode
    closest_solver.travelDirection = arcpy.nax.TravelDirection.ToFacility
    # TODO: How to set this to distance?
    closest_solver.timeUnits = arcpy.nax.TimeUnits.Minutes
    closest_solver.distanceUnits = arcpy.nax.DistanceUnits.Miles
    closest_solver.defaultTargetFacilityCount = dest_count
    closest_solver.routeShapeType = arcpy.nax.RouteShapeType.TrueShapeWithMeasures
    closest_solver.searchTolerance = 5000
    closest_solver.searchToleranceUnits = arcpy.nax.DistanceUnits.Meters

    # since maximum distance is optional, well, make it optional
    if max_dist is not None:
        closest_solver.defaultImpedanceCutoff = max_dist

    # load the origin and destination feature data frames into memory and load into the solver object instance
    # TODO: test if can use 'memory' workspace instead of scratch
    origin_fc = origin_df.spatial.to_featureclass(os.path.join(arcpy.env.scratchGDB, 'origin_tmp'))
    closest_solver.load(arcpy.nax.ClosestFacilityInputDataType.Incidents, origin_fc)

    dest_fc = dest_df.spatial.to_featureclass(os.path.join(arcpy.env.scratchGDB, 'dest_tmp'))
    closest_solver.load(arcpy.nax.ClosestFacilityInputDataType.Facilities, dest_fc)

    # run the solve, and get comfortable
    closest_result = closest_solver.solve()

    # export the results to a spatially enabled data frame, and do a little cleanup
    # TODO: test if can use 'memory/routes' instead - the more current method
    route_fc = 'in_memory/routes'
    closest_result.export(arcpy.nax.ClosestFacilityOutputDataType.Routes, route_fc)
    route_oid_col = arcpy.Describe(route_fc).OIDFieldName
    closest_df = GeoAccessor.from_featureclass(route_fc)
    arcpy.management.Delete(route_fc)
    if route_oid_col:
        closest_df.drop(columns=[route_oid_col], inplace=True)

    # get rid of the extra empty columns the local network solve adds
    closest_df.dropna(axis=1, how='all', inplace=True)

    # populate the origin and destination fields so the schema matches what online solve returns
    name_srs = closest_df.Name.str.split(' - ')
    closest_df['IncidentID'] = name_srs.apply(lambda val: val[0])
    closest_df['FacilityID'] = name_srs.apply(lambda val: val[1])

    return closest_df


def _reformat_closest_result_dataframe(closest_df):
    """
    Reformat the schema, dropping unneeded columns and renaming those kept to be more in line with this workflow.
    :param closest_df: Dataframe of the raw output routes from the find closest analysis.
    :return: Spatially Enabled Dataframe reformatted.
    """
    # create a list of columns containing proximity metrics
    proximity_src_cols = [col for col in closest_df.columns if col.startswith('Total_')]

    # if both miles and kilometers, drop miles, and keep kilometers
    miles_lst = [col for col in proximity_src_cols if 'miles' in col.lower()]
    kilometers_lst = [col for col in proximity_src_cols if 'kilometers' in col.lower()]
    if len(miles_lst) and len(kilometers_lst):
        proximity_src_cols = [col for col in proximity_src_cols if col != miles_lst[0]]

    # calculate side of street columns
    closest_df['proximity_side_street_right'] = (closest_df['FacilityCurbApproach'] == 1).astype('int64')
    closest_df['proximity_side_street_left'] = (closest_df['FacilityCurbApproach'] == 2).astype('int64')
    side_cols = ['proximity_side_street_left', 'proximity_side_street_right']

    # filter the dataframe to just the columns we need
    src_cols = ['IncidentID', 'FacilityRank', 'FacilityID'] + proximity_src_cols + side_cols + ['SHAPE']
    closest_df = closest_df[src_cols].copy()

    # replace total in proximity columns for naming convention
    closest_df.columns = [col.lower().replace('total', 'proximity') if col.startswith('Total_') else col
                          for col in closest_df.columns]

    # rename the columns for the naming convention
    rename_dict = {'IncidentID': 'origin_id', 'FacilityRank': 'destination_rank', 'FacilityID': 'destination_id'}
    closest_df = closest_df.rename(columns=rename_dict)

    return closest_df


def _explode_closest_rank_dataframe(closest_df, origin_id_col='origin_id', rank_col='destination_rank',
                                    dest_id_col='destination_id'):
    """
    Effectively explode out or pivot the data so there is only a single record for each origin.
    :param closest_df: Spatially Enabled Dataframe reformatted from the raw output of find nearest.
    :param origin_id_col: Column uniquely identifying each origin - default 'origin_id'
    :param rank_col: Column identifying the rank of each destination - default 'destination_rank'
    :param dest_id_col: Column uniquely identifying each destination - default 'destination_id'
    :return: Dataframe with a single row for each origin with multiple destination metrics for each.
    """
    # create a dataframe to start working with comprised of only the unique origins to start with
    origin_dest_df = pd.DataFrame(closest_df[origin_id_col].unique(), columns=[origin_id_col])

    # get a list of the proximity columns
    proximity_cols = [col for col in closest_df.columns if col.startswith('proximity_')]

    # iterate the closest destination ranking
    for rank_val in closest_df[rank_col].unique():

        # filter the dataframe to just the records with this destination ranking
        rank_df = closest_df[closest_df[rank_col] == rank_val]

        # create a temporary dataframe to begin building the columns onto
        df_temp = rank_df[origin_id_col].to_frame()

        # iterate the relevant columns
        for col in [dest_id_col] + proximity_cols:
            # create a new column name from the unique value and the original row name
            new_name = f'{col}_{rank_val:02d}'

            # filter the data in the column with the unique value
            df_temp[new_name] = rank_df[col].values

        # set the index to the origin id for joining
        df_temp.set_index(origin_id_col, inplace=True)

        # join the temporary dataframe to the master
        origin_dest_df = origin_dest_df.join(df_temp, on=origin_id_col)

    return origin_dest_df

@local_vs_gis
def get_nearest_solution(origins, destinations, origin_id_fld='LOCNUM', dest_id_fld='LOCNUM', destination_count=4):
    """
    Create a closest destination dataframe using origin and destination Spatially Enabled Dataframes, but keep
        each origin and destination still in a discrete row instead of collapsing to a single row per origin. The main
        reason to use this is if needing the geometry for visualization.
    Args:
        origins: Origins for networks solves.
        origin_id_fld: Optional - Column in the origin points Spatially Enabled Dataframe uniquely identifying
            each feature. Default is 'LOCNUM'.
        destinations: Destination points in one of the supported input formats.
        dest_id_fld: Column in the destination points Spatially Enabled Dataframe uniquely identifying each feature
        destination_count: Integer number of destinations to search for from every origin point.
    Returns: Spatially Enabled Dataframe with a row for each origin id, and metrics for each nth destinations.
    """
    pass


def _get_nearest_solution_local(origin_df:pd.DataFrame, destination_df:pd.DataFrame, origin_id_fld: str = 'LOCNUM',
                                dest_id_fld: str = 'LOCNUM', destination_count: int = 4) -> pd.DataFrame:
    """Local implementation of get nearest solution."""
    # check to make sure network analyst is available using the env object to make it simplier
    env = util.Environment()
    if 'Network' in env.arcpy_extensions:
        env.arcpy_checkout_extension('Network')
    else:
        raise Exception('To perform network routing locally you must have access to the ArcGIS Network Analyst '
                        'extension. It appears this extension is either not installed or not licensed.')

    # ensure the dataframes are in the right schema and have the right geometry (points)
    origin_df = prep_sdf_for_nearest(origin_df, origin_id_fld)
    dest_df = prep_sdf_for_nearest(destination_df, dest_id_fld)

    # get the path to the network dataset from the registry
    network_dataset = get_ba_key_value('StreetsNetwork', origin_df._cntry.geo_name)

    # run the closest analysis locally
    closest_df = _get_closest_network_local(origin_df, dest_df, destination_count)

    # reformat the results to be a single row for each origin
    closest_df = _reformat_closest_result_dataframe(closest_df)

    return closest_df
