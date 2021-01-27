import importlib
import os
from pathlib import Path
import tempfile
import uuid

from arcgis.features import GeoAccessor
from arcgis.geometry import Geometry
from arcgis.gis import GIS
import pandas as pd

from . import utils
from .country import Country
from ._registry import get_ba_key_value

arcpy_avail = True if importlib.util.find_spec("arcpy") else False

if arcpy_avail:
    import arcpy

# location to store temp files if necessary
csv_file_prefix = 'temp_closest'
temp_file_root = os.path.join(tempfile.gettempdir(), csv_file_prefix)

# ensure previous runs do not interfere
arcpy.env.overwriteOutput = True


def _prep_sdf_for_nearest(input_dataframe: pd.DataFrame, id_column: str):
    """
    Given an input Spatially Enabled Dataframe, prepare it to work
        well with the nearest solver.

    Args:
        input_dataframe: Spatially Enabled Dataframe with really
            any geometry.
        id_column: Field uniquely identifying each of location to
            be used for routing to nearest.

    Returns: Spatially Enabled Dataframe of points with correct
        columns for routing to nearest.
    """
    # check inputs
    assert isinstance(input_dataframe, pd.DataFrame), f'The input dataframe must be a Pandas DataFrame, not ' \
                                                      f'{type(input_dataframe)}.'

    # ensure the geometry is set
    geom_col_lst = [c for c in input_dataframe.columns if input_dataframe[c].dtype.name.lower() == 'geometry']
    assert len(geom_col_lst) > 0, 'The DataFrame does not appear to have a geometry column defined. This can be ' \
                                  'accomplished using the "input_dataframe.spatial.set_geometry" method.'
    geom_col = geom_col_lst[0]

    # ensure the column is in the dataframe columns
    assert id_column in input_dataframe.columns, f'The provided id_column, "{id_column}," does not appear to be in ' \
                                                 f'the columns [{", ".join(input_dataframe.columns)}]"'

    # par down the input dataframe to just the columns needed
    input_dataframe = input_dataframe[[id_column, geom_col]].copy()

    # rename the columns to follow the schema needed for routing
    input_dataframe.columns = ['ID', 'SHAPE']

    # ensure the spatial reference is WGS84 - if not, make it so
    if input_dataframe.spatial.sr.wkid != 4326:
        input_dataframe = input_dataframe.dm.project(4326)

    # if the geometry is not points, we still need points, so get the geometric centroids
    if input_dataframe.spatial.geometry_type != ['point']:
        input_dataframe['SHAPE'] = input_dataframe[geom_col].apply(
            lambda geom: Geometry({'x': geom.centroid[0], 'y': geom.centroid[1], 'spatialReference': {'wkid': 4326}}))
        input_dataframe.spatial.set_geometry('SHAPE')

    # add a second column for the ID as Name
    input_dataframe['Name'] = input_dataframe['ID']

    # ensure the geometry is correctly being recognized
    input_dataframe.spatial.set_geometry('SHAPE')

    # set the order of the columns and return
    return input_dataframe[['ID', 'Name', 'SHAPE']].copy()


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

        # remove the temporary table to ensure not stuff lying around and consuming RAM
        arcpy.management.Delete(temp_table)

        # get the maximum near distance (in meters)
        max_near_dist = meters * 0.00062137

    return max_near_dist


def _get_nearest_solve_local(origin_dataframe: pd.DataFrame, destination_dataframe: pd.DataFrame,
                             destination_count: int, network_dataset: [Path, str],
                             maximum_distance: [int, float] = None):
    """
    Perform network solve using local resources with assumption of standard input.

    Args:
        origin_dataframe: Origin points Spatially Enabled Dataframe
        destination_dataframe: Destination points Spatially Enabled Dataframe
        destination_count: Destination points Spatially Enabled Dataframe
        network_dataset: Path to ArcGIS Network dataset for performing routing.
        maximum_distance: Maximum nearest routing distance in miles.

    Returns: Spatially Enabled Dataframe of solved closest facility routes.
    """
    # make sure the path to the network dataset is a string
    network_dataset = str(network_dataset) if isinstance(network_dataset, Path) else network_dataset

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
    closest_solver.defaultTargetFacilityCount = destination_count
    closest_solver.routeShapeType = arcpy.nax.RouteShapeType.TrueShapeWithMeasures
    closest_solver.searchTolerance = 5000
    closest_solver.searchToleranceUnits = arcpy.nax.DistanceUnits.Meters

    # since maximum distance is optional, well, make it optional
    if maximum_distance is not None:
        closest_solver.defaultImpedanceCutoff = maximum_distance

    # load the origin and destination feature data frames into memory and load into the solver object instance
    # TODO: test if can use 'memory' workspace instead of scratch
    origin_fc = origin_dataframe.spatial.to_featureclass(os.path.join(arcpy.env.scratchGDB, 'origin_tmp'))
    closest_solver.load(arcpy.nax.ClosestFacilityInputDataType.Incidents, origin_fc)

    dest_fc = destination_dataframe.spatial.to_featureclass(os.path.join(arcpy.env.scratchGDB, 'dest_tmp'))
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


def _reformat_closest_result_dataframe(closest_df: pd.DataFrame):
    """
    Reformat the schema, dropping unneeded columns and renaming those kept to be more in line with this workflow.

    Args:
        closest_df: Dataframe of the raw output routes from the find closest analysis.

    Returns: Spatially Enabled Dataframe reformatted.
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


def _explode_closest_rank_dataframe(closest_df: pd.DataFrame, origin_id_col: str = 'origin_id',
                                    rank_col: str = 'destination_rank',
                                    dest_id_col: str = 'destination_id',
                                    dest_keep_cols: list = None):
    """
    Effectively explode out or pivot the data so there is only a single record for each origin.

    Args:
        closest_df: Spatially Enabled Dataframe reformatted from the raw output of find nearest.
        origin_id_col: Column uniquely identifying each origin - default 'origin_id'
        rank_col: Column identifying the rank of each destination - default 'destination_rank'
        dest_id_col: Column uniquely identifying each destination - default 'destination_id'

    Returns: Dataframe with a single row for each origin with multiple destination metrics for each.
    """
    # create a dataframe to start working with comprised of only the unique origin_dataframe to start with
    origin_dest_df = pd.DataFrame(closest_df[origin_id_col].unique(), columns=[origin_id_col])

    # get a list of the proximity columns
    proximity_cols = [col for col in closest_df.columns if col.startswith('proximity_')]

    # add any destination columns
    if dest_keep_cols:
        proximity_cols = proximity_cols + dest_keep_cols

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


def _get_nearest_local(origin_dataframe: pd.DataFrame, destination_dataframe: pd.DataFrame,
                       network_dataset: [str, Path], single_row_per_origin=True, origin_id_column: str = 'LOCNUM',
                       destination_id_column: str = 'LOCNUM', destination_count: int = 4,
                       dest_cols: list = None) -> pd.DataFrame:
    """Local implementation of get nearest solution."""
    # check to make sure network analyst is available using the env object to make it simplier
    env = utils.Environment()
    if 'Network' in env.arcpy_extensions:
        env.arcpy_checkout_extension('Network')
    else:
        raise Exception('To perform network routing locally you must have access to the ArcGIS Network Analyst '
                        'extension. It appears this extension is either not installed or not licensed.')

    # ensure the dataframes are in the right schema and have the right geometry (points)
    origin_net_df = _prep_sdf_for_nearest(origin_dataframe, origin_id_column)
    dest_net_df = _prep_sdf_for_nearest(destination_dataframe, destination_id_column)

    # run the closest analysis locally
    closest_df = _get_nearest_solve_local(origin_net_df, dest_net_df, destination_count, network_dataset)

    # reformat and standardize the output
    std_clstst_df = _reformat_closest_result_dataframe(closest_df)

    if dest_cols:
        if len(dest_cols):
            # add the columns onto the near dataframe for output
            dest_join_df = destination_dataframe[dest_cols].set_index(destination_id_column)
            std_clstst_df = std_clstst_df.join(dest_join_df, on='destination_id')

    # pivot and explode the results to be a single row for each origin if desired
    if single_row_per_origin:
        xplod_dest_cols = [c for c in dest_cols if c != destination_id_column]
        out_df = _explode_closest_rank_dataframe(std_clstst_df, dest_keep_cols=xplod_dest_cols)
    else:
        out_df = std_clstst_df

    return out_df


def get_nearest(origin_dataframe: pd.DataFrame, destination_dataframe: pd.DataFrame,
                source: [str, Path, Country, GIS], single_row_per_origin: bool = True,
                origin_id_column: str = 'LOCNUM', destination_id_column: str = 'LOCNUM',
                destination_count: int = 4, near_prefix: str = None,
                destination_columns_to_keep: [str, list] = None) -> pd.DataFrame:
    """
    Create a closest destination dataframe using origin and destination Spatially Enabled
        Dataframes, but keep each origin and destination still in a discrete row instead
        of collapsing to a single row per origin. The main reason to use this is if
        needing the geometry for visualization.

    Args:
        origin_dataframe: Origins for networks solves.
        destination_dataframe: Destination points in one of the supported input formats.
        source: Either the path to the network dataset, the Country object associated with
            the Business Analyst source being used, or a GIS object instance.
        single_row_per_origin: Optional - Whether or not to pivot the results to return
            only one row for each origin location. Default is True.
        origin_id_column: Optional - Column in the origin points Spatially Enabled Dataframe
            uniquely identifying each feature. Default is 'LOCNUM'.
        destination_id_column: Column in the destination points Spatially Enabled Dataframe
            uniquely identifying each feature
        destination_count: Integer number of destinations to search for from every origin
            point.
        near_prefix: String prefix to prepend onto near column names in the output.
        destination_columns_to_keep: List of columns to keep in the output. Commonly, if
            businesses, this includes the column with the business names.

    Returns: Spatially Enabled Dataframe with a row for each origin id, and metrics for
        each nth destinations.
    """

    for df in [origin_dataframe, destination_dataframe]:
        assert isinstance(df, pd.DataFrame), 'Origin and destination dataframes must both be pd.DataFrames'
        assert df.spatial.validate(), 'Origin and destination dataframes must be valid Spatially enabled DataFrames.' \
                                      'This can be checked using df.spatial.validate()'

    assert isinstance(source, (str, Path, Country, GIS)), 'source must be either a path to the network dataset, a ' \
                                                          'dm.Country object instance, or a reference to a GIS.'

    assert isinstance(single_row_per_origin, bool)

    assert origin_id_column in origin_dataframe.columns, f'The provided origin_id_column does not appear to be in ' \
                                                         f'the origin_dataframe columns ' \
                                                         f'[{", ".join(origin_dataframe.columns)}]'

    assert destination_id_column in destination_dataframe.columns, f'The provided destination_id_column does not ' \
                                                                   f'appear to be in the destination_dataframe ' \
                                                                   f'columns ' \
                                                                   f'[{", ".join(destination_dataframe.columns)}]'

    # if the source is a country set to local, we are using Business Analyst, so interrogate the source
    if isinstance(source, Country):

        # if local, get the path to the network dataset
        if source.source == 'local':
            source = get_ba_key_value('StreetsNetwork', source.geo_name)

        # if not local, set the source to the GIS object instance
        else:
            source = source.source

    # if the source is a path, convert it to a string because arcpy doesn't do well with path objects
    source = str(source) if isinstance(source, Path) else source

    # if a path, ensure it exists
    if isinstance(source, str):
        assert arcpy.Exists(source), f'The path to the network dataset provided does not appear to exist - ' \
                                     f'"{str(source)}".'

    # include any columns to be retained in the output
    if destination_columns_to_keep is not None:

        # if just a single column is provided in a string, make it into a list
        if isinstance(destination_columns_to_keep, list):
            dest_cols = destination_columns_to_keep
        else:
            dest_cols = [destination_columns_to_keep]

        # make sure the destination columns include the id columns
        dest_cols = dest_cols if destination_id_column in dest_cols else [destination_id_column] + dest_cols

        # check all the columns to make sure they are in the output dataframe
        for col in dest_cols:
            assert col in destination_dataframe.columns, f'One of the destination_columns_to_keep {col}, does not ' \
                                                         f'appear to be in the destination_dataframe columns ' \
                                                         f'[{", ".join(destination_dataframe.columns)}].'

    # if no columns, just populate an empty list so nested functions work
    else:
        dest_cols = []

    # now, the source is either a path to the network source or a GIS object instance, so call each as necessary
    if isinstance(source, str):
        near_df = _get_nearest_local(origin_dataframe, destination_dataframe, source, single_row_per_origin,
                                     origin_id_column, destination_id_column, destination_count, dest_cols)

    else:
        raise Exception('Nearest not yet implemented using GIS object instance.')

    # add prefixes to columns if provided
    if near_prefix is not None:
        near_df.columns = [f'{near_prefix}_{c}' for c in near_df.columns]
        near_oid_col = f'{near_prefix}_origin_id'
    else:
        near_oid_col = 'origin_id'

    # add results to input data
    if single_row_per_origin:

        out_df = origin_dataframe.join(near_df.set_index(near_oid_col), on=origin_id_column)
    else:
        out_df = near_df.join(origin_dataframe.drop(columns='SHAPE').set_index(origin_id_column), on=near_oid_col)

        out_df.columns = [c if not c.endswith('_SHAPE') else 'SHAPE' for c in out_df.columns]

    # recognize geometry
    out_df.spatial.set_geometry('SHAPE')

    return out_df
