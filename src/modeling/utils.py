"""
Utility functions useful for the modeling module - the glue functions not fitting neatly anywhere else.
"""
from functools import wraps
import importlib
from pathlib import Path
from typing import AnyStr, Union

from arcgis.env import active_gis
from arcgis.gis import GIS, User
from arcgis.geometry import Geometry
import pandas as pd


def module_avail(module_name: AnyStr) -> bool:
    """
    Determine if module is available in this environment.
    """
    if importlib.util.find_spec(module_name) is not None:
        avail = True
    else:
        avail = False
    return avail


# flag a couple of useful module availabilities
avail_arcpy = module_avail('arcpy')
avail_shapely = module_avail('shapely')


def local_vs_gis(fn):
    """Decorator to facilitate bridging between local and remote resources."""
    # get the method - this will be used to redirect the function call
    fn_name = fn.__name__

    @wraps(fn)
    def wrapped(self, *args, **kwargs):

        # if performing analysis locally, try to access the function locally, but if not implemented, catch the error
        if self.source == 'local':
            try:
                fn_to_call = getattr(self, f'_{fn_name}_local')
            except AttributeError:
                raise AttributeError(f"'{fn_name}' not available using 'local' as the source.")

        # now, if performing analysis using a Web GIS, then access the function referencing remote resources
        elif isinstance(self.source, GIS):
            try:
                fn_to_call = getattr(self, f'_{fn_name}_gis')
            except AttributeError:
                raise AttributeError(f"'{fn_name}' not available using a Web GIS as the source.")

        # if another source, we don't plan on doing that any time soon
        else:
            raise AttributeError(f"'{self.source}' is not a recognized demographic modeling source.")

        return fn_to_call(*args, **kwargs)

    return wrapped


def local_business_analyst_avail() -> bool:
    """
    Check if a local installation of Business Analyst is available.
    """
    avail = False
    if avail_arcpy is True:
        import arcpy
        if arcpy.CheckExtension('Business'):
            avail = True
    return avail


def set_source(in_source: Union[str, GIS] = None) -> Union[str, GIS]:
    """
    Helper function to check source input. The source can be set explicitly, but if nothing is provided, it
    assumes the order of local first and then a Web GIS. Along the way, it also checks to see if a GIS object
    instance is available in the current session.
    """

    # if string input is provided, ensure setting to local and lowercase
    if isinstance(in_source, str):
        if in_source.lower() != 'local':
            raise Exception(f'Source must be either "local" or a Web GIS instance, not {in_source}.')

        # TODO: add check for local business analyst data
        elif in_source.lower() == 'local' and not local_business_analyst_avail():
            raise Exception(f'If using local source, the Business Analyst extension must be available')

        elif in_source.lower() == 'local':
            source = 'local'

    # if nothing provided, default to local if arcpy is available, and remote if arcpy not available
    if in_source is None and local_business_analyst_avail():
        source = 'local'

    # TODO: add check if web gis routing and enrich active - error if not available
    elif in_source is None and active_gis:
        source = active_gis

    # if not using local, use a GIS
    elif isinstance(in_source, GIS):
        source = in_source

    return source


def _assert_privileges_access(user: User) -> None:
    """Helper function determining if can access user privileges."""
    assert 'privileges' in user.keys(), f'Cannot access privileges of {user.username}. Please ensure either this is ' \
                                        'your username, or you are logged in as an administrator to be able to ' \
                                        'view privileges.'


def can_enrich_gis(user: User) -> bool:
    """ Determine if the provided user has data enrichment privileges in the Web GIS.

    .. note::

        The current user can be retrieved using `gis.users.me` for input.

    Args:
        user: Required `arcgis.gis.User` object instance.

    Returns:
       Boolean indicating if the provided user has enrichment privileges in the Web GIS.
    """
    # privileges may no be available
    _assert_privileges_access(user)

    # use list comprehension to account for future modifications to privilege descriptions
    bool_enrich = len([priv for priv in user['privileges'] if 'enrich' in priv]) > 0

    return bool_enrich


def has_networkanalysis_gis(user: User, network_function: str = None) -> bool:
    """Determine if the provided user has network analysis privileges in the Web GIS.

    .. note::

        The current user can be retrieved using `gis.users.me` for input.

    Args:
        user: Required `arcgis.gis.User` object instance.
        network_function: Optional string describing specific network function to check for.
            Valid values include 'closestfacility', 'locationallocation', 'optimizedrouting',
            'origindestinationcostmatrix', 'routing', 'servicearea', or 'vehiclerouting'.

    Returns: Boolean indicating if the provided user has network analysis privileges in the Web GIS.
    """
    # validate network_function if provided
    if network_function is not None:
        ntwrk_fn_lst = ['closestfacility', 'locationallocation', 'optimizedrouting', 'origindestinationcostmatrix',
                        'routing', 'servicearea', 'vehiclerouting']
        assert network_function in ntwrk_fn_lst, f'The network function provided, f"{network_function}," is not in ' \
                                                 f'the list of network functions [{", ".join(ntwrk_fn_lst)}.'

    # privileges may no be available
    _assert_privileges_access(user)

    # get the network analysis capabilities from the privileges
    prv_tail_lst = [prv.split(':')[-1] for prv in user['privileges'] if 'networkanalysis' in prv]

    # if nothing was found, there are not any capabilities
    if len(prv_tail_lst) == 0:
        bool_net = False

    # if no specific network capability is being requested, check for overall access
    elif not network_function:
        bool_net = 'networkanalysis' in prv_tail_lst

    # if a specific network function is being checked
    else:

        # just in case, ensure the input is lowercase
        network_function = network_function.lower()

        # check to ensure function is in capabilities determining if user has privileges
        bool_net = network_function in prv_tail_lst

    return bool_net


def geography_iterable_to_arcpy_geometry_list(geography_iterable: Union[pd.DataFrame, pd.Series, Geometry,
                                                                        list], geometry_filter: str = None) -> list:
    """
    Processing helper to convert a iterable of geography_levels to a list of ArcPy Geometry objects suitable for input
    into ArcGIS ArcPy geoprocessing tools.

    Args:
        geography_iterable:
            Iterable containing valid geometries.
        geometry_filter:
            String point|polyline|polygon to use for validation to ensure correct geometry type.

    Returns:
        List of ArcPy objects.
    """
    if not avail_arcpy:
        raise Exception('Converting to ArcPy geometry requires an environment with ArcPy available.')

    # do some error checking on the geometry filter
    geometry_filter = geometry_filter.lower() if geometry_filter is not None else geometry_filter
    if geometry_filter not in ['point', 'polyline', 'polygon'] and geometry_filter is not None:
        raise Exception(f'geometry_filter must be point|polyline|polygon {geometry_filter}')

    # if a DataFrame, check to ensure is spatial, and convert to list of arcgis Geometry objects
    if isinstance(geography_iterable, pd.DataFrame):
        if geography_iterable.spatial.validate() is True:
            geom_col = [col for col in geography_iterable.columns
                        if geography_iterable[col].dtype.name.lower() == 'geometry'][0]
            geom_lst = list(geography_iterable[geom_col].values)
        else:
            raise Exception('The provided geography_iterable DataFrame does not appear to be a Spatially Enabled '
                            'DataFrame or if so, all geometries do not appear to be valid.')

    # accommodate handling pd.Series input
    elif isinstance(geography_iterable, pd.Series):
        if 'SHAPE' not in geography_iterable.keys() or geography_iterable.name != 'SHAPE':
            raise Exception('SHAPE geometry field must be in the pd.Series or the pd.Series must be the SHAPE to use '
                            'a pd.Series as input.')

        # if just a single row passed in
        elif 'SHAPE' in geography_iterable.keys():
            geom_lst = [geography_iterable['SHAPE']]

        # otherwise, is a series of geometry objects, and just pass over to geom_lst since will be handled as iterable
        else:
            geom_lst = geography_iterable

    # if a list, ensure all child objects are polygon geometries and convert to list of arcpy.Geometry objects
    elif isinstance(geography_iterable, list):
        for geom in geography_iterable:
            if not isinstance(geom, Geometry):
                raise Exception('The provided geometries in the selecting_geometry list do not appear to all be '
                                'valid.')
        geom_lst = geography_iterable

    # if a single geometry object instance, ensure is polygon and make into single item list of arcpy.Geometry
    elif isinstance(geography_iterable, Geometry):
        geom_lst = [geography_iterable]

    else:
        raise Exception('geography_iterable must be either a Spatially Enabled Dataframe, pd.Series with a SHAPE '
                        f'column, list or single geometry object - not {type(geography_iterable)}.')

    # ensure all geometries are correct geometry type if provided
    if geometry_filter is not None:
        for geom in geom_lst:
            if geom.geometry_type != geometry_filter:
                raise Exception('geography_iterable geometries must be polygons. It appears you have provided at '
                                f'least one "{geom.geometry_type}" geometry.')

    # convert the objects in the geometry list to ArcPy Geometry objects
    arcpy_lst = [geom.as_arcpy for geom in geom_lst]

    return arcpy_lst
