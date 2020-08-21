from functools import wraps # This convenience func preserves name and docstring
import pathlib
from typing import IO, AnyStr

from arcgis.features import GeoAccessor
from arcgis.geometry import Geometry
from arcgis.features.geo._internals import register_dataframe_accessor
import pandas as pd
import swifter


# https://medium.com/@mgarod/dynamically-add-a-method-to-a-class-in-python-c49204b85bd6
def add_method(cls):
    """Helper decorator method for adding methods onto an existing object."""

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            return func(*args, **kwargs)

        # Note we are not binding func, but wrapper which accepts self but does exactly the same as func
        setattr(cls, func.__name__, wrapper)

        return func  # returning func means func can still be used normally

    return decorator

@register_dataframe_accessor("spatial")
class GeoAccessorIO(GeoAccessor):

    @staticmethod
    def _geometry_column_to_str(df: pd.DataFrame, geometry_column: str = 'SHAPE'):
        """
        Helper function to follow the paradigm of DRYD (don't repeat yourself, DUMMY!) for
            converting a column froma valid geometry type to an object(str). This is
            required to be able to save a Spatially Enabled DataFrameto flat outputs such
            as CSV and parquet.
    
        Args:
            df: Spatially Enabled DataFrame
            geometry_column: Optional: Column containing valid Esri Geometry objects. This
                only needs to be specified if the name of the column is not SHAPE.
    
        Returns: Pandas DataFrame ready for export.
        """
        # check to ensure the geometry field exists in the df
        if geometry_column not in df.columns.values:
            raise Exception(
                f'The geometry column provided, "{geometry_column}," does not appear to be in the df columns.')

        # convert the geometry object to a straing for export
        df[geometry_column] = df[geometry_column].swifter.allow_dask_on_strings(True).apply(
            lambda geom: geom.JSON)

        return df

    @staticmethod
    def _convert_geometry_column_to_geometry(df, geometry_column):
        """
        Helper function to follow the paradigm of DRYD (don't repeat yourself, DUMMY!) for
            converting a column from object (str) to a valid Geometry type, and enabling
            this column as the recognized geometry column for a fully functional Spatially
            Enabled Dataframe.

        Args:
            df: Pandas DataFrame with a column containing valid Esri JSON objects
                describing Geometries.
            geometry_column: Column containing the valid Esri Geometry object instances.

        Returns: Spatially Enabled Pandas DataFrame
        """
        # convert the values in the geometry column to geometry objects
        df[geometry_column] = df[geometry_column].swifter.allow_dask_on_strings(True).apply(
            lambda val: Geometry(eval(val)))

        # tell the GeoAccessor to recognize the column
        df.spatial.set_geometry(geometry_column)

        return df

    def read_csv(self, filepath_or_buffer: [str, pathlib.Path, IO[AnyStr]], geometry_column: str = 'SHAPE',
                 **kwargs) -> pd.DataFrame:
        """
        Read a CSV file and convert the geometry column to geometry objects for a fully
            functioning SpatiallyEnabled DataFrame. This function also accepts all valid
            input parameters as the Pandas `read_csv` function.
            https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.read_csv.html

        Args:
            filepath_or_buffer: String or Path object defining location of CSV file.
            geometry_column: Optional: String column name for the column containing the
                geometries as Esri JSON. This only needs to be specified if the name of the
                column is not SHAPE.

        Returns: Spatially Enabled Pandas DataFrame
        """
        # read in the pandas df just like normal
        self._data = pd.read_csv(filepath_or_buffer, **kwargs)

        # set the geometry column
        return self._convert_geometry_column_to_geometry(self._data, geometry_column)

    def read_parquet(self, path: [str, pathlib.Path, IO[AnyStr]], geometry_column: str = 'SHAPE', **kwargs) -> pd.DataFrame:
        """
        Read a parquet file and convert the geometry column to a geometry objects for a
            fully functioning Spatially Enabled DataFrame. This function also accepts all
            valid input parameters as the Pandas `read_parquet` function, meaning you can
            explicitly specify only the columns you want to read in. Also, to use this
            functionality, you must have a library supporting interacting with parquet files
            such as PyArrow or FastParquet.
            https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.read_parquet.html

        Args:
            path: String or Path object defining location of parquet file.
            geometry_column: Optional: String column name for the column containing the
                geometries as Esri JSON. This only needs to be specified if the name of the
                column is not SHAPE.

        Returns: Spatially Enabled Pandas DataFrame
        """
        # read in the pandas df just like normal
        self._data = self._data.read_csv(path, **kwargs)

        # set the geometry column
        return self._convert_geometry_column_to_geometry(self._data, geometry_column)

    def to_csv(self, path_or_buf:[str, pathlib.Path, IO[AnyStr]], geometry_column:str='SHAPE', **kwargs):
        """
        Export Spatially Enabled DataFrame to a CSV file and handling the special data type for
            the Geometry column. This function, in addition to the geometry column parameter,
            accepts all valid input parameters for the Pandas DataFrame `to_csv` function.
            https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_csv.html

        Args:
            path_or_buf: File path or object, if None is provided the result is returned as a
                string. If a file object is passed it should be opened with newline=’’,
                disabling universal newlines.
            geometry_column: Optional: Column containing valid Esri Geometry objects. This only
                needs to be specified if the name of the column is not SHAPE.

        Returns: String, path or IO object referencing the output.
        """
        # convert the geometry column to object(str) in the self._data property, which is the instance of the df
        self._data[geometry_column] = self._data[geometry_column].swifter.allow_dask_on_strings(True).apply(
            lambda val: val.JSON)

        # export just like normal
        return self._data.to_csv(path_or_buf, **kwargs)

    def to_parquet(self, path:[str, pathlib.Path, IO[AnyStr]], geometry_column:str= 'SHAPE', **kwargs):
        """
        Export Spatially Enabled DataFrame to a parquet file and handling the special data type for
            the Geometry column. This function, in addition to the geometry column parameter,
            accepts all valid input parameters for the Pandas DataFrame `to_parquet` function.
            https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_parquet.html

        Args:
            path: File path or object, if None is provided the result is returned as a
                string. If a file object is passed it should be opened with newline=’’,
                disabling universal newlines.
            geometry_column: Optional: Column containing valid Esri Geometry objects. This only
                needs to be specified if the name of the column is not SHAPE.

        Returns: String, path or IO object referencing the output.
        """
        # convert the geometry column to object(str) in the self._data property, which is the instance of the df
        self._data[geometry_column] = self._data[geometry_column].swifter.allow_dask_on_strings(True).apply(
            lambda val: val.JSON)

        # export just like normal
        return self._data.to_parquet(path, **kwargs)
