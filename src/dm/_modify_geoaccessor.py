from functools import wraps # This convenience func preserves geographic_level and docstring
import json
import pathlib
from typing import IO, AnyStr

from arcgis.features import GeoAccessor
from arcgis.geometry import Geometry
from arcgis.features.geo._internals import register_dataframe_accessor
import numpy as np
import pandas as pd


# https://medium.com/@mgarod/dynamically-add-a-method-to-a-class-in-python-c49204b85bd6
def add_method(cls):
    """Helper decorator method for adding methods onto an existing object."""

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            return func(*args, **kwargs)

        # Note we are not binding func, but wrapper which accepts cls but does exactly the same as func
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
                only needs to be specified if the geographic_level of the column is not SHAPE.
    
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
        df[geometry_column] = df[geometry_column].apply(lambda val: Geometry(json.loads(val)))

        # tell the GeoAccessor to recognize the column
        df.spatial.set_geometry(geometry_column)

        return df

    @classmethod
    def read_csv(cls, filepath_or_buffer: [str, pathlib.Path, IO[AnyStr]], geometry_column: str = 'SHAPE',
                 **kwargs) -> pd.DataFrame:
        """
        Read a CSV file and convert the geometry column to geometry objects for a fully
            functioning SpatiallyEnabled DataFrame. This function also accepts all valid
            input parameters as the Pandas `read_csv` function.
            https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.read_csv.html

        Args:
            filepath_or_buffer: String or Path object defining location of CSV file.
            geometry_column: Optional: String column geographic_level for the column containing the
                geometries as Esri JSON. This only needs to be specified if the geographic_level of the
                column is not SHAPE.

        Returns: Spatially Enabled Pandas DataFrame
        """
        # read in the pandas df just like normal
        cls._data = pd.read_csv(filepath_or_buffer, **kwargs)

        # set the geometry column
        return cls._convert_geometry_column_to_geometry(cls._data, geometry_column)

    @classmethod
    def read_parquet(cls, path: [str, pathlib.Path, IO[AnyStr]],
                     geometry_column: str = 'SHAPE', **kwargs) -> pd.DataFrame:
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
            geometry_column: Optional: String column geographic_level for the column containing the
                geometries as Esri JSON. This only needs to be specified if the geographic_level of the
                column is not SHAPE.

        Returns: Spatially Enabled Pandas DataFrame
        """
        # read in the pandas df just like normal
        cls._data = cls._data.read_csv(path, **kwargs)

        # set the geometry column
        return cls._convert_geometry_column_to_geometry(cls._data, geometry_column)

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
                needs to be specified if the geographic_level of the column is not SHAPE.

        Returns: String, path or IO object referencing the output.
        """
        # copy the output dataframe to not modify the dataframe in place
        out_df = self._data.copy()

        # convert the geometry column to object(str) in the cls._data property, which is the instance of the df
        out_df[geometry_column] = out_df[geometry_column].apply(lambda val: val.JSON)

        # export just like normal
        return out_df.to_csv(path_or_buf, **kwargs)

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
                needs to be specified if the geographic_level of the column is not SHAPE.

        Returns: String, path or IO object referencing the output.
        """
        # copy the output dataframe to not modify the dataframe in place
        out_df = self._data.copy()

        # convert the geometry column to object(str) in the cls._data property, which is the instance of the df
        out_df[geometry_column] = out_df[geometry_column].apply(lambda val: val.JSON)

        # export just like normal
        return out_df.to_parquet(path, **kwargs)

    def enrich(self, enrich_variables: [list, np.array, pd.Series] = None,
               data_collections: [str, list, np.array, pd.Series] = None) -> pd.DataFrame:
        """
        Enrich the DataFrame using the provided enrich variable list or data collections list. Either a variable list
            or list of data collections can be provided, but not both.
        Args:
            enrich_variables: List of data variables for enrichment.
            data_collections: List of data collections for enrichment.

        Returns: pd.DataFrame with enriched data.
        """
        # get the data from the GeoAccessor _data property
        data = self._data

        #TODO: Ensure the environment is using the correct data source

        # get the country from the data
        cntry = data._cntry

        # invoke the enrich method from the country
        out_df = cntry.enrich(data, enrich_variables, data_collections)

        return out_df
