import astropy.table
import numpy as np
import pyarrow as pa

from hats_import.catalog.file_readers.input_reader import InputReader


def _np_to_pyarrow_array(array: np.ndarray, *, flatten_tensors: bool) -> pa.Array:
    """Convert a numpy array to a pyarrow"""
    # We usually have the "wrong" byte order from FITS
    array = np.asanyarray(array, dtype=array.dtype.newbyteorder("="))
    # "Base" type
    if array.ndim == 1:
        return pa.array(array)
    # Flat multidimensional nested values if asked
    if array.ndim > 2 and flatten_tensors:
        array = array.reshape(array.shape[0], -1)
    values = pa.array(array.reshape(-1))
    if array.ndim == 2:
        return pa.FixedSizeListArray.from_arrays(values, array.shape[1])
    # Use tensors if ndim > 2
    tensor_type = pa.fixed_shape_tensor(pa.from_numpy_dtype(array.dtype), array.shape[1:])
    return pa.FixedShapeTensorArray.from_storage(tensor_type, values)


def _astropy_to_pyarrow_table(astropy_table: astropy.table.Table, *, flatten_tensors: bool) -> pa.Table:
    """Convert astropy.table.Table to pyarrow.Table"""
    pa_arrays = {}
    for column in astropy_table.columns:
        np_array = astropy_table[column]
        pa_arrays[column] = _np_to_pyarrow_array(np_array, flatten_tensors=flatten_tensors)
    return pa.table(pa_arrays)


class FitsReader(InputReader):
    """Chunked FITS file reader.

    There are two column-level arguments for reading fits files:
    `column_names` and `skip_column_names`.

        - If neither is provided, we will read and process all columns in the fits file.
        - If `column_names` is given, we will use *only* those names, and
          `skip_column_names` will be ignored.
        - If `skip_column_names` is provided, we will remove those columns from processing stages.

    NB: Uses astropy table memmap to avoid reading the entire file into memory.
    See: https://docs.astropy.org/en/stable/io/fits/index.html#working-with-large-files


    Attributes:
        chunksize (int): number of rows of the file to process at once.
            For large files, this can prevent loading the entire file
            into memory at once.
        column_names (list[str]): list of column names to keep. only use
            one of `column_names` or `skip_column_names`
        skip_column_names (list[str]): list of column names to skip. only use
            one of `column_names` or `skip_column_names`
        flatten_tensors (bool): whether to flatten tensors. If True, the
            fixed-length list-array will be used, otherwise the arrow
            extension fixed-shape tensor will be used.
        kwargs: keyword arguments passed along to astropy.Table.read.
            See https://docs.astropy.org/en/stable/api/astropy.table.Table.html#astropy.table.Table.read
    """

    _default_kwargs = {"memmap": True}

    def __init__(
        self,
        chunksize=500_000,
        column_names=None,
        skip_column_names=None,
        flatten_tensors: bool = True,
        **kwargs,
    ):
        self.chunksize = chunksize
        self.column_names = column_names
        self.skip_column_names = skip_column_names
        self.flatten_tensors = flatten_tensors
        self.kwargs = self._default_kwargs | kwargs

    def read(self, input_file, read_columns=None):
        input_file = self.regular_file_exists(input_file)
        with input_file.open("rb") as file_handle:
            table = astropy.table.Table.read(file_handle, **self.kwargs)
            if read_columns:
                table.keep_columns(read_columns)
            elif self.column_names:
                table.keep_columns(self.column_names)
            elif self.skip_column_names:
                table.remove_columns(self.skip_column_names)

            for i_start in range(0, len(table), self.chunksize):
                table_chunk = table[i_start : i_start + self.chunksize]
                yield _astropy_to_pyarrow_table(table_chunk, flatten_tensors=self.flatten_tensors)
