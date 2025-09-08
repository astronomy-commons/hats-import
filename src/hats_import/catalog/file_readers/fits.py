import astropy.table
import numpy as np
import pyarrow as pa
from astropy.io import fits

from hats_import.catalog.file_readers.input_reader import InputReader


def _np_to_pyarrow_array(array: np.ndarray, *, flatten_tensors: bool) -> pa.Array:
    """Convert a numpy array to a pyarrow"""
    # We usually have the "wrong" byte order from FITS
    array = np.asanyarray(array, dtype=array.dtype.newbyteorder("="))
    values = pa.array(array.reshape(-1))
    # "Base" type
    if array.ndim == 1:
        return values
    # Flat multidimensional nested values if asked
    if flatten_tensors and array.ndim > 2:
        array = array.reshape(array.shape[0], -1)
    pa_list_array = pa.FixedSizeListArray.from_arrays(values, np.prod(array.shape[1:]))
    # An extra dimension is represented as a list array
    if array.ndim == 2:
        return pa_list_array
    # array.ndim > 2
    # Multiple extra dimensions are represented as a tensor array
    tensor_type = pa.fixed_shape_tensor(values.type, shape=array.shape[1:])
    return pa.FixedShapeTensorArray.from_storage(tensor_type, pa_list_array)


def _astropy_to_pyarrow_table(astropy_table: astropy.table.Table, *, flatten_tensors: bool) -> pa.Table:
    """Convert astropy.table.Table to pyarrow.Table"""
    pa_arrays = {}
    for column in astropy_table.columns:
        np_array = np.asarray(astropy_table[column])
        pa_arrays[column] = _np_to_pyarrow_array(np_array, flatten_tensors=flatten_tensors)
    return pa.table(pa_arrays)


def _first_table_hdu(hdul: fits.HDUList) -> int:
    """Get an index of the first HDU with a table"""
    for i, hdu in enumerate(hdul):
        if isinstance(hdu, (fits.TableHDU, fits.BinTableHDU, fits.GroupsHDU)):
            return i
    raise ValueError("No HDU with a table found")


class FitsReader(InputReader):
    """Chunked FITS file reader.

    There are two column-level arguments for reading fits files: ``column_names`` and ``skip_column_names``.

    - If neither is provided, we will read and process all columns in the fits file.
    - If ``column_names`` is given, we will use _only_ those names, and ``skip_column_names`` will be ignored.
    - If ``skip_column_names`` is provided, we will remove those columns from processing stages.

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
        hdu (int | None): index of HDU to read. If None, use the first HDU
            with a table.
        flatten_tensors (bool): whether to flatten tensors. If True, the
            fixed-length list-array will be used, otherwise the arrow
            extension fixed-shape tensor will be used.
        fits_kwargs: keyword arguments passed along to ``astropy.io.fits.open(file_handler, **kwargs)``.
            See https://docs.astropy.org/en/stable/io/fits/api/files.html#astropy.io.fits.open
        table_kwargs: keyword arguments passed along to ``astropy.Table.read(hdu, **kwargs)``.
            See https://docs.astropy.org/en/stable/api/astropy.table.Table.html#astropy.table.Table.read
    """

    # First, we open FITS file "lazily": with the memory map and not decoding bytes to UTF8
    _default_fits_kwargs = {"memmap": True, "character_as_bytes": True}
    # Next, for each batch, we decode bytes
    _default_table_kwargs = {
        "format": "fits",
        "character_as_bytes": False,
    }

    def __init__(
        self,
        chunksize=500_000,
        column_names=None,
        skip_column_names=None,
        hdu: int | None = None,
        flatten_tensors: bool = True,
        fits_kwargs: dict[str, object] | None = None,
        table_kwargs: dict[str, object] | None = None,
    ):
        self.chunksize = chunksize
        self.column_names = column_names
        self.skip_column_names = None if skip_column_names is None else frozenset(skip_column_names)
        self.hdu_index = hdu
        self.flatten_tensors = flatten_tensors
        self.fits_kwargs = self._default_fits_kwargs | (fits_kwargs or {})
        self.table_kwargs = self._default_table_kwargs | (table_kwargs or {})

    def read(self, input_file, read_columns=None):
        input_file = self.regular_file_exists(input_file)
        with input_file.open("rb") as file_handle, fits.open(file_handle, **self.fits_kwargs) as hdul:
            if self.hdu_index is None:
                hdu_index = _first_table_hdu(hdul)
                hdu = hdul[hdu_index]
            else:
                hdu = hdul[self.hdu_index]

            column_names = hdu.columns.names
            if read_columns is not None:
                column_names = read_columns
            elif self.column_names is not None:
                column_names = self.column_names
            elif self.skip_column_names is not None:
                column_names = [col for col in column_names if col not in self.skip_column_names]

            for i_start in range(0, hdu.data.shape[0], self.chunksize):
                data_chunk = hdu.data[i_start : i_start + self.chunksize]
                hdu_chunk = type(hdu)(
                    data=data_chunk,
                    header=hdu.header,
                    ver=hdu.ver,
                )
                table_chunk = astropy.table.Table.read(hdu_chunk, **self.table_kwargs)
                yield _astropy_to_pyarrow_table(
                    table_chunk[column_names], flatten_tensors=self.flatten_tensors
                )
