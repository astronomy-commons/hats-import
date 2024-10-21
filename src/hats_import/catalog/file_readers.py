"""File reading generators for common file types."""

import abc

import pandas as pd
import pyarrow as pa
import pyarrow.dataset
import pyarrow.parquet as pq
from astropy.io import ascii as ascii_reader
from astropy.table import Table
from hats.io import file_io
from upath import UPath

# pylint: disable=too-few-public-methods,too-many-arguments


def get_file_reader(
    file_format,
    chunksize=500_000,
    schema_file=None,
    column_names=None,
    skip_column_names=None,
    type_map=None,
    **kwargs,
):
    """Get a generator file reader for common file types

    Args:
        file_format (str): specifier for the file type and extension.
            Currently supported formats include:

            - `csv`, comma separated values. may also be tab- or pipe-delimited
              includes `.csv.gz` and other compressed csv files
            - `fits`, flexible image transport system. often used for astropy tables.
            - `parquet`, compressed columnar data format
            - `ecsv`, astropy's enhanced CSV
            - `indexed_csv`, "index" style reader, that accepts a file with a list
              of csv files that are appended in-memory
            - `indexed_parquet`, "index" style reader, that accepts a file with a list
              of parquet files that are appended in-memory

        chunksize (int): number of rows to read in a single iteration.
            for single-file readers, large files are split into batches based on this value.
            for index-style readers, we read files until we reach this chunksize and
            create a single batch in-memory.
        schema_file (str): path to a parquet schema file. if provided, header names
            and column types will be pulled from the parquet schema metadata.
        column_names (list[str]): for CSV files, the names of columns if no header
            is available. for fits files, a list of columns to *keep*.
        skip_column_names (list[str]): for fits files, a list of columns to remove.
        type_map (dict): for CSV files, the data types to use for columns
        kwargs: additional keyword arguments to pass to the underlying file reader.
    """
    if file_format == "csv":
        return CsvReader(
            chunksize=chunksize,
            schema_file=schema_file,
            column_names=column_names,
            type_map=type_map,
            **kwargs,
        )
    if file_format == "ecsv":
        return AstropyEcsvReader(**kwargs)
    if file_format == "fits":
        return FitsReader(
            chunksize=chunksize,
            column_names=column_names,
            skip_column_names=skip_column_names,
            **kwargs,
        )
    if file_format == "parquet":
        return ParquetReader(chunksize=chunksize, **kwargs)
    if file_format == "indexed_csv":
        return IndexedCsvReader(
            chunksize=chunksize,
            schema_file=schema_file,
            column_names=column_names,
            type_map=type_map,
            **kwargs,
        )
    if file_format == "indexed_parquet":
        return IndexedParquetReader(chunksize=chunksize, **kwargs)
    raise NotImplementedError(f"File Format: {file_format} not supported")


class InputReader(abc.ABC):
    """Base class for chunking file readers."""

    @abc.abstractmethod
    def read(self, input_file, read_columns=None):
        """Read the input file, or chunk of the input file.

        Args:
            input_file(str): path to the input file.
            read_columns(List[str]): subset of columns to read.
                if None, all columns are read
        Yields:
            DataFrame containing chunk of file info.
        """

    def regular_file_exists(self, input_file, **_kwargs):
        """Check that the `input_file` points to a single regular file

        Raises:
            FileNotFoundError: if nothing exists at path, or directory found.
        """
        if not file_io.does_file_or_directory_exist(input_file):
            raise FileNotFoundError(f"File not found at path: {input_file}")
        if not file_io.is_regular_file(input_file):
            raise FileNotFoundError(f"Directory found at path - requires regular file: {input_file}")

    def read_index_file(self, input_file, upath_kwargs=None, **kwargs):
        """Read an "indexed" file.

        This should contain a list of paths to files to be read and batched.

        In order to create a valid connection to the string paths, provide any
        additional universal pathlib (i.e. fsspec) arguments to the `upath_kwargs` kwarg.
        In this way, the "index" file may contain a list of paths on a remote service,
        and the `upath_kwargs` will be used to create a connection to that remote service.

        Raises:
            FileNotFoundError: if nothing exists at path, or directory found.
        """
        input_file = file_io.get_upath(input_file)
        self.regular_file_exists(input_file, **kwargs)
        file_names = file_io.load_text_file(input_file)
        file_names = [f.strip() for f in file_names]
        if upath_kwargs is None:
            upath_kwargs = {}

        file_paths = [UPath(f, **upath_kwargs) for f in file_names if f]

        return file_paths


class CsvReader(InputReader):
    """CSV reader for the most common CSV reading arguments.

    This uses `pandas.read_csv`, and you can find more information on
    additional arguments in the pandas documentation:
    https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html

    Attributes:
        chunksize (int): number of rows to read in a single iteration.
        header (int, list of int, None, default 'infer'): rows to
            use as the header with column names
        schema_file (str): path to a parquet schema file. if provided, header names
            and column types will be pulled from the parquet schema metadata.
        column_names (list[str]): the names of columns if no header is available
        type_map (dict): the data types to use for columns
        parquet_kwargs (dict): additional keyword arguments to use when
            reading the parquet schema metadata, passed to pandas.read_parquet.
            See https://pandas.pydata.org/docs/reference/api/pandas.read_parquet.html
        kwargs (dict): additional keyword arguments to use when reading
            the CSV files with pandas.read_csv.
            See https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
    """

    def __init__(
        self,
        chunksize=500_000,
        header="infer",
        schema_file=None,
        column_names=None,
        type_map=None,
        parquet_kwargs=None,
        upath_kwargs=None,
        **kwargs,
    ):
        self.chunksize = chunksize
        self.header = header
        self.schema_file = schema_file
        self.column_names = column_names
        self.type_map = type_map
        self.parquet_kwargs = parquet_kwargs
        self.upath_kwargs = upath_kwargs
        self.kwargs = kwargs

        schema_parquet = None
        if self.schema_file:
            if self.parquet_kwargs is None:
                self.parquet_kwargs = {}
            schema_parquet = file_io.read_parquet_file_to_pandas(
                self.schema_file,
                **self.parquet_kwargs,
            )

        if self.column_names:
            self.kwargs["names"] = self.column_names
        elif not self.header and schema_parquet is not None:
            self.kwargs["names"] = list(schema_parquet.columns)

        if self.type_map:
            self.kwargs["dtype"] = self.type_map
        elif schema_parquet is not None:
            self.kwargs["dtype"] = schema_parquet.dtypes.to_dict()

    def read(self, input_file, read_columns=None):
        self.regular_file_exists(input_file, **self.kwargs)

        if read_columns:
            self.kwargs["usecols"] = read_columns

        return file_io.load_csv_to_pandas_generator(
            input_file,
            chunksize=self.chunksize,
            header=self.header,
            **self.kwargs,
        )


class IndexedCsvReader(CsvReader):
    """Reads an index file, containing paths to CSV files to be read and batched

    See CsvReader for additional configuration for reading CSV files.
    """

    def read(self, input_file, read_columns=None):
        file_paths = self.read_index_file(
            input_file=input_file, upath_kwargs=self.upath_kwargs, **self.kwargs
        )

        batch_size = 0
        batch_frames = []
        for file in file_paths:
            for single_frame in super().read(file, read_columns=read_columns):
                if batch_size + len(single_frame) >= self.chunksize:
                    # We've hit our chunksize, send the batch off to the task.
                    if len(batch_frames) == 0:
                        yield single_frame
                        batch_size = 0
                    else:
                        yield pd.concat(batch_frames, ignore_index=True)
                        batch_frames = []
                        batch_frames.append(single_frame)
                        batch_size = len(single_frame)
                else:
                    batch_frames.append(single_frame)
                    batch_size += len(single_frame)

        if len(batch_frames) > 0:
            yield pd.concat(batch_frames, ignore_index=True)


class AstropyEcsvReader(InputReader):
    """Reads astropy ascii .ecsv files.

    Note that this is NOT a chunked reader. Use caution when reading
    large ECSV files with this reader.

    Attributes:
        kwargs: keyword arguments passed to astropy ascii reader.
            See https://docs.astropy.org/en/stable/api/astropy.io.ascii.read.html#astropy.io.ascii.read
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def read(self, input_file, read_columns=None):
        self.regular_file_exists(input_file, **self.kwargs)
        if read_columns:
            self.kwargs["include_names"] = read_columns

        astropy_table = ascii_reader.read(input_file, format="ecsv", **self.kwargs)
        yield astropy_table.to_pandas()


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
        kwargs: keyword arguments passed along to astropy.Table.read.
            See https://docs.astropy.org/en/stable/api/astropy.table.Table.html#astropy.table.Table.read
    """

    def __init__(self, chunksize=500_000, column_names=None, skip_column_names=None, **kwargs):
        self.chunksize = chunksize
        self.column_names = column_names
        self.skip_column_names = skip_column_names
        self.kwargs = kwargs

    def read(self, input_file, read_columns=None):
        self.regular_file_exists(input_file, **self.kwargs)
        table = Table.read(input_file, memmap=True, **self.kwargs)
        if read_columns:
            table.keep_columns(read_columns)
        elif self.column_names:
            table.keep_columns(self.column_names)
        elif self.skip_column_names:
            table.remove_columns(self.skip_column_names)

        total_rows = len(table)
        read_rows = 0

        while read_rows < total_rows:
            df_chunk = table[read_rows : read_rows + self.chunksize].to_pandas()
            for column in df_chunk.columns:
                if (
                    df_chunk[column].dtype == object
                    and df_chunk[column].apply(lambda x: isinstance(x, bytes)).any()
                ):
                    df_chunk[column] = df_chunk[column].apply(
                        lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
                    )

            yield df_chunk

            read_rows += self.chunksize


class ParquetReader(InputReader):
    """Parquet reader for the most common Parquet reading arguments.

    Attributes:
        chunksize (int): number of rows of the file to process at once.
            For large files, this can prevent loading the entire file
            into memory at once.
        column_names (list[str] or None): Names of columns to use from the input dataset.
            If None, use all columns.
        kwargs: arguments to pass along to pyarrow.parquet.ParquetFile.
            See https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetFile.html
    """

    def __init__(self, chunksize=500_000, column_names=None, **kwargs):
        self.chunksize = chunksize
        self.column_names = column_names
        self.kwargs = kwargs

    def read(self, input_file, read_columns=None):
        self.regular_file_exists(input_file, **self.kwargs)
        columns = read_columns or self.column_names
        parquet_file = pq.ParquetFile(input_file, **self.kwargs)
        for smaller_table in parquet_file.iter_batches(
            batch_size=self.chunksize, columns=columns, use_pandas_metadata=True
        ):
            yield smaller_table.to_pandas()


class ParquetPyarrowReader(InputReader):
    """Parquet reader for the most common Parquet reading arguments.

    Attributes:
        chunksize (int): number of rows of the file to process at once.
            For large files, this can prevent loading the entire file
            into memory at once.
        column_names (list[str] or None): Names of columns to use from the input dataset.
            If None, use all columns.
        kwargs: arguments to pass along to pyarrow.parquet.ParquetFile.
            See https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetFile.html
    """

    def __init__(self, chunksize=500_000, column_names=None, **kwargs):
        self.chunksize = chunksize
        self.column_names = column_names
        self.kwargs = kwargs

    def read(self, input_file, read_columns=None):
        self.regular_file_exists(input_file, **self.kwargs)
        columns = read_columns or self.column_names
        parquet_file = pq.ParquetFile(input_file, **self.kwargs)
        for smaller_table in parquet_file.iter_batches(batch_size=self.chunksize, columns=columns):
            table = pa.Table.from_batches([smaller_table])
            table = table.replace_schema_metadata()
            yield table


class IndexedParquetReader(InputReader):
    """Reads an index file, containing paths to parquet files to be read and batched

    Attributes:
        chunksize (int): maximum number of rows to process at once.
            Large files will be processed in chunks. Small files will be concatenated.
            Also passed to pyarrow.dataset.Dataset.to_batches as `batch_size`.
        batch_readahead (int): number of batches to read ahead.
            Passed to pyarrow.dataset.Dataset.to_batches.
        fragment_readahead (int): number of fragments to read ahead.
            Passed to pyarrow.dataset.Dataset.to_batches.
        use_threads (bool): whether to use multiple threads for reading.
            Passed to pyarrow.dataset.Dataset.to_batches.
        column_names (list[str] or None): Names of columns to use from the input dataset.
            If None, use all columns.
        kwargs: additional arguments to pass along to InputReader.read_index_file.
    """

    def __init__(
        self,
        chunksize=500_000,
        batch_readahead=16,
        fragment_readahead=4,
        use_threads=True,
        column_names=None,
        upath_kwargs=None,
        **kwargs,
    ):
        self.chunksize = chunksize
        self.batch_readahead = batch_readahead
        self.fragment_readahead = fragment_readahead
        self.use_threads = use_threads
        self.column_names = column_names
        self.upath_kwargs = upath_kwargs
        self.kwargs = kwargs

    def read(self, input_file, read_columns=None):
        columns = read_columns or self.column_names
        file_names = self.read_index_file(
            input_file=input_file, upath_kwargs=self.upath_kwargs, **self.kwargs
        )
        (_, input_dataset) = file_io.read_parquet_dataset(file_names, **self.kwargs)

        batches, nrows = [], 0
        for batch in input_dataset.to_batches(
            batch_size=self.chunksize,
            batch_readahead=self.batch_readahead,
            fragment_readahead=self.fragment_readahead,
            use_threads=self.use_threads,
            columns=columns,
        ):
            if nrows + batch.num_rows > self.chunksize:
                # We've hit the chunksize so load to a DataFrame and yield.
                # There should always be at least one batch in here since batch_size == self.chunksize above.
                yield pyarrow.Table.from_batches(batches).to_pandas()
                batches, nrows = [], 0

            batches.append(batch)
            nrows += batch.num_rows

        if len(batches) > 0:
            yield pyarrow.Table.from_batches(batches).to_pandas()
