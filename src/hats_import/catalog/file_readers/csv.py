import pandas as pd
import pyarrow as pa
from hats.io import file_io
from pyarrow import csv

from hats_import.catalog.file_readers.input_reader import InputReader


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
        input_file = self.regular_file_exists(input_file, **self.kwargs)

        if read_columns:
            self.kwargs["usecols"] = read_columns

        return file_io.load_csv_to_pandas_generator(
            input_file,
            chunksize=self.chunksize,
            header=self.header,
            **self.kwargs,
        )


class CsvPyarrowReader(InputReader):
    """CSV reader that uses the pyarrow library for reading.

    This *can* be faster than the pandas reader, and *can* have different handling of
    data types.

    Attributes:
        chunksize (int): number of BYTES of the file to process at once.
            This is different from chunksize seen in other readers!!
            For large files, this can prevent loading the entire file
            into memory at once.
        column_names (list[str] or None): Names of columns to use from the input dataset.
            If None, use all columns.
        schema_file (str): path to a parquet schema file. if provided, column names and types
            will match those of the schema.
        read_options (csv.ReadOptions): options for reading CSV files using pyarrow.
            We will set the ``block_size`` argument based on the value for ``chunksize``.
            See https://arrow.apache.org/docs/python/generated/pyarrow.csv.ReadOptions.html
        convert_options (csv.ConvertOptions): options for converting CSV data to pyarrow Table.
            We will pass the pyarrow schema from ``schema_file`` to the ``column_types`` property.
            See https://arrow.apache.org/docs/python/generated/pyarrow.csv.ConvertOptions.html
        kwargs: arguments to pass along to pyarrow.parquet.ParquetFile.
            See https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetFile.html
    """

    def __init__(
        self,
        *,
        chunksize=10 * 1024**2,
        compression=None,
        column_names=None,
        schema_file=None,
        read_options=None,
        convert_options=None,
        **kwargs,
    ):
        self.kwargs = kwargs
        self.column_names = column_names
        self.read_options = read_options or csv.ReadOptions(block_size=chunksize)
        self.convert_options = convert_options or csv.ConvertOptions()
        self.compression = compression
        if schema_file:
            schema = file_io.read_parquet_metadata(schema_file).schema.to_arrow_schema()
            if not self.convert_options.column_types:
                self.convert_options.column_types = schema
            if not self.read_options.column_names:
                self.read_options.column_names = schema.names
                # Set skip_rows here to skip parsing the header row with column names.
                # See https://arrow.apache.org/docs/python/generated/pyarrow.csv.ReadOptions.html
                self.read_options.skip_rows = 1

    def read(self, input_file, read_columns=None):
        input_file = self.regular_file_exists(input_file, **self.kwargs)

        # If we only want to read some columns (e.g. radec), we must specify
        # in the convert_options argument
        read_columns = read_columns or self.column_names
        convert_options = self.convert_options
        if read_columns:
            convert_options.include_columns = read_columns
        with input_file.open(mode="rb", compression=self.compression) as file_handle:
            with csv.open_csv(
                file_handle, convert_options=convert_options, read_options=self.read_options, **self.kwargs
            ) as reader:
                for next_chunk in reader:
                    table = pa.Table.from_batches([next_chunk])
                    table = table.replace_schema_metadata()
                    yield table


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
