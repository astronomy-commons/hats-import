import pyarrow as pa
from hats.io import file_io

from hats_import.catalog.file_readers.input_reader import InputReader


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
        input_file = self.regular_file_exists(input_file, **self.kwargs)
        columns = read_columns or self.column_names
        parquet_file = file_io.read_parquet_file(input_file, **self.kwargs)
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
        input_file = self.regular_file_exists(input_file, **self.kwargs)
        columns = read_columns or self.column_names
        parquet_file = file_io.read_parquet_file(input_file, **self.kwargs)
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
                yield pa.Table.from_batches(batches).replace_schema_metadata()
                batches, nrows = [], 0

            batches.append(batch)
            nrows += batch.num_rows

        if len(batches) > 0:
            yield pa.Table.from_batches(batches).replace_schema_metadata()
