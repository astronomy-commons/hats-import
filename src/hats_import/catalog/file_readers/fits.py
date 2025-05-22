from astropy.table import Table

from hats_import.catalog.file_readers.input_reader import InputReader


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
        input_file = self.regular_file_exists(input_file, **self.kwargs)
        with input_file.open("rb") as file_handle:
            table = Table.read(file_handle, memmap=True, **self.kwargs)
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
