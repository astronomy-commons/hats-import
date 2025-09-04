"""File reading generators for common file types."""

from .csv import CsvPyarrowReader, CsvReader, IndexedCsvReader
from .ecsv import AstropyEcsvReader
from .fits import FitsReader
from .input_reader import InputReader
from .parquet import IndexedParquetReader, ParquetPyarrowReader, ParquetReader


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

    Currently supported formats include:

    - ``"csv"``, comma separated values. may also be tab- or pipe-delimited
      includes `.csv.gz` and other compressed csv files
    - ``"fits"``, flexible image transport system. often used for astropy tables.
    - ``"parquet"``, compressed columnar data format
    - ``"ecsv"``, astropy's enhanced CSV
    - ``"indexed_csv"``, "index" style reader, that accepts a file with a list
      of csv files that are appended in-memory
    - ``"indexed_parquet"``, "index" style reader, that accepts a file with a list
      of parquet files that are appended in-memory

    Args:
        file_format (str): specifier for the file type and extension.
            If using an ``input_path`` argument, we will look for files with this string
            as the extension.
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
