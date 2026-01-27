Zubercal
===============================================================================

Getting the data
-------------------------------------------------------------------------------

See docs at CalTech.

http://atua.caltech.edu/ZTF/Fields/ReadMe.txt


Challenges with this data set
-------------------------------------------------------------------------------

- The ``__index_level_0__`` pandas index should be ignored when reading.

  - it is identical to the ``objectid`` column, and is just bloat

  - it is non-unique, and that makes it tricky when splitting the file up

- The files are written out by band, and the band is included in the file
  name (but not as a field in the resulting data product). We use a 
  regular expression to parse out the band and insert it as a column in
  the dataframe.
- The parquet files are all a fine size for input files, so we don't mess
  with another chunk size.
- There are over 500 thousand data files, so we pass each **directory** to 
  the map job workers, and loop over all files in the directory ourselves.
- You may want to remove the input file, ``F0065/ztf_0065_1990_g.parquet``, 
  as this file seems to have corrupted snappy compression. All other files
  are readable as downloaded from caltech.

.. code-block:: python

    import hats_import.pipeline as runner
    from hats_import.catalog.arguments import ImportArguments
    from hats_import.catalog.file_readers import ParquetPandasReader
    import pyarrow.parquet as pq
    import pyarrow as pa
    import re
    import glob


    class ZubercalParquetReader(ParquetPandasReader):
        def read(self, input_file):
            """Reader for the specifics of zubercal parquet files."""
            columns = [
                "mjd",
                "mag",
                "objdec",
                "objra",
                "magerr",
                "objectid",
                "info",
                "flag",
                "rcidin",
                "fieldid",
            ]

            ## Find all the parquet files in the directory
            files = glob.glob(f"{input_file}/**.parquet")
            files.sort()

            for file in files:
                ## Parse the band from the file name, and hold onto it for later.
                match = re.match(r".*ztf_[\d]+_[\d]+_([gir]).parquet", str(file))
                band = match.group(1)
                parquet_file = pq.read_table(file, columns=columns, **self.kwargs)
                for smaller_table in parquet_file.to_batches(max_chunksize=self.chunksize):
                    frame = pa.Table.from_batches([smaller_table]).to_pandas()
                    frame["band"] = band
                    yield frame


    files = glob.glob("/path/to/downloads/F**/")
    files.sort()

    args = ImportArguments(
        output_artifact_name="zubercal",
        input_file_list=files,
        ## NB - you need the parens here!
        file_reader=ZubercalParquetReader(),
        file_reader="parquet",
        catalog_type="source",
        ra_column="objra",
        dec_column="objdec",
        sort_columns="objectid",
        highest_healpix_order=10,
        pixel_threshold=20_000_000,
        output_path="/path/to/catalogs/",
    )

    runner.pipeline(args)

Our performance
-------------------------------------------------------------------------------

Running on dedicated hardware, with only 5 workers, this import took around
two weeks.

- Mapping stage: around 15 hours
- Splitting stage: around 243 hours
- Reducing stage: around 70 hours
