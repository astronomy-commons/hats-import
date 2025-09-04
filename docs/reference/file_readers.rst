File Readers
===========================

In the catalog import pipeline, we use ``InputReader`` objects to iterate through
the input files. This allows us to take in a variety of file formats, without having
to re-write the entire processing pipeline for each kind of file format we encounter.

Your own read method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you need to write your own input file reader, it should subclass ``InputReader``
and must provide a ``read`` method.

On the first pass through the data, we just need to read the RA and Dec columns, and so
the ``read_columns`` value will be a list with just these two columns. You can use this 
information to read less data, if possible given the file format.

You can either yield a ``pandas.DataFrame`` object or a ``pyarrow.Table``. If yield
a pandas dataframe, note that further pipeline stages will convert into a ``pyarrow.Table`` 
anyway.

You must yield the result, and should utlize chunking to avoid memory issues.

Index Readers
~~~~~~~~~~~~~~~~~~~~~~~~~~

If you have many small files (think 400k+ CSV files with a few rows each), you
may benefit from "index" file readers. These allow you to explicitly create 
batches for tasks by providing a set of index files, where each file is a 
text file that contains only paths to data files.

Benefits:

1. If you have 400k+ input files, you don't want to create 400k+ dask tasks
   to process these files.
2. If the files are very small, batching them in this way allows the import 
   process to *combine* several small files into a single chunk for processing.
   This will result in fewer intermediate files during the ``splitting`` stage.
3. If you have parquet files over a slow networked file system, we support
   pyarrow's readahead protocol through index readers.

Warnings:

1. If you have 20 dask workers in your pool, you may be tempted to create 
   20 index files. This is not always an efficient use of resources! 
   You'd be better served by 200 index files, so that:

   a. dask can spread the load if some lists of files take longer to process
      than others
   b. if the pipeline dies after successfully processing 15 lists, when you 
      retry the pipeline, you'll only be processing 5 lists with those same 20 
      workers and many workers will be sitting idle.


The ``read_index_file`` method is provided as a convenience to read these 
index files, and returns a list of paths to be read by your method.

.. code-block:: python

    def read(self, input_file, read_columns=None):
        file_paths = self.read_index_file(
            input_file=input_file, upath_kwargs=self.upath_kwargs, **self.kwargs
        )

.. currentmodule:: hats_import.catalog.file_readers

Built-in Classes and Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api/

    get_file_reader
    InputReader
    CsvReader
    CsvPyarrowReader
    IndexedCsvReader
    AstropyEcsvReader
    FitsReader
    ParquetPyarrowReader
    ParquetReader
    IndexedParquetReader