Catalog Import Arguments
===============================================================================

This page discusses a few topics around setting up a catalog pipeline.

Once your catalog has been imported, you can verify and inspect the metadata, 
either with the :doc:`Verification Pipeline </guide/verification>`, or methods
described in the `Manual catalog verification notebook <https://docs.lsdb.io/en/stable/tutorials/pre_executed/manual_verification.html>`__

At a minimum, you need arguments that include where to find the input files,
the column names for RA, and DEC, and where to put the output files. 
A minimal arguments block will look something like:

.. code-block:: python

    from hats_import.catalog.arguments import ImportArguments

    args = ImportArguments(
        sort_columns="ObjectID",
        ra_column="ObjectRA",
        dec_column="ObjectDec",
        input_path="./my_data",
        file_reader="csv",
        output_artifact_name="test_cat",
        output_path="./output",
    )


More details on each of these parameters is provided in sections below.

For the curious, see the API documentation for 
:py:class:`hats_import.catalog.arguments.ImportArguments`.

Pipeline setup
-------------------------------------------------------------------------------

Dask
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We will either use a user-provided dask ``Client``, or create a new one with
arguments:

``dask_tmp`` - ``str`` - directory for dask worker space. this should be local to
the execution of the pipeline, for speed of reads and writes. For much more 
information, see :doc:`temp_files`.

``dask_n_workers`` - ``int`` - number of workers for the dask client. Defaults to 1.

``dask_threads_per_worker`` - ``int`` - number of threads per dask worker. Defaults to 1.

If you find that you need additional parameters for your dask client (e.g are creating
a SLURM worker pool), you can instead create your own dask client and pass along 
to the pipeline, ignoring the above arguments. This would look like:

.. code-block:: python

    from dask.distributed import Client
    from hats_import.pipeline import pipeline_with_client

    args = ...  # ImportArguments()
    with Client('scheduler:port') as client:
        pipeline_with_client(args, client)

If you're running within a ``.py`` file, we recommend you use a ``main`` guard to
potentially avoid some python threading issues with dask:

.. code-block:: python

    from hats_import.pipeline import pipeline

    def import_pipeline():
        args = ...
        pipeline(args)

    if __name__ == '__main__':
        import_pipeline()

Resuming
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The import pipeline has the potential to be a very long-running process, if 
you're importing large amounts of data, or performing complex transformations
on the data before writing.

While the pipeline runs, we take notes of our progress so that the pipeline can
be resumed at a later time, if the job is pre-empted or canceled for any reason.

When instantiating a pipeline, you can use the ``resume`` flag to indicate that
we can resume from an earlier execution of the pipeline. By default, if any resume
files are found, we will restore the pipeline's previous progress.

If you want to start the pipeline from scratch you can simply set ``resume=False``.
Alternatively, go to the temp directory you've specified and remove any intermediate
files created by the previous runs of the ``hats-import`` pipeline. You should also
remove the output directory if it has any content. The resume argument performs these
cleaning operations automatically for you.

For more information about the kinds of files we write to enable this feature,
see :doc:`temp_files`.

By default, these notes files, and the intermediate parquet leaf files are removed
once they're no longer needed by the pipeline. If, for whatever reason, you would
like to retain these files, you can use flags:

- ``delete_resume_log_files=False`` will keep resume notes/logs files
- ``delete_intermediate_parquet_files=False`` will retain ALL of the intermediate 
  parquet leaf files.

Reading input files
-------------------------------------------------------------------------------

Catalog import reads through a list of files and converts them into a hats-sharded catalog.

If you already know the size of your catalog, and would like us to verify that 
the same number of rows exists at each checkpoint in the pipeline, pass ``expected_total_rows=<int>``.

Which files?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are a few ways to specify the files to read:

* ``input_path``: 
    will include all files in the indicated directory.
* ``input_file_list``: 
    a list of fully-specified paths you want to read.

    * this strategy can be useful to first run the import on a single input
      file and validate the input, then run again on the full input set, or 
      to debug a single input file with odd behavior. 
    * if you have a mix of files in your target directory, you can use a glob
      statement like the following to gather input files:

.. code-block:: python

    in_file_paths = glob.glob("/data/object_and_source/object**.csv")
    in_file_paths.sort()

.. important::
    We will create one Dask task per element inside the ``input_file_list``. 

    If you have only one large file, we can only read the file with one Dask worker.
    If you have lots of files, you might want to consider the Indexed batching
    strategy described below.

How to read them?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Specify an instance of ``InputReader`` for the ``file_reader`` parameter.

We use the ``InputReader`` class to read files in chunks and pass the chunks
along to the map/reduce stages. We've provided reference implementations for 
reading CSV, FITS, and Parquet input files, but you can subclass the reader 
type to suit whatever input files you've got.

You only need to provide an object ``file_reader`` argument if you are using a custom file reader
or passing parameters to the file reader. For example you might use ``file_reader=CsvReader(sep="\s+")``
to parse a whitespace separated file. Otherwise, you can use a short string to 
specify an existing file reader type e.g. ``file_reader="csv"``.

You can find the full API documentation for 
:py:class:`hats_import.catalog.file_readers.InputReader`

.. code-block:: python

    class StarrReader(InputReader):
        """Class for fictional Starr file format."""
        def __init__(self, chunksize=500_000, **kwargs):
            self.chunksize = chunksize
            self.kwargs = kwargs

        def read(self, input_file):
            starr_file = starr_io.read_table(input_file, **self.kwargs)
            for smaller_table in starr_file.to_batches(max_chunksize=self.chunksize):
                smaller_table = filter_nonsense(smaller_table)
                yield smaller_table.to_pandas()

    ...

    args = ImportArguments(
        ...
        ## Locates files like "/directory/to/files/**starr"
        input_path="/directory/to/files/",
        ## NB - you need the parens here!
        file_reader=StarrReader(),

    )

If you're reading from cloud storage, or otherwise have some filesystem credential
dict, initialize ``input_file`` using ``universal_pathlib``'s utilities.

Indexed batching strategy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you have many small files (think 400k+ CSV files with a few rows each), you
may benefit from "indexed" file readers. These allow you to explicitly create 
batches for tasks by providing a set of index files, where each file is a 
text file that contains only paths to data files.

Benefits:

1. If you have 400k+ input files, you don't want to create 400k+ dask tasks
   to process these files.
2. If the files are very small, batching them in this way allows the import 
   process to *combine* several small files into a single chunk for processing.
   This will result in fewer intermediate files during the ``splitting`` stage.
3. If you have parquet files over a slow networked file system, we support
   pyarrow's readahead protocol through indexed readers.

Warnings:

1. If you have 20 dask workers in your pool, you may be tempted to create 
   20 index files. This is not always an efficient use of resources! 
   You'd be better served by 200 index files, so that:

   a. dask can spread the load if some lists of files take longer to process
      than others
   b. if the pipeline dies after successfully processing 15 lists, when you 
      retry the pipeline, you'll only be processing 5 lists with those same 20 
      workers and many workers will be sitting idle.

Which fields?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Specify the ``ra_column`` and ``dec_column`` for the dataset.

There are two fields that we require in order to make a valid hats-sharded
catalog, the right ascension and declination. At this time, this is the only 
supported system for celestial coordinates.

If you're importing data that has previously been hats-sharded, you may use
``use_healpix_29 = True``. This will use that previously computed hats spatial
index as the position, instead of ra/dec.

Healpix order and thresholds
-------------------------------------------------------------------------------

When creating a new catalog through the hats-import process, we try to 
create partitions with approximately the same number of rows per partition. 
This isn't perfect, because the sky is uneven, but we still try to create 
smaller-area pixels in more dense areas, and larger-area pixels in less dense 
areas. 

We use the argument ``pixel_threshold`` and will split a partition into 
smaller healpix pixels until the number of rows is smaller than ``pixel_threshold``.
We will only split by healpix pixels up to the ``highest_healpix_order``. If we
would need to split further, we'll throw an error at the "Binning" stage, and you 
should adjust your parameters.

For more discussion of the ``pixel_threshold`` argument and a strategy for setting
this parameter, see notebook :doc:`/notebooks/estimate_pixel_threshold`

For more discussion of the "Binning" and all other stages, see :doc:`temp_files`.

Sparse Datasets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For sparse datasets you might want to force your catalog partitioning to avoid
partitions with very large area on they sky. 

Why? If you have sparse data that you know you will want to cross-match or join
to a catalog that is much denser, you may find yourself trying to match a large
(in terms of area on the sky) pixel to thousands of smaller pixels in the denser
catalog that occupy the same large region in the sky. Using more pixels of higher
order will have some inefficiencies in terms of on-disk storage, but will be 
easier to compute joins and cross-matches to large datasets.

There are a few strategies for tweaking the partitioning:

* **order range** - use the ``lowest_healpix_order`` argument, in addition
  to the ``highest_healpix_order``.
* **constant order** - use the ``constant_healpix_order`` argument. This will 
  **ignore** the ``pixel_threshold``, ``highest_healpix_order``, and 
  ``lowest_healpix_order`` arguments and the catalog will be partitioned by 
  healpix pixels at the ``constant_healpix_order``.
* **drop empty siblings** - by default, the ``drop_empty_siblings`` flag is
  set to ``True``, and will try to use smaller area pixels in sparse regions
  of the sky. This can make the catalog footprint closer to the actual survey
  footprint, but can be disabled if desired.

Progress Reporting
-------------------------------------------------------------------------------

By default, we will display some progress bars during pipeline execution. To 
disable these (e.g. when you expect no output to standard out), you can set
``progress_bar=False``.

There are several stages to the pipeline execution, and you can expect progress
reporting to look like the following:

.. code-block::
    :class: no-copybutton

    Mapping  : 100%|██████████| 72/72 [58:55:18<00:00, 2946.09s/it]
    Binning  : 100%|██████████| 1/1 [01:15<00:00, 75.16s/it]
    Splitting: 100%|██████████| 72/72 [72:50:03<00:00, 3641.71s/it]
    Reducing : 100%|██████████| 10895/10895 [7:46:07<00:00,  2.57s/it]
    Finishing: 100%|██████████| 6/6 [08:03<00:00, 80.65s/it]

We use ``tqdm`` to render these beautiful progress bars. ``tqdm`` will try to 
make a guess about the type of output to provide: plain
text as for a command line, or a pretty ipywidget. If it tries to use a pretty
widget but your execution environment can't support the widget, you can 
force the pipeline to use a simple progress bar with the ``simple_progress_bar``
argument. If you want to configure the progress bar any further, pass 
values to the ``tqdm_kwargs`` parameter. You can find more details on available
arguments at `the tqdm documentation <https://tqdm.github.io/docs/tqdm/>`__.

For very long-running pipelines (e.g. multi-TB inputs), you can get an 
email notification when the pipeline completes using the 
``completion_email_address`` argument. This will send a brief email, 
for either pipeline success or failure.

Output
-------------------------------------------------------------------------------

Where?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You must specify a name for the catalog, using ``output_artifact_name``.

You must specify where you want your catalog data to be written, using
``output_path``. This path should be the base directory for your catalogs, as 
the full path for the catalog will take the form of ``output_path/output_artifact_name``.

If you're writing to cloud storage, or otherwise have some filesystem credential
dict, initialize ``output_path`` using ``universal_pathlib``'s utilities.

In addition, you can specify directories to use for various intermediate files:

- dask worker space (``dask_tmp``)
- sharded parquet files (``tmp_dir``)
- intermediate resume files (``resume_tmp``)

Most users are going to be ok with simply setting the ``tmp_dir`` for all intermediate
file use. For more information on these parameters, when you would use each, 
and demonstrations of temporary file use see :doc:`temp_files`

How to write the parquet files?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You may want to tweak parameters of the final catalog output, and we have helper 
arguments for a few of those.

``write_table_kwargs`` - any additional arguments to customize the 
``pyarrow.parquet.ParquetWriter``. This can be used for alternative compression
strategies. By default we use ZSTD compression at level 15.

Parquet compression is also effected by the row groups. We can attempt to split 
the data partition into row group chunks according to a few strategies:

- **Number of rows** - Chunk according to integer number of rows in ``num_rows`` arg.

- **HEALPix subtiles** - Chunk according to finer spatial divisions within the HEALPix pixel of the data partition.
  This is controlled via the ``subtile_order_delta``. Reasonable values here are ``1``, ``2``, or ``3``.

.. hint::
    This behavior may be confusing, so let's talk about it a little bit more. 

    Say we're writing a data partition for a HEALPix order 6 tile.

    - With a ``1``, there are 4 maximum row groups, for each of the 4 subtiles 
      of HEALPix order 7.
    - With a ``2``, there are 16 maximum row groups, for each of the 16 subtiles 
      of HEALPix order 8.
    - With a ``3``, there are 64 maximum row groups, for each of the 64 subtiles 
      of HEALPix order 9.
    - With a value any higher, there will be so many row groups in each file that
      you will likely stop seeing the benefits of these divisions.

    Within **the same pipeline**, we would also write some partitions at HEALPix order 4,
    and they would have a similar number of row group divisions:

    - With a ``1``, there are 4 subtiles of HEALPix order 5.
    - With a ``2``, there are 16 subtiles of HEALPix order 6.
    - With a ``3``, there are 64 subtiles of HEALPix order 7.

    This is why we describe the subdivisions as a "delta", instead of an absolute HEALPix order.

If you're interested in other row group strategies, please reach out and we 
can consider adding them.

``add_healpix_29`` - ``bool`` - whether or not to add the hats spatial index
as a column in the resulting catalog. The ``_healpix_29`` field is designed to make many 
dask operations more performant, but if you do not intend to publish your dataset
and do not intend to use dask, then you can suppress generation of this column to
save a little space in your final disk usage.

The ``_healpix_29`` uses a high healpix order to create
values that can order all points in the sky, according to a nested healpix scheme.

``sort_columns`` - ``str`` - column for survey identifier, or other sortable column. 
If sorting by multiple columns, they should be comma-separated. 
If ``add_healpix_29=True``, ``_healpix_29`` will be the primary sort key, but the 
provided sorting will be used for any rows within the same higher-order pixel space.

``use_schema_file`` - ``str`` - path to a parquet file with schema metadata. 
This will be used for column metadata when writing the files, if specified.
For more information on why you would want this file and how to generate it,
check out our notebook :doc:`/notebooks/unequal_schema`.

``debug_stats_only`` - ``bool`` - If ``True``, we will not create the leaf
parquet files with the catalog data, and will only generate root-level metadata
files representing the full statistics of the final catalog. This can be useful
when probing the import process for effectiveness on processing a target dataset.

How to write the metadata files?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
``catalog_type`` - ``"object"`` or ``"source"``. Indicates the level of catalog data,
using the LSST nomenclature:

- object - things in the sky (e.g. stars, galaxies)
- source - detections of things in the sky at some point in time.
- map - non-point-source catalogs (e.g. dust maps)

Some data providers split detection-level data into a separate catalog, to make object
catalogs smaller, and reflects a relational data model.

``create_thumbnail`` - optionally create a ``dataset/data_thumbnail.parquet`` file, 
that will contain the first row of each data partition. This can be useful as a 
representative sample of the catalog data.

``should_write_skymap`` - defaults to ``True``, but you can disable writing 
a ``skymap.fits`` file with this argument. Further, to create several down-sampled
skymaps, pass a list of HEALPix orders to ``skymap_alt_orders`` (e.g.
``skymap_alt_orders=[2,4,6]``).

The HATS format allows for many additional key-values in the high-level ``hats.properties``
file. Many of these values are automatically set by the import process itself, but 
catalog providers may want to set additional fields for data provenance.

This full set of properties is outlined on a separate page (:doc:`properties`), 
but you can pass these key-value sets to the import process with the ``addl_hats_properties`` 
argument, and they will appear in the final ``hats.properties`` file:

.. code-block::

    addl_hats_properties={"hats_cols_default": "id, mjd", "obs_regime": "Optical"},

Incremental catalogs
-------------------------------------------------------------------------------

An "incremental catalog" is one that can be appended to by adding more files into
the leaf data partition directory.

We need to start with a catalog that has leaf directories instead of leaf parquet files.
You can import the starting data with additional arguments:

- ``npix_suffix="/"`` - this tells the pipeline to create a directory per-leaf.
- ``npix_parquet_name`` - this is the file name given to the first file in the 
  directory. By default, this will be ``Npix=M.parquet``.

To append to an incremental catalog, the pipeline should be run with the same
partitioning, potentially adding new pixels to fill in coverage gaps.
To make sure the same set of pixels are used, provide ``existing_pixels`` 
as a list of HEALPix ``(order, pixel)`` tuples.
