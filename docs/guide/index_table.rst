Index Table
===============================================================================

This page discusses topics around setting up a pipeline to generate a secondary
index lookup for a field on an existing hats catalog on disk.

This is useful if you would like to have quick access to rows of your table using
a survey-provided unique identifier that is NOT spatially correlated. To find 
the row, you would have to perform a full scan of all partitions. A secondary
index performs this full scan and shuffle and writes out a simple mapping from 
the unique identifier to where it can be found in the sky. 

At a minimum, you need arguments that include where to find the input files,
and where to put the output files. A minimal arguments block will look something like:

.. code-block:: python

    from hats_import.index.arguments import IndexArguments

    args = IndexArguments(
        input_catalog_path="./my_data/my_catalog",
        indexing_column="target_id",
        output_path="./output",
        output_artifact_name="my_catalog_target_id",
    )

More details on each of these parameters is provided in sections below.

For the curious, see the API documentation for 
:py:class:`hats_import.index.arguments.IndexArguments`.

Dask setup
-------------------------------------------------------------------------------

We will either use a user-provided dask ``Client``, or create a new one with
arguments:

``dask_tmp`` - ``str`` - directory for dask worker space. this should be local to
the execution of the pipeline, for speed of reads and writes. For much more 
information, see :doc:`/catalogs/temp_files`

``dask_n_workers`` - ``int`` - number of workers for the dask client. Defaults to 1.

``dask_threads_per_worker`` - ``int`` - number of threads per dask worker. Defaults to 1.

If you find that you need additional parameters for your dask client (e.g are creating
a SLURM worker pool), you can instead create your own dask client and pass along 
to the pipeline, ignoring the above arguments. This would look like:

.. code-block:: python

    from dask.distributed import Client
    from hats_import.pipeline import pipeline_with_client

    args = IndexArguments(...)
    with Client('scheduler:port') as client:
        pipeline_with_client(args, client)

If you're running within a ``.py`` file, we recommend you use a ``main`` guard to
potentially avoid some python threading issues with dask:

.. code-block:: python

    from hats_import.pipeline import pipeline

    def index_pipeline():
        args = IndexArguments(...)
        pipeline(args)

    if __name__ == '__main__':
        index_pipeline()

Input Catalog
-------------------------------------------------------------------------------

For this pipeline, you will need to have already transformed your catalog into 
hats parquet format. Provide the path to the catalog data with the argument
``input_catalog_path``.

``indexing_column`` is required, and is the column that you would like to create
and index and sort over.

``extra_columns`` will also be stored alongside the ``indexing_column``. This 
can be useful if you will need to perform some lookups frequently, and you 
can just perform a single read.


Divisions
-------------------------------------------------------------------------------

In generating an index over a catalog column, we use dask's ``set_index``
method to shuffle the catalog data around. This can be a very expensive operation. 
We can save a lot of time and general compute resources if we have some intelligent 
prior information about the distribution of the values inside the column we're 
building an index on.

Roughly speaking, the index table will have some even buckets of values for 
the ``indexing_column``. The ``division_hints`` argument provides a reasonable
prior for starting values for those histogram bins.

Note that these will NOT necessarily be the divisions that the data is 
partitioned along.

.. note:: 
    Use a python list

    It's important to dask that the divisions be a list, and not a numpy array,
    and don't forget to append the maximum value as an extra division at the end.


String IDs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Gaia DR3 uses a string identifier for its catalog and we show how to create
sample divisions for that data.

We can create these divisions with just the **prefixes** of strings, and 
string sorting will be smart enough to collate the various strings appropriately.

.. code-block:: python

    divisions = [f"Gaia DR3 {i}" for i in range(10_000, 99_999, 12)]
    divisions.append("Gaia DR3 999999988604363776")

Getting hints from ``_metadata``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note:: 
    Don't panic!

    This is totally optional, and just provided here for reference if you
    really aren't sure how to provide some division priors.

Parquet's ``_metadata`` file provides some high-level statistics about its columns,
which includes the minimum and maximum value within individual parquet files.
By reading just the ``_metadata`` file, we can construct a reasonable set 
of divisions.

First, find the minimum and maximum values across all of our data. We do this
just by looking inside that _metadata file - we don't need to do a full 
catalog scan for these high-level statistics!

Then use those values, and a little arithmetic to create a **list** of divisions 
(it's important to dask that this be a list, and not a numpy array). Pass this 
list along to your ``ImportArguments``!

.. code-block:: python

    import numpy as np
    import os
    from hats.io.parquet_metadata import write_parquet_metadata
    from hats.io import file_io

    ## Specify the catalog and column you're making your index over.
    input_catalog_path="/data/input_catalog"
    indexing_column="target_id"

    ## you might not need to change anything after that.
    catalog = hats.read_hats(input_catalog_path)
    all_stats = catalog.aggregate_column_statistics()

    global_min = all_stats.at[indexing_column, "min_value"]
    global_max = all_stats.at[indexing_column, "max_value"]
    num_partitions = len(catalog.get_healpix_pixels())

    print("global min", global_min)
    print("global max", global_max)

    increment = int((global_max-global_min)/num_partitions)

    divisions = np.append(np.arange(start = global_min, stop = global_max, step = increment), global_max)
    divisions = divisions.tolist()


Progress Reporting
-------------------------------------------------------------------------------

By default, we will display some progress bars during pipeline execution. To 
disable these (e.g. when you expect no output to standard out), you can set
``progress_bar=False``.

There are several stages to the pipeline execution, and you can expect progress
reporting to look like the following:

.. code-block::
    :class: no-copybutton

    Mapping  : 100%|██████████| 2352/2352 [9:25:00<00:00, 14.41s/it]
    Reducing : 100%|██████████| 2385/2385 [00:43<00:00, 54.47it/s] 
    Finishing: 100%|██████████| 4/4 [00:03<00:00,  1.15it/s]

For very long-running pipelines (e.g. multi-TB inputs), you can get an 
email notification when the pipeline completes using the 
``completion_email_address`` argument. This will send a brief email, 
for either pipeline success or failure.

Output
-------------------------------------------------------------------------------

Where?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You must specify a name for the index table, using ``output_artifact_name``.
A good convention is the name of the primary input catalog, followed by the
index column threshold, e.g. ``gaia_designation`` would be an index table
based on gaia that indexes over the ``designation`` field.

You must specify where you want your index table to be written, using
``output_path``. This path should be the base directory for your catalogs, as 
the full path for the index will take the form of ``output_path/output_artifact_name``.

If you're writing to cloud storage, or otherwise have some filesystem credential
dict, initialize ``output_path`` using ``universal_pathlib``'s utilities.

In addition, you can specify directories to use for various intermediate files:

- dask worker space (``dask_tmp``)
- sharded parquet files (``tmp_dir``)

Most users are going to be ok with simply setting the ``tmp_dir`` for all intermediate
file use. For more information on these parameters, when you would use each, 
and demonstrations of temporary file use see :doc:`/catalogs/temp_files`

How?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You may want to tweak parameters of the final index output, and we have helper 
arguments for a few of those.

``compute_partition_size`` - ``int`` - partition size used when 
computing the leaf parquet files.

``include_healpix_29`` - ``bool`` - whether or not to include the 64-bit
hats spatial index in the index table. Defaults to ``True``. 

``include_order_pixel`` - ``bool`` - whether to include partitioning columns, 
``Norder``, ``Dir``, and ``Npix``. You probably want to keep these!
Defaults to ``True``. If you change this, there might be unexpected behavior
when trying to use the index table.

``drop_duplicates`` - ``bool`` - drop duplicate occurrences of all fields
that are included in the index table. This is enabled by default, but can be
**very** slow. This has an interaction with the above ``include_healpix_29``
and ``include_order_pixel`` options above. We desribe some common patterns below:

- I want to create an index over the target ID in my catalog. There are no
  lightcurves in my data and it is a flat catalog.

    .. code-block:: python

        indexing_column="target_id",
        include_healpix_29=False,
        # I want to know where my data is in the sky.
        include_order_pixel=True,
        # target_id is unique, and I don't need to do extra work to de-duplicate
        drop_duplicates=False,

- I have a catalog of light curve data. there is a unique ``detection_id``
  and light curves are grouped by the ``target_id``. I want to create an 
  index over the ``target_id`` to quickly get a light curve for a target.
  I want one row in my index for each partition with a given ``target_id``

    .. code-block:: python

        indexing_column="target_id",
        # target_id is NOT unique
        drop_duplicates=True,
        # including the _healpix_29 will bloat results
        include_healpix_29=False,
        # I want to know where my data is in the sky.
        include_order_pixel=True,
