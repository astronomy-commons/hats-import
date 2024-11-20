Converting from HiPSCat
===============================================================================

This page discusses topics around setting up a pipeline to generate a new valid 
HATS catalog from an existing HiPSCat catalog. There are several breaking
incompatibilities between these two versions of the format, and users should
migrate catalogs to the new format before using HATS or LSDB later than v0.4.

At a minimum, you need arguments that include where to find the original files,
and where to put the output files. A minimal arguments block will look something like:

.. code-block:: python


    import hats_import.pipeline as runner
    from hats_import.hipscat_conversion.arguments import ConversionArguments

    args = ConversionArguments(
        input_catalog_path="./hipscat_catalogs/my_catalog",
        output_path="./hats_catalogs/",
        output_artifact_name="my_catalog",
    )
    runner.pipeline(args)

More details on each of these parameters is provided in sections below.

For the curious, see the API documentation for 
:py:class:`hats_import.hipscat_conversion.arguments.ConversionArguments`,
and its superclass :py:class:`hats_import.runtime_arguments.RuntimeArguments`.

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

    args = ConversionArguments(...)
    with Client('scheduler:port') as client:
        pipeline_with_client(args, client)

If you're running within a ``.py`` file, we recommend you use a ``main`` guard to
potentially avoid some python threading issues with dask:

.. code-block:: python

    from hats_import.pipeline import pipeline

    def conversion_pipeline():
        args = ConversionArguments(...)
        pipeline(args)

    if __name__ == '__main__':
        conversion_pipeline()

Input Catalog
-------------------------------------------------------------------------------

For this pipeline, you will need to have already transformed your catalog into 
HiPSCat parquet format. Provide the path to the catalog data with the argument
``input_catalog_path``.

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

You must specify a name for the new HATS table, using ``output_artifact_name``.
It's totally fine to simply use the name of the original input HiPSCat catalog,
so long as they will be written to different directories.

You must specify where you want your HATS table to be written, using
``output_path``. This path should be the base directory for your catalogs, as 
the full path for the HATS table will take the form of ``output_path/output_artifact_name``.

If there is already catalog data in the indicated directory, you can 
force new data to be written in the directory with the ``overwrite`` flag. It's
preferable to delete any existing contents, however, as this may cause 
unexpected side effects.

If you're writing to cloud storage, or otherwise have some filesystem credential
dict, initialize ``output_path`` using ``universal_pathlib``'s utilities.

In addition, you can specify directories to use for various intermediate files:

- dask worker space (``dask_tmp``)
- sharded parquet files (``tmp_dir``)

Most users are going to be ok with simply setting the ``tmp_dir`` for all intermediate
file use. For more information on these parameters, when you would use each, 
and demonstrations of temporary file use see :doc:`/catalogs/temp_files`

What next?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can validate that your new HATS catalog meets both the HATS/LSDB expectations,
as well as your own expectations of the data contents. You can follow along with the
`Manual catalog verification <https://docs.lsdb.io/en/stable/tutorials/manual_verification.html>`_.
