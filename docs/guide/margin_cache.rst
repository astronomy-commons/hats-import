Margin Cache
===============================================================================

For more discussion of the whys and hows of margin caches, please see the 
`demo notebook in LSDB documentation <https://docs.lsdb.io/en/latest/tutorials/margins.html>`__ and
`Max's AAS iPoster <https://aas242-aas.ipostersessions.com/?s=66-E9-54-B6-6B-C3-4B-47-79-24-44-5A-13-25-82-E7>`__
for more information.

This page discusses topics around setting up a pipeline to generate a margin
cache from an existing HATS catalog on disk.

At a minimum, you need arguments that include where to find the input files,
and where to put the output files. A minimal arguments block will look something like:

.. code-block:: python

    from hats_import.margin_cache.margin_cache_arguments import MarginCacheArguments

    args = MarginCacheArguments(
        input_catalog_path="./my_data/my_catalog",
        output_path="./output",
        margin_threshold=10.0,
        output_artifact_name="my_catalog_10arcs",
    )
    

More details on each of these parameters is provided in sections below.

For the curious, see the API documentation for 
:py:class:`hats_import.margin_cache.margin_cache_arguments.MarginCacheArguments`.

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

    args = MarginCacheArguments(...)
    with Client('scheduler:port') as client:
        pipeline_with_client(args, client)

If you're running within a ``.py`` file, we recommend you use a ``main`` guard to
potentially avoid some python threading issues with dask:

.. code-block:: python

    from hats_import.pipeline import pipeline

    def margin_pipeline():
        args = MarginCacheArguments(...)
        pipeline(args)

    if __name__ == '__main__':
        margin_pipeline()

Input Catalog
-------------------------------------------------------------------------------

For this pipeline, you will need to have already transformed your catalog into 
hats parquet format. Provide the path to the catalog data with the argument
``input_catalog_path``.

The input hats catalog will provide its own right ascension and declination
that will be used when computing margin populations.

Margin calculation parameters
-------------------------------------------------------------------------------

When creating a margin catalog, we need to know how large of a margin to include
around each pixel in the input catalog.

``margin_threshold`` is the size of the margin cache boundary, given in arcseconds.
This defaults to 5 arcseconds, but you should set this value to whatever is
appropriate for the astrometry error/PSF width for your instruments. If you're
not sure how to determine this, please reach out! We'd love to help! :doc:`/guide/contact`.

This is equivalent to setting the ``margin_order``. We use a lookup, with roughly
the following table of values. This is the minimum separation angle possible within 
healpix pixels of a given order.

=================  =========================
``margin_order``   minimum separation angle 
=================  =========================
10                 2.15 arcmin
11                 1.07 arcmin
12                 32.21 arcsec
13                 16.10 arcsec
14                 8.05 arcsec
15                 4.03 arcsec
16                 2.01 arcsec
17                 1.01 arcsec
18                 0.50 arcsec
19                 0.25 arcsec
20                 0.13 arcsec
21                 62.91 msec
22                 31.45 msec
=================  =========================


For each input catalog partition, we can quickly determine all possible neighboring
healpix pixels at the given ``margin_order``. All of these partitions *may* contain 
points that are inside the ``margin_threshold``. For each point in the input catalog, 
we can quickly determine the healpix pixel at ``margin_order`` and filter points 
based on this.

In the figure below, the central yellow pixel is the primary catalog pixel at order 10,
and the surrounding pink order 13 pixels represent the margin for 10 arcsec.

.. figure:: /static/margin_pixels.png
   :class: no-scaled-link
   :align: center
   :alt: Visual of primary catalog pixel and the margin pixels.

   Visual of primary catalog pixel and the margin pixels.


For reasons of runtime performance and numerical precision, we do not perform precise
boundary checking on individual points.

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

You must specify a name for the margin catalog, using ``output_artifact_name``.
A good convention is the name of the primary input catalog, followed by the
margin threshold, e.g. ``gaia_10arcs`` would be a margin catalog based on gaia
that uses 10 arcseconds for margins.

You must specify where you want your margin data to be written, using
``output_path``. This path should be the base directory for your catalogs, as 
the full path for the margin will take the form of ``output_path/output_artifact_name``.

If you're writing to cloud storage, or otherwise have some filesystem credential
dict, initialize ``output_path`` using ``universal_pathlib``'s utilities.

In addition, you can specify directories to use for various intermediate files:

- dask worker space (``dask_tmp``)
- sharded parquet files (``tmp_dir``)

Most users are going to be ok with simply setting the ``tmp_dir`` for all intermediate
file use. For more information on these parameters, when you would use each, 
and demonstrations of temporary file use see :doc:`/catalogs/temp_files`
