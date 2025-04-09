Reimporting HATS Catalogs
===============================================================================

This page discusses how to setup a pipeline to reimport a HATS catalog with different import arguments.

When importing a catalog to HATS, there are many parameters that can make the catalog better or worse suited for
different use cases. For example, if you're running analysis on a machine with more RAM or using fewer columns
from the dataset, you might want larger partitions to make the analysis faster. Or if you're trying to work
with HATS catalogs on smaller machines, you might want smaller partitions. Or depending on what type of
analysis you're running, you might want to sort the partitions by different columns to make filtering faster.

To reimport a catalog with different parameters, you can use the :meth:`reimport_from_hats() <hats_import.catalog.arguments.ImportArguments.reimport_from_hats>`
method to make an `ImportArguments` object to reimport a HATS catalog.

.. code-block:: python

    from dask.distributed import Client
    from hats_import.pipeline import pipeline_with_client
    from hats_import.catalog.arguments import ImportArguments

    args = ImportArguments.reimport_from_hats(
        "</path/to/hats>",
        "</path/to/output>",
        output_artifact_name="<output_catalog_name>",
        pixel_threshold=pixel_thresh,
    )

    with Client(n_workers=10, threads_per_worker=1, ...) as client:
        pipeline_with_client(args, client)


Any arguments which would normally be passed to :func:`ImportArguments <hats_import.catalog.arguments.ImportArguments>` can be passed to the method. Like
with any other arguments, this will output a catalog at `output_path/output_artifact_name`.

For example, to reimport a catalog at a constant HEALPix pixel with different sorting within each partition,
you could use:

.. code-block:: python

    args = ImportArguments.reimport_from_hats(
        "</path/to/hats>",
        "</path/to/output>",
        output_artifact_name="<output_catalog_name>",
        constant_healpix_order=5,
        sort_columns="ObjectID",
    )
