Catalog Collections
===============================================================================

What's a catalog collection?
-------------------------------------------------------------------------------

Catalog collections are a particular directory structure that provide explicit
linking with a primary object catalog and the supplemental tables that 
support users in their analysis of those catalogs.

Below is an example of a collection directory structure:

.. code-block::
    :class: no-copybutton

    gaia_dr3 /
    |-- collection.properties
    |-- gaia_dr3 /
    | | -- hats.properties
    | + -- . . .
    |-- gaia_dr3_5arcs /
    | | -- hats.properties
    | + -- . . .
    |-- gaia_dr3_100arcs /
    | | -- hats.properties
    | + -- . . .
    +-- gaia_dr3_designation /
        |-- hats.properties
        + -- . . .

Here, the primary object catalog is named simply "gaia_dr3". There are two margin
tables (with widths of 5 and 100 arcseconds), as you may want to have different 
margins for different scientific use cases. The last table, "gaia_dr3_designation",
is an index table to enable fast lookup of objects by their "designation", which 
is not a spatial property.

The ``collection.properties`` file provides the links between those tables.

Below is an example of what the properties file would look like for the above directory 
structure:

.. code-block::
    :class: no-copybutton

    # HATS Collection
    obs_collection = gaia_dr3
    hats_primary_table_url = catalog
    all_margins = gaia_dr3_5arcs gaia_dr3_100arcs
    default_margin = gaia_dr3_5arcs
    all_indexes = designation gaia_dr3_designation

The collection pipeline is intended to be a single pipeline that will create the object table,
requested margins and indexes, and the properties file to link them all together.

If you have existing catalogs and tables, this pipeline can be used to create the properties
file, and any tables that haven't been created yet.

Pipeline setup
-------------------------------------------------------------------------------

The arguments for a collection pipeline use a builder pattern, meaning we will
construct a custom pipeline, step by step.

.. code-block:: python

    from hats_import import CollectionArguments

    args = (
        CollectionArguments(
            output_artifact_name="/path/to/hats/catalogs/",
            output_path=tmp_path,
            ## dask arguments go here
            ## progress reporting arguments go here
        )
        .catalog(
            input_path=small_sky_source_dir,
            file_reader="csv",
            ra_column="ra",
            dec_column="dec",
            sort_columns="source_id",
        )
        .add_margin(margin_threshold=5.0, is_default=True)
        .add_margin(margin_threshold=100.0)
        .add_index(indexing_column="designation", include_healpix_29=False)
    )

Overall collection arguments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First, you should specify the root directory for all of the collection
to live in.

Any arguments dealing with the creation of the dask client, resume behavior,
or progress reporting can be specified at the initialization of the ``CollectionArguments`` 
object, and they will be carried through to all of the other parts of the pipeline.

Catalog arguments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You MUST specify a SINGLE catalog to serve as the object catalog for your collection.

You do this by calling ``.catalog()`` on the ``CollectionArguments`` object.

Every argument that can be specified in the :doc:`ImportArguments <arguments>` can be used 
with this method, so you have the same level of flexibility over the creation of your 
dataset here as you do in the primary catalog creation pipeline.

**If your catalog already exists**, that's ok! You can get away with just specifying
the ``catalog_path`` argument, and the pipeline will take care of the rest.

Margin arguments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can specify as many margins as you'd like for your catalog collection.
Or none!

You do this by calling ``.add_margin()``  on the ``CollectionArguments`` object
for each margin you would like to create.

Every argument that can be specified in the :doc:`MarginCacheArguments </guide/margin_cache>`
can be used with this method, so you have the same level of flexibility over the creation 
of your dataset here as you do in the traditional margin cache creation pipeline.

Additionally, you can add the ``is_default`` argument to *at most* one of the margins. 
If a margin is set as the default, then analysis tools like LSDB can use that margin
for cross-matching, if none other is specified.

For convenience, you are allowed to specify *fewer* arguments than the traditional
margin cache pipeline. You MUST specify the margin distance, either in ``margin_threshold``
or ``margin_order``. If you don't specify a name for the margin catalog, we will construct
it based on the primary catalog name and the margin distance.


Index arguments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can specify as many indexes as you'd like for your catalog collection.
Or none!

You do this by calling ``.add_index()``  on the ``CollectionArguments`` object
for each index you would like to create.

Every argument that can be specified in the :doc:`IndexArguments </guide/index_table>`
can be used with this method, so you have the same level of flexibility over the creation 
of your dataset here as you do in the tradition index table creation pipeline.

For convenience, you are allowed to specify *fewer* arguments than the traditional
margin cache pipeline. You MUST specify the ``indexing_column``. If you don't specify a name 
for the index table, we will construct it based on the primary catalog name and the indexing column.
