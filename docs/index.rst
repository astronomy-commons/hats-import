HATS Import
========================================================================================

HATS is a directory structure and metadata for spatially arranging large catalog survey data. 
This was originally motivated by a desire to perform spatial cross-matching between surveys 
at large scale, but is applicable to a range of spatial analysis and algorithms.

.. attention::

    If you already have catalog in HATS format
    and are just looking to ingest it or analyze it, you can look instead to ``LSDB``, 
    a python tool for scalable analysis of large catalogs (e.g. querying 
    and crossmatching ~10⁹ sources). It aims to address large-scale data processing 
    challenges, in particular those brought up by `LSST <https://www.lsst.org/about>`_.

The ``hats-import`` package provides purpose-built map-reduce pipelines for converting
large or custom catalogs into HATS format. 

.. toctree::
   :maxdepth: 1
   :caption: Using HATS import

   Home page <self>
   Getting Started <getting_started>
   guide/contact
   About & Citation <citation>

.. toctree::
   :maxdepth: 1
   :caption: Typical Catalog Creation

   catalogs/arguments
   catalogs/temp_files
   catalogs/properties
   catalogs/collections

.. toctree::
   :maxdepth: 1
   :caption: Catalog Customization

   catalogs/public/index
   guide/verification
   guide/reimporting
   guide/hipscat_conversion
   Notebooks <notebooks>
   guide/dask_on_ray

.. toctree::
   :maxdepth: 1
   :caption: Other Datasets

   guide/margin_cache
   guide/index_table   

.. toctree::
   :maxdepth: 1
   :caption: Developers

   guide/contributing
   API Reference <reference>