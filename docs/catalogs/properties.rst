
HATS properties
===============================================================================

The HATS format allows for many additional key-values in the high-level ``properties``
file. Many of these values are automatically set by the import process itself, but 
catalog providers may want to set additional fields for data provenance.

Properties from Arguments
-------------------------------------------------------------------------------

In some cases, the argument name was created before the HATS properties were settled,
so we have different names for arguments/properties. Below is a list of properties
that we will set automatically in the import process, based on the arguments provided.

.. list-table::
   :header-rows: 1

   * - **HATS Keyword**
     - **Argument**
     - **Format**
     - **Notes**
   * - ``obs_collection``
     - ``output_artifact_name``
     - one word (e.g. ``"2MASS"``) [#f1]_
     - Short name of original data set
   * - ``dataproduct_type``
     - ``catalog_type`` (defaults to ``"object"``)
     - one word (object, nested, margin, association, index, ...) [#f1]_
     - Variety of HATS catalog generated.
   * - ``hats_col_ra``
     - ``ra_column``
     - one word [#f1]_
     - Column name of the ra coordinate. Used for partitioning and default cross-matching.
   * - ``hats_col_dec``
     - ``dec_column``
     - one word [#f1]_
     - Column name of the dec coordinate. Used for partitioning and default cross-matching.
   * - ``hats_cols_sort``
     - ``sort_columns``
     - one or more words [#f2]_
     - At catalog creation time, the columns used to sort the data, in addition to `healpix_29` column.
   * - ``hats_max_rows``
     - ``pixel_threshold``
     - positive integer
     - At catalog creation time, the maximum number of rows per file before breaking into 4 new 
       files at higher order.

Constructed Properties
-------------------------------------------------------------------------------

Below is a list of properties that we will set automatically in the import process, 
based on the execution environment and output of the import pipeline.

.. list-table::
   :header-rows: 1

   * - **HATS Keyword**
     - **Format**
     - **Notes**
   * - ``hats_nrows``
     - positive integer
     - Number of rows of the HATS catalog
   * - ``hats_order``
     - positive integer
     - Deepest HATS order of catalog
   * - ``moc_sky_fraction``
     - real between 0 and 1
     - Fraction of the sky covered by the MOC associated with the catalog
   * - ``hats_estsize``
     - positive integer, with unit ``KB``
     - HATS on-disk size estimation
   * - ``hats_builder``
     - free text (e.g. ``"hats-import v0.6.4"``)
     - Name and version of the tool used for building the HATS catalog
   * - ``hats_creation_date``
     - ISO 8601 ``YYYY-mm-ddTHH:MMZ``
     - HATS first creation date
   * - ``hats_release_date``
     - ISO 8601 ``YYYY-mm-ddTHH:MMZ``
     - Last update to the HATS format
   * - ``hats_version``
     - semantic version (e.g. ``"0.1"``)
     - Number of HATS specification document version

User-provided Properties
-------------------------------------------------------------------------------

In addition, catalog providers can set provenance-related fields in the
``properties`` file by setting the ``addl_hats_properties`` argument on import. 
This will carry key-value pairs into the final ``properties`` file.

All of these fields are optional.

.. code-block:: python

    args = ImportArguments(
       ...
       addl_hats_properties={"hats_cols_default": "id, mjd", "obs_regime": "Optical"},
       ...
    )

.. list-table::
   :header-rows: 1

   * - **HATS Keyword**
     - **Format**
     - **Notes**
   * - ``hats_cols_default``
     - one or more words [#f2]_
     - Which columns should be read from parquet files, when user doesn't otherwise specify. 
       Useful for wide tables.
   * - ``hats_cols_survey_id``
     - one or more words [#f2]_
     - The primary key used in the original survey data. May be multiple columns if 
       the survey uses a composite key (e.g. ``object_id MJD`` for detections)
   * - ``hats_coordinate_epoch``
     - one word [#f1]_
     - For the default ra and dec (hats_col_ra, hats_col_dec), the measurement epoch
   * - ``hats_frame``
     - one word ("equatorial" (ICRS), "galactic", or "ecliptic") [#f1]_
     - Coordinate frame reference
   * - ``creator_did``
     - IVOID (e.g. ``ivo://CDS/P/2MASS/J``)
     - Unique ID of the HATS
   * - ``addendum_did``
     - IVOID
     - If content has been added after initial catalog creation, ``creator_did``
       of any added data
   * - ``bib_reference``
     - free text
     - Bibliographic reference
   * - ``bib_reference_url``
     - URL
     - URL to bibliographic reference
   * - ``data_ucd``
     - IVOA UCD
     - UCD describing data contents
   * - ``hats_copyright``
     - free text
     - Copyright mention associated to the HATS
   * - ``hats_creator``
     - free text
     - Institute or person who built the HATS
   * - ``hats_progenitor_url``
     - URL
     - URL to an associated progenitor HATS
   * - ``hats_service_url``
     - URL
     - HATS access url
   * - ``hats_status``
     - one or more words (("private" or "public"), ("main", "mirror", or "partial"), 
       ("clonable", "unclonable" or "clonableOnce")) [#f2]_
     - HATS catalog status
   * - ``obs_ack``
     - free text
     - Acknowledgment mention.
   * - ``obs_copyright``
     - free text
     - Copyright mention associated to the original data
   * - ``obs_copyright_url``
     - URL
     - URL to a copyright mention
   * - ``obs_description``
     - free text, longer free text description of the dataset
     - Data set description
   * - ``obs_regime``
     - one word ("Radio", "Millimeter", "Infrared", "Optical", "UV", "EUV", 
       "X-ray", "Gamma-ray") [#f1]_
     - General wavelength
   * - ``obs_title``
     - free text, one line. (e.g. "HST F110W observations")
     - Data set title
   * - ``prov_progenitor``
     - free text
     - Provenance of the original data
   * - ``publisher_id``
     - IVOID (e.g. "ivo://CDS")
     - Unique ID of the HATS publisher
   * - ``t_min``
     - real, MJD
     - Start time of the observations
   * - ``t_max``
     - real, MJD
     - Stop time of the observations

.. rubric:: Notes on "Words"

.. [#f1] "one word" implies a single name, consisting of alphanumerics, underscore or hyphens.
         the field may act either as an identifier, column name, or an enumeration of valid values.
.. [#f2] "one or more words" generally means a set of values, taken from some valid
         enumeration. These can either be separated by spaces (``"value1 value2"``), 
         or commas (``"value1,value2"``).