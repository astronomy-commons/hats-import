HATS Import
========================================================================================

Utility for ingesting large survey data into HATS structure.

Installation
-------------------------------------------------------------------------------

We recommend installing in a virtual environment, like venv or conda. You may
need to install or upgrade versions of dependencies to work with hats-import.

.. code-block:: console

    pip install hats-import

.. tip::
    Installing optional dependencies

    There are some extra dependencies that can make running hats-import in a jupyter
    environment easier, or connecting to a variety of remote file systems.

    These can be installed with the ``full`` extra.

    .. code-block:: console

        >> pip install hats-import[full]

.. tip::
    Installing on Mac

    ``healpy`` is an optional dependency for hats-import (included in the ``full`` extra)
    to support converting from older HiPSCat catalogs, but
    native prebuilt binaries for healpy on Apple Silicon Macs 
    `do not yet exist <https://healpy.readthedocs.io/en/latest/install.html#binary-installation-with-pip-recommended-for-most-other-python-users>`_, 
    so it's recommended to install via conda before proceeding to hats-import.

    .. code-block:: console

        >> conda config --append channels conda-forge
        >> conda install healpy



Setting up a pipeline
-------------------------------------------------------------------------------

For each type of dataset the hats-import tool can generate, there is an argument
container class that you will need to instantiate and populate with relevant arguments.

See dataset-specific notes on arguments:

* :doc:`catalogs/arguments` (most common)
* :doc:`guide/hipscat_conversion`
* :doc:`guide/margin_cache`
* :doc:`guide/index_table`

Once you have created your arguments object, you pass it into the pipeline control,
and then wait. Running within a main guard will potentially avoid some python
threading issues with dask:

.. code-block:: python

    from dask.distributed import Client
    from hats_import.pipeline import pipeline_with_client

    def main():
        args = ...
        with Client(
            n_workers=10,
            threads_per_worker=1,
            ... 
        ) as client:
            pipeline_with_client(args, client)

    if __name__ == '__main__':
        main()

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Catalogs

   catalogs/arguments
   guide/hipscat_conversion
   catalogs/temp_files
   catalogs/public/index

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Other Datasets

   guide/margin_cache
   guide/index_table   
   Notebooks <notebooks>
   guide/dask_on_ray

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Developers

   guide/contributing
   API Reference <autoapi/index>
   guide/contact

Acknowledgements
-------------------------------------------------------------------------------

This project is supported by Schmidt Sciences.

This project is based upon work supported by the National Science Foundation
under Grant No. AST-2003196.

This project acknowledges support from the DIRAC Institute in the Department of 
Astronomy at the University of Washington. The DIRAC Institute is supported 
through generous gifts from the Charles and Lisa Simonyi Fund for Arts and 
Sciences, and the Washington Research Foundation.