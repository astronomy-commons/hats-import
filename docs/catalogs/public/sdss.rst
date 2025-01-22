SDSS
===============================================================================

Getting the data
-------------------------------------------------------------------------------

See docs at SDSS: https://data.sdss.org/sas/dr12/boss/sweeps/dr9/301/

We wanted to use the SDSS dr16q ``stars`` slice, so that started with this
:download:`wget_script</static/sdss_wget.bash>`.

Challenges with this data set
-------------------------------------------------------------------------------

The FITS files contain several multidimensional fields. These are not
easily convertible into numpy/pandas (this will just fail in the astropy
library).

The approach we took was to convert the FITS files into parquet in a separate
process. In this process, we expand all of the multidimensional fields into
simple scalars. This works on this data set because these multidimensional
are just encoding data on the five bands as an array with 5 elements.

Example conversion
-------------------------------------------------------------------------------

.. code-block:: python

    import glob
    import re
    from tqdm.auto import tqdm
    from astropy.table import Table
    from astropy.table.table import descr

    files = glob.glob("/data/sdss/**.fits.gz")
    files.sort()

    for in_file in tqdm(files, bar_format='{l_bar}{bar:80}{r_bar}'):
        
        match = re.match(r".*(calibObj-.*-star).fits.gz", str(in_file))
        file_prefix = match.group(1)
        out_file = f"/data/sdss/parquet/{file_prefix}.parquet"

        table = Table.read(in_file)        
        new_table = Table()

        for col in table.columns.values():
            descriptor = descr(col)
            col_name = descriptor[0]
            col_shape = descriptor[2]
            if col_shape == (5,):
                data_t = col.data.T
                for index, band_char in enumerate('ugriz'):
                    new_table.add_column(data_t[index], name=f"{col_name}_{band_char}")
            elif col_shape == ():
                new_table.add_column(col)

        new_table.to_pandas().to_parquet(out_file)


Example import
-------------------------------------------------------------------------------

.. code-block:: python

    from hats_import.catalog.arguments import ImportArguments
    import hats_import.pipeline as runner

    args = ImportArguments(
        output_artifact_name="sdss_dr16q",
        input_path="/data/sdss/parquet/",
        file_reader="parquet",
        ra_column="RA",
        dec_column="DEC",
        sort_columns="ID",
        pixel_threshold=1_000_000,
        highest_healpix_order=7,
        output_path="/path/to/catalogs/",
    )
    runner.pipeline(args)
