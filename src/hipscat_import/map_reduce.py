"""Import a set of non-hipscat files using dask for parallelization"""

import healpy as hp
import numpy as np
import pandas as pd
from hipscat import pixel_math
from hipscat.io import FilePointer, file_io, paths

# pylint: disable=too-many-locals,too-many-arguments


def _get_pixel_directory(cache_path: FilePointer, pixel: np.int64):
    dir_number = int(pixel / 10_000) * 10_000
    return file_io.append_paths_to_pointer(
        cache_path, f"dir_{dir_number}", f"pixel_{pixel}"
    )

def map_to_pixels(
    input_file: FilePointer,
    file_reader,
    shard_suffix,
    highest_order,
    ra_column,
    dec_column,
    cache_path: FilePointer = None,
    filter_function=None,
):
    """Map a file of input objects to their healpix pixels."""

    # Perform checks on the provided path
    if not file_io.does_file_or_directory_exist(input_file):
        raise FileNotFoundError(f"File not found at path: {input_file}")
    if not file_io.is_regular_file(input_file):
        raise FileNotFoundError(
            f"Directory found at path - requires regular file: {input_file}"
        )
    if not file_reader:
        raise NotImplementedError("No file reader implemented")

    required_columns = [ra_column, dec_column]
    histo = pixel_math.empty_histogram(highest_order)

    for chunk_number, data in enumerate(file_reader.read(input_file)):
        data.reset_index(inplace=True)
        if not all(x in data.columns for x in required_columns):
            raise ValueError(
                f"Invalid column names in input file: {ra_column}, {dec_column} not in {input_file}"
            )
        # Set up the data we want (filter and find pixel)
        if filter_function:
            data = filter_function(data)
            data.reset_index(inplace=True)
        mapped_pixels = hp.ang2pix(
            2**highest_order,
            data[ra_column].values,
            data[dec_column].values,
            lonlat=True,
            nest=True,
        )
        mapped_pixel, count_at_pixel = np.unique(mapped_pixels, return_counts=True)
        histo[mapped_pixel] += count_at_pixel.astype(np.int64)

        if cache_path:
            for pixel in mapped_pixel:
                data_indexes = np.where(mapped_pixels == pixel)
                filtered_data = data.filter(items=data_indexes[0].tolist(), axis=0)

                pixel_dir = _get_pixel_directory(cache_path, pixel)
                file_io.make_directory(pixel_dir, exist_ok=True)
                output_file = file_io.append_paths_to_pointer(
                    pixel_dir, f"shard_{shard_suffix}_{chunk_number}.parquet"
                )
                filtered_data.to_parquet(output_file)
            del filtered_data, data_indexes

        ## Pesky memory!
        del mapped_pixels, mapped_pixel, count_at_pixel
    return histo


def reduce_pixel_shards(
    cache_path,
    origin_pixel_numbers,
    destination_pixel_order,
    destination_pixel_number,
    destination_pixel_size,
    output_path,
    id_column,
    delete_input_files=True,
):
    """Reduce sharded source pixels into destination pixels."""
    destination_dir = paths.pixel_directory(
        output_path, destination_pixel_order, destination_pixel_number
    )
    file_io.make_directory(destination_dir, exist_ok=True)

    destination_file = paths.pixel_catalog_file(
        output_path, destination_pixel_order, destination_pixel_number
    )

    tables = []
    for pixel in origin_pixel_numbers:
        pixel_dir = _get_pixel_directory(cache_path, pixel)

        for file in file_io.get_directory_contents(pixel_dir):
            tables.append(pd.read_parquet(file, engine="pyarrow"))

    merged_table = pd.concat(tables, ignore_index=True, sort=False)
    if id_column:
        merged_table = merged_table.sort_values(by=id_column)

    rows_written = len(merged_table)

    if rows_written != destination_pixel_size:
        raise ValueError(
            "Unexpected number of objects at pixel "
            f"({destination_pixel_order}, {destination_pixel_number})."
            f" Expected {destination_pixel_size}, wrote {rows_written}"
        )

    merged_table.to_parquet(destination_file)

    del merged_table, tables

    if delete_input_files:
        for pixel in origin_pixel_numbers:
            pixel_dir = _get_pixel_directory(cache_path=cache_path, pixel=pixel)

            shutil.rmtree(pixel_dir, ignore_errors=True)
