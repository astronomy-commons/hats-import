import hats.pixel_math.healpix_shim as hp
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from hats.io import file_io, paths
from hats.pixel_math.healpix_pixel import HealpixPixel
from hats.pixel_math.spatial_index import SPATIAL_INDEX_COLUMN

from hats_import.margin_cache.margin_cache_resume_plan import MarginCachePlan
from hats_import.pipeline_resume_plan import get_pixel_cache_directory, print_task_failure


# pylint: disable=too-many-arguments,disable=too-many-locals
def map_pixel_shards(
    partition_file,
    mapping_key,
    original_catalog_metadata,
    margin_pair_file,
    margin_threshold,
    output_path,
    margin_order,
    ra_column,
    dec_column,
    fine_filtering,
):
    """Creates margin cache shards from a source partition file."""
    try:
        if fine_filtering:
            raise NotImplementedError("Fine filtering temporarily removed.")

        schema = file_io.read_parquet_metadata(original_catalog_metadata).schema.to_arrow_schema()
        data = pq.read_table(partition_file, filesystem=partition_file.fs, schema=schema).combine_chunks()
        source_pixel = HealpixPixel(data["Norder"][0].as_py(), data["Npix"][0].as_py())

        # Constrain the possible margin pairs, first by only those `margin_order` pixels
        # that **can** be contained in source pixel, then by `margin_order` pixels for rows
        # in source data
        margin_pairs = pd.read_csv(margin_pair_file)
        explosion_factor = 4 ** int(margin_order - source_pixel.order)
        margin_pixel_range_start = source_pixel.pixel * explosion_factor
        margin_pixel_range_end = (source_pixel.pixel + 1) * explosion_factor
        margin_pairs = margin_pairs.query(
            f"margin_pixel >= {margin_pixel_range_start} and margin_pixel < {margin_pixel_range_end}"
        )

        margin_pixel_list = hp.radec2pix(
            margin_order,
            data[ra_column].to_numpy(),
            data[dec_column].to_numpy(),
        )
        margin_pixel_filter = pd.DataFrame(
            {"margin_pixel": margin_pixel_list, "filter_value": np.arange(0, len(margin_pixel_list))}
        ).merge(margin_pairs, on="margin_pixel")

        # For every possible output pixel, find the full margin_order pixel filter list,
        # perform the filter, and pass along to helper method to compute fine filter
        # and write out shard file.
        num_rows = 0
        for partition_key, data_filter in margin_pixel_filter.groupby(["partition_order", "partition_pixel"]):
            data_filter = np.unique(data_filter["filter_value"])
            filtered_data = data.take(data_filter)
            pixel = HealpixPixel(partition_key[0], partition_key[1])

            num_rows += _to_pixel_shard(
                filtered_data=filtered_data,
                pixel=pixel,
                margin_threshold=margin_threshold,
                output_path=output_path,
                ra_column=ra_column,
                dec_column=dec_column,
                source_pixel=source_pixel,
                fine_filtering=fine_filtering,
            )

        MarginCachePlan.mapping_key_done(output_path, mapping_key, num_rows)
    except Exception as exception:  # pylint: disable=broad-exception-caught
        print_task_failure(f"Failed MAPPING stage for pixel: {mapping_key}", exception)
        raise exception


# pylint: disable=too-many-arguments, unused-argument
def _to_pixel_shard(
    filtered_data,
    pixel,
    margin_threshold,
    output_path,
    ra_column,
    dec_column,
    source_pixel,
    fine_filtering,
):
    """Do boundary checking for the cached partition and then output remaining data."""
    margin_data = filtered_data

    num_rows = len(margin_data)
    if num_rows:
        # generate a file name for our margin shard, that uses both sets of Norder/Npix
        partition_dir = get_pixel_cache_directory(output_path, pixel)
        shard_dir = paths.pixel_directory(partition_dir, source_pixel.order, source_pixel.pixel)

        file_io.make_directory(shard_dir, exist_ok=True)

        shard_path = paths.pixel_catalog_file(partition_dir, source_pixel)

        margin_data = _rename_original_pixel_columns(margin_data)
        margin_data = _append_margin_pixel_columns(margin_data, pixel)
        margin_data = margin_data.sort_by(SPATIAL_INDEX_COLUMN)

        pq.write_table(margin_data, shard_path.path, filesystem=shard_path.fs)
    return num_rows


def _rename_original_pixel_columns(margin_data):
    """Rename source pixel columns to include margin prefix"""
    rename_columns = {
        paths.PARTITION_ORDER: paths.MARGIN_ORDER,
        paths.PARTITION_DIR: paths.MARGIN_DIR,
        paths.PARTITION_PIXEL: paths.MARGIN_PIXEL,
    }
    return margin_data.rename_columns(rename_columns)


def _append_margin_pixel_columns(margin_data, pixel):
    """Append margin pixel columns to the shard table"""
    num_rows = len(margin_data)
    order_values = pa.repeat(pa.scalar(pixel.order, type=pa.uint8()), num_rows)
    dir_values = pa.repeat(pa.scalar(pixel.dir, type=pa.uint64()), num_rows)
    pixel_values = pa.repeat(pa.scalar(pixel.pixel, type=pa.uint64()), num_rows)
    pixel_columns = {
        paths.PARTITION_ORDER: order_values,
        paths.PARTITION_DIR: dir_values,
        paths.PARTITION_PIXEL: pixel_values,
    }
    for col_name, col_values in pixel_columns.items():
        margin_data = margin_data.append_column(col_name, col_values)
    return margin_data


def reduce_margin_shards(
    intermediate_directory,
    reducing_key,
    output_path,
    partition_order,
    partition_pixel,
    delete_intermediate_parquet_files,
):
    """Reduce all partition pixel directories into a single file"""
    try:
        healpix_pixel = HealpixPixel(partition_order, partition_pixel)
        shard_dir = get_pixel_cache_directory(intermediate_directory, healpix_pixel)

        if file_io.does_file_or_directory_exist(shard_dir):
            margin_table = ds.dataset(shard_dir, format="parquet").to_table()

            if len(margin_table):
                margin_cache_dir = paths.pixel_directory(output_path, partition_order, partition_pixel)
                file_io.make_directory(margin_cache_dir, exist_ok=True)

                margin_cache_file_path = paths.pixel_catalog_file(output_path, healpix_pixel)
                pq.write_table(
                    margin_table, margin_cache_file_path.path, filesystem=margin_cache_file_path.fs
                )

                if delete_intermediate_parquet_files:
                    file_io.remove_directory(shard_dir)

        MarginCachePlan.reducing_key_done(intermediate_directory, reducing_key)
    except Exception as exception:  # pylint: disable=broad-exception-caught
        print_task_failure(f"Failed REDUCING stage for pixel: {reducing_key}", exception)
        raise exception
