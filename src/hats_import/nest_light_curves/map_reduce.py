"""Inner methods for nest light curves pipeline"""

import numpy as np
import pandas as pd
from hats import pixel_math
from hats.catalog.partition_info import PartitionInfo
from hats.io import file_io, paths
from hats.pixel_math.healpix_pixel import HealpixPixel
from hats.pixel_math.sparse_histogram import SparseHistogram
from hats.pixel_math.spatial_index import SPATIAL_INDEX_COLUMN, spatial_index_to_healpix
from nested_pandas.utils import count_nested

from hats_import.nest_light_curves.arguments import NestLightCurveArguments
from hats_import.pipeline_resume_plan import print_task_failure


def _join_nested(
    args: NestLightCurveArguments, object_pixel: HealpixPixel, source_pixels: list[HealpixPixel]
):
    ## TODO: dir / alternate suffix
    object_path = paths.pixel_catalog_file(args.object_catalog_dir, object_pixel)
    object_data = file_io.read_parquet_file_to_pandas(
        object_path,
        schema=args.object_catalog.schema,
    )
    object_index = object_data[[args.object_id_column]].set_index(args.object_id_column)

    results = []

    for source_pixel in source_pixels:
        source_path = paths.pixel_catalog_file(args.source_catalog_dir, source_pixel)
        source_data = file_io.read_parquet_file_to_pandas(
            source_path,
            schema=args.source_catalog.schema,
            columns=args.read_source_columns(),
        ).set_index(args.source_object_id_column)

        joined_data = source_data.merge(object_index, how="inner", left_index=True, right_index=True)

        results.append(joined_data)
    sources = pd.concat(results)

    return (
        object_data.set_index(args.object_id_column)
        .join_nested(sources, args.nested_column_name)
        .reset_index()
    )


def _generate_alignment(args, light_curves):
    mapped_pixels = spatial_index_to_healpix(
        light_curves[SPATIAL_INDEX_COLUMN], target_order=args.highest_order
    )
    if args.partition_strategy == "object_count":
        mapped_pixel, count_at_pixel = np.unique(mapped_pixels, return_counts=True)
        row_count_histo = SparseHistogram(mapped_pixel, count_at_pixel, args.highest_order)
    elif args.partition_strategy == "source_count":
        ## TODO: line these up nicely.
        print(np.sum(count_nested(light_curves, args.nested_column_name, join=False)))
    elif args.partition_strategy == "mem_size":
        ## TODO: implement
        pass
    else:
        raise ValueError(f"Unknown partition strategy: {args.partition_strategy}")

    alignment = pixel_math.generate_alignment(
        row_count_histo.to_array(),
        highest_order=args.highest_order,
        lowest_order=args.lowest_order,
        threshold=args.partition_threshold,
    )
    alignment = np.array([x if x is not None else [-1, -1, 0] for x in alignment], dtype=np.int64)
    return alignment


def _split_to_partitions(args, light_curves, alignment, target_order):
    mapped_pixels = spatial_index_to_healpix(light_curves[SPATIAL_INDEX_COLUMN], target_order=target_order)

    aligned_pixels = alignment[mapped_pixels]
    unique_pixels, unique_inverse = np.unique(aligned_pixels, return_inverse=True, axis=0)

    for unique_index, pixel_alignment_count in enumerate(unique_pixels):
        order = pixel_alignment_count[0]
        pixel = pixel_alignment_count[1]

        destination_dir = paths.pixel_directory(args.output_path, order, pixel)
        file_io.make_directory(destination_dir, exist_ok=True)

        destination_file = paths.pixel_catalog_file(
            args.output_path, HealpixPixel(order, pixel), npix_suffix=args.npix_suffix
        )
        filtered_data = light_curves.iloc[unique_inverse == unique_index]

        filtered_data.to_parquet(destination_file.path, filesystem=destination_file.fs)
        del filtered_data


def _write_partition_info(args, object_pixel, alignment):
    object_pixel_list = list(
        {HealpixPixel(tuple[0], tuple[1]) for tuple in alignment if tuple is not None and int(tuple[2]) > 0}
    )
    partition_info = PartitionInfo.from_healpix(object_pixel_list)
    file_io.write_dataframe_to_csv(
        dataframe=partition_info.as_dataframe(),
        file_pointer=args.tmp_path / f"{object_pixel.order}_{object_pixel.pixel}.csv",
        index=False,
    )


def count_joins(args: NestLightCurveArguments, object_pixel: HealpixPixel, source_pixels: list[HealpixPixel]):
    try:
        ## ..........    MAPPING  ..............
        ## Get the object data partition, and join in all of the matching
        ## source data partitions, keeping where object id matches.
        light_curves = _join_nested(args, object_pixel=object_pixel, source_pixels=source_pixels)

        ## ..........    BINNING  ..............
        ## Determine the output partitions
        alignment = _generate_alignment(args, light_curves)

        ## ..........    SPLITTING  ..............
        ## Split the object data partition, according to the output partitions
        ## TODO: use good row groups / compression
        _split_to_partitions(args, light_curves, alignment, args.highest_order)

        ## ..........    FINISHING  ..............
        ## Write the new partition list.
        ## There aren't any intermediate files to clean up!
        _write_partition_info(args, object_pixel, alignment)
    except Exception as exception:  # pylint: disable=broad-exception-caught
        print_task_failure(f"Failed stage for shard: {object_pixel}", exception)
        raise exception
