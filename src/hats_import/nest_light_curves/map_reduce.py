"""Inner methods for nest light curves pipeline"""

import numpy as np
import pandas as pd
from hats import pixel_math
from hats.io import file_io, paths, size_estimates
from hats.pixel_math.healpix_pixel import HealpixPixel
from hats.pixel_math.sparse_histogram import HistogramAggregator, supplemental_count_histogram
from hats.pixel_math.spatial_index import SPATIAL_INDEX_COLUMN, spatial_index_to_healpix
from nested_pandas.utils import count_nested

from hats_import.nest_light_curves.arguments import NestLightCurveArguments
from hats_import.pipeline_resume_plan import get_pixel_cache_directory, print_task_failure


def _do_mapping(args: NestLightCurveArguments, object_pixel: HealpixPixel, source_pixels: list[HealpixPixel]):
    """Determine the row count of objects, at a finer grain level, since we likely need to split
    the object pixel further to account for larger data size with the nested columns.

    Some objects may have sources in many neighboring pixels (that's why this pipeline exists),
    but it complicates the histogram calculation: we only want to consider them for the "row count"
    aspect if they have *some* matches in the source catalog (and that's not guaranteed).
    """
    object_path = paths.pixel_catalog_file(
        args.object_catalog_dir, object_pixel, npix_suffix=args.object_npix_suffix
    )
    object_data = file_io.read_parquet_file_to_pandas(
        object_path,
        schema=args.object_catalog_schema,
    )
    object_index = object_data[[args.object_id_column]].set_index(args.object_id_column)
    object_mask = object_data[[args.object_id_column, SPATIAL_INDEX_COLUMN]].set_index(args.object_id_column)
    object_mask["mask"] = False

    object_data = object_data.set_index(args.object_id_column)
    non_empty_sources = []
    supplemental_size_aggregator = HistogramAggregator(args.highest_healpix_order)

    for source_pixel in source_pixels:
        # Load the source pixel data, and attempt to join in any matching detections.
        source_path = paths.pixel_catalog_file(
            args.source_catalog_dir, source_pixel, npix_suffix=args.source_npix_suffix
        )
        source_data = file_io.read_parquet_file_to_pandas(
            source_path, columns=args.read_source_columns()
        ).set_index(args.source_object_id_column)

        joined_data = source_data.merge(object_index, how="inner", left_index=True, right_index=True)
        if len(joined_data) == 0:
            continue

        # Mark this pixel for future actions.
        non_empty_sources.append(source_pixel)
        # Mark objects that are active in the join.
        object_mask.loc[joined_data.index, "mask"] = True

        if args.partition_strategy == "object_count":
            continue

        # If we want to use a supplemental count strategy, do the full join
        # for more accurate histograms/counting.
        light_curves = (
            object_data.join_nested(joined_data, args.nested_column_name, how="inner")
            .reset_index()
            .dropna(subset=args.nested_column_name)
        )

        mapped_pixels = spatial_index_to_healpix(
            light_curves[SPATIAL_INDEX_COLUMN], target_order=args.highest_healpix_order
        )

        supplemental_count = None
        if args.partition_strategy == "source_count":
            supplemental_count = count_nested(light_curves, args.nested_column_name, join=False)[
                f"n_{args.nested_column_name}"
            ].values
        elif args.partition_strategy == "mem_size":
            supplemental_count = size_estimates.get_mem_size_per_row(light_curves)

        _, supplemental_size_partial = supplemental_count_histogram(
            mapped_pixels, supplemental_count, highest_order=args.highest_healpix_order
        )
        supplemental_size_aggregator.add(supplemental_size_partial)

    row_count_historam, _ = supplemental_count_histogram(
        spatial_index_to_healpix(
            object_mask.loc[object_mask["mask"] == True][SPATIAL_INDEX_COLUMN],
            target_order=args.highest_healpix_order,
        ),
        None,
        highest_order=args.highest_healpix_order,
    )

    alignment = pixel_math.generate_alignment(
        row_count_historam.to_array(),
        highest_order=args.highest_healpix_order,
        lowest_order=object_pixel.order,
        threshold=args.partition_threshold,
        mem_size_histogram=(
            None if args.partition_strategy == "object_count" else supplemental_size_aggregator.full_histogram
        ),
    )
    alignment = np.array([x if x is not None else [-1, -1, 0] for x in alignment], dtype=np.int64)
    return alignment, non_empty_sources


def _split_to_partitions(args, alignment, object_pixel: HealpixPixel, source_pixels: list[HealpixPixel]):
    object_path = paths.pixel_catalog_file(
        args.object_catalog_dir, object_pixel, npix_suffix=args.object_npix_suffix
    )
    object_data = file_io.read_parquet_file_to_pandas(
        object_path,
        schema=args.object_catalog_schema,
    )
    object_index = object_data[[args.object_id_column]].set_index(args.object_id_column)
    object_data = object_data.set_index(args.object_id_column)
    mapped_pixels = spatial_index_to_healpix(
        object_data[SPATIAL_INDEX_COLUMN], target_order=args.highest_healpix_order
    )

    aligned_pixels = alignment[mapped_pixels]
    zipper = dict(zip(object_data.index, aligned_pixels))

    for source_pixel in source_pixels:
        source_path = paths.pixel_catalog_file(
            args.source_catalog_dir, source_pixel, npix_suffix=args.source_npix_suffix
        )
        source_data = file_io.read_parquet_file_to_pandas(
            source_path, columns=args.read_source_columns()
        ).set_index(args.source_object_id_column)

        joined_data = source_data.merge(object_index, how="inner", left_index=True, right_index=True)
        mapped_from_object = [zipper[object_id] for object_id in joined_data.index]

        unique_pixels, unique_inverse = np.unique(mapped_from_object, return_inverse=True, axis=0)
        unique_pixels = [HealpixPixel(pixel[0], pixel[1]) for pixel in unique_pixels]

        for unique_index, destination_pixel in enumerate(unique_pixels):
            pixel_dir = get_pixel_cache_directory(args.tmp_path, destination_pixel)

            file_io.make_directory(pixel_dir, exist_ok=True)
            output_file = file_io.append_paths_to_pointer(
                pixel_dir, f"shard_{source_pixel.order}_{source_pixel.pixel}.parquet"
            )

            filtered_data = joined_data.iloc[unique_inverse == unique_index]

            filtered_data.to_parquet(
                output_file.path,
                filesystem=output_file.fs,
                **args.write_table_kwargs,
            )
            del filtered_data


def _reduce_partitions(args, object_pixel, pixel_list):
    object_path = paths.pixel_catalog_file(
        args.object_catalog_dir, object_pixel, npix_suffix=args.object_npix_suffix
    )
    object_data = file_io.read_parquet_file_to_pandas(
        object_path,
        schema=args.object_catalog_schema,
    )
    object_data = object_data.set_index(args.object_id_column)

    for order, pix, row_count in pixel_list:
        destination_pixel = HealpixPixel(order, pix)

        destination_file = paths.new_pixel_catalog_file(
            args.catalog_path,
            destination_pixel,
            npix_suffix=args.npix_suffix,
            npix_parquet_name=args.npix_parquet_name,
        )

        try:
            _check_destination_file(
                destination_file,
                row_count,
                destination_pixel,
                args.delete_intermediate_parquet_files,
                args.tmp_path,
            )

            return
        except:  # pylint: disable=bare-except
            pass

        pixel_dir = get_pixel_cache_directory(args.tmp_path, destination_pixel)
        flat_light_curves = file_io.read_parquet_file_to_pandas(pixel_dir, is_dir=True).set_index(
            args.source_object_id_column
        )

        light_curves = object_data.join_nested(flat_light_curves, args.nested_column_name).reset_index()
        light_curves = light_curves.dropna(subset=args.nested_column_name)
        if args.nested_sort_column is not None:
            light_curves = light_curves.sort_values(args.nested_sort_column)

        light_curves.to_parquet(destination_file, **args.write_table_kwargs)

        _check_destination_file(
            destination_file,
            row_count,
            destination_pixel,
            args.delete_intermediate_parquet_files,
            args.tmp_path,
        )


def _check_destination_file(
    destination_file,
    destination_pixel_size,
    healpix_pixel,
    delete_input_files,
    cache_shard_path,
):
    if not destination_file.exists():
        raise FileNotFoundError(f"Reduced file not found where expected ({destination_file})")

    rows_written = file_io.read_parquet_metadata(destination_file).num_rows

    if rows_written != destination_pixel_size:
        raise ValueError(
            "Unexpected number of objects in pixel data "
            f"({healpix_pixel})."
            f" Expected {destination_pixel_size}, found {rows_written}"
        )
    if delete_input_files:
        pixel_dir = get_pixel_cache_directory(cache_shard_path, healpix_pixel)
        file_io.remove_directory(pixel_dir, ignore_errors=True)

    return True


def _write_partition_info(args, object_pixel, pixel_list):
    partition_info = pd.DataFrame(pixel_list, columns=["Norder", "Npix", "num_rows"])
    file_io.write_dataframe_to_csv(
        dataframe=partition_info,
        file_pointer=args.tmp_path / f"{object_pixel.order}_{object_pixel.pixel}.csv",
        index=False,
    )


def _perform_nest(
    args: NestLightCurveArguments, object_pixel: HealpixPixel, source_pixels: list[HealpixPixel]
):
    try:
        # ..........    MAPPING / BINNING  ..............
        # Get the object data partition, and join in all of the matching
        # source data partitions, keeping where object id matches.
        alignment, non_empty_sources = _do_mapping(
            args, object_pixel=object_pixel, source_pixels=source_pixels
        )

        # ..........    SPLITTING  ..............
        # Split the object and source data partitions, according to the output partitions
        _split_to_partitions(
            args,
            alignment,
            object_pixel=object_pixel,
            source_pixels=non_empty_sources,
        )
        pixel_list = np.unique(alignment, axis=0)
        pixel_list = {(order, pix, row_count) for (order, pix, row_count) in pixel_list if int(row_count) > 0}

        # ..........    REDUCING   ..............
        # Reduce output partitions into final parquet files.
        if len(non_empty_sources) > 0:
            _reduce_partitions(args, object_pixel, pixel_list)

        # ..........    FINISHING  ..............
        # Write the new partition list.
        _write_partition_info(args, object_pixel, pixel_list)
    except Exception as exception:  # pylint: disable=broad-exception-caught
        print_task_failure(f"Failed stage for shard: {object_pixel}", exception)
        raise exception
