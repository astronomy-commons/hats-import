"""Import a set of non-hats files using dask for parallelization

Methods in this file set up a dask pipeline using futures.
The actual logic of the map reduce is in the `map_reduce.py` file.
"""

import os
import pickle

import hats.io.file_io as io
from hats.catalog import PartitionInfo
from hats.io import paths
from hats.io.parquet_metadata import write_parquet_metadata
from hats.io.validation import is_valid_catalog

import hats_import.catalog.map_reduce as mr
from hats_import.catalog.arguments import ImportArguments
from hats_import.catalog.resume_plan import ResumePlan


def run(args, client):
    """Run catalog creation pipeline."""
    if not args:
        raise ValueError("args is required and should be type ImportArguments")
    if not isinstance(args, ImportArguments):
        raise ValueError("args must be type ImportArguments")

    resume_plan = ResumePlan(import_args=args)

    pickled_reader_file = os.path.join(resume_plan.tmp_path, "reader.pickle")
    with open(pickled_reader_file, "wb") as pickle_file:
        pickle.dump(args.file_reader, pickle_file)

    if resume_plan.should_run_mapping:
        futures = []
        for key, file_path in resume_plan.map_files:
            futures.append(
                client.submit(
                    mr.map_to_pixels,
                    input_file=file_path,
                    resume_path=resume_plan.tmp_path,
                    pickled_reader_file=pickled_reader_file,
                    mapping_key=key,
                    highest_order=args.mapping_healpix_order,
                    ra_column=args.ra_column,
                    dec_column=args.dec_column,
                    use_healpix_29=args.use_healpix_29,
                )
            )
        resume_plan.wait_for_mapping(futures)

    with resume_plan.print_progress(total=2, stage_name="Binning") as step_progress:
        raw_histogram = resume_plan.read_histogram(args.mapping_healpix_order)
        total_rows = int(raw_histogram.sum())
        if args.expected_total_rows > 0 and args.expected_total_rows != total_rows:
            raise ValueError(
                f"Number of rows ({total_rows}) does not match expectation ({args.expected_total_rows})"
            )

        step_progress.update(1)
        alignment_file = resume_plan.get_alignment_file(
            raw_histogram,
            args.constant_healpix_order,
            args.highest_healpix_order,
            args.lowest_healpix_order,
            args.pixel_threshold,
            args.drop_empty_siblings,
            total_rows,
        )

        step_progress.update(1)

    if resume_plan.should_run_splitting:
        futures = []
        for key, file_path in resume_plan.split_keys:
            futures.append(
                client.submit(
                    mr.split_pixels,
                    input_file=file_path,
                    pickled_reader_file=pickled_reader_file,
                    highest_order=args.mapping_healpix_order,
                    ra_column=args.ra_column,
                    dec_column=args.dec_column,
                    splitting_key=key,
                    cache_shard_path=args.tmp_path,
                    resume_path=resume_plan.tmp_path,
                    alignment_file=alignment_file,
                    use_healpix_29=args.use_healpix_29,
                )
            )

        resume_plan.wait_for_splitting(futures)

    if resume_plan.should_run_reducing:
        futures = []
        for (
            destination_pixel,
            source_pixel_count,
            destination_pixel_key,
        ) in resume_plan.get_reduce_items():
            futures.append(
                client.submit(
                    mr.reduce_pixel_shards,
                    cache_shard_path=args.tmp_path,
                    resume_path=resume_plan.tmp_path,
                    reducing_key=destination_pixel_key,
                    destination_pixel_order=destination_pixel.order,
                    destination_pixel_number=destination_pixel.pixel,
                    destination_pixel_size=source_pixel_count,
                    output_path=args.catalog_path,
                    ra_column=args.ra_column,
                    dec_column=args.dec_column,
                    sort_columns=args.sort_columns,
                    add_healpix_29=args.add_healpix_29,
                    use_schema_file=args.use_schema_file,
                    use_healpix_29=args.use_healpix_29,
                    delete_input_files=args.delete_intermediate_parquet_files,
                )
            )

        resume_plan.wait_for_reducing(futures)

    # All done - write out the metadata
    if resume_plan.should_run_finishing:
        with resume_plan.print_progress(total=5, stage_name="Finishing") as step_progress:
            partition_info = PartitionInfo.from_healpix(resume_plan.get_destination_pixels())
            partition_info_file = paths.get_partition_info_pointer(args.catalog_path)
            partition_info.write_to_file(partition_info_file)
            if not args.debug_stats_only:
                parquet_rows = write_parquet_metadata(args.catalog_path)
                if total_rows > 0 and parquet_rows != total_rows:
                    raise ValueError(
                        f"Number of rows in parquet ({parquet_rows}) "
                        f"does not match expectation ({total_rows})"
                    )
            step_progress.update(1)
            io.write_fits_image(raw_histogram, paths.get_point_map_file_pointer(args.catalog_path))
            step_progress.update(1)
            catalog_info = args.to_table_properties(
                total_rows, partition_info.get_highest_order(), partition_info.calculate_fractional_coverage()
            )
            catalog_info.to_properties_file(args.catalog_path)
            step_progress.update(1)
            resume_plan.clean_resume_files()
            step_progress.update(1)
            assert is_valid_catalog(args.catalog_path)
            step_progress.update(1)
