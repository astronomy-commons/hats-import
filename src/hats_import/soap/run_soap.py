"""Create pixel-to-pixel association between object and source catalogs.
Methods in this file set up a dask pipeline using futures.
The actual logic of the map reduce is in the `map_reduce.py` file.
"""

from hats.catalog import PartitionInfo, PartitionJoinInfo
from hats.io import parquet_metadata, paths

from hats_import.soap.arguments import SoapArguments
from hats_import.soap.map_reduce import combine_partial_results, count_joins, reduce_joins
from hats_import.soap.resume_plan import SoapPlan


def run(args, client):
    """Run the association pipeline"""
    if not args:
        raise TypeError("args is required and should be type SoapArguments")
    if not isinstance(args, SoapArguments):
        raise TypeError("args must be type SoapArguments")

    resume_plan = SoapPlan(args)
    if not resume_plan.is_counting_done():
        futures = []
        for source_pixel, object_pixels, _ in resume_plan.count_keys:
            futures.append(
                client.submit(
                    count_joins,
                    soap_args=args,
                    source_pixel=source_pixel,
                    object_pixels=object_pixels,
                )
            )

        resume_plan.wait_for_counting(futures)

    if args.write_leaf_files and not resume_plan.is_reducing_done():
        for object_pixel, object_key in resume_plan.reduce_keys:
            futures.append(
                client.submit(
                    reduce_joins,
                    soap_args=args,
                    object_pixel=object_pixel,
                    object_key=object_key,
                    delete_input_files=args.delete_intermediate_parquet_files,
                )
            )

        resume_plan.wait_for_reducing(futures)

    # All done - write out the metadata
    with resume_plan.print_progress(total=4, stage_name="Finishing") as step_progress:
        if args.write_leaf_files:
            total_rows = parquet_metadata.write_parquet_metadata(args.catalog_path)
            metadata_path = paths.get_parquet_metadata_pointer(args.catalog_path)
            partition_join_info = PartitionJoinInfo.read_from_file(metadata_path)
            partition_join_info.write_to_csv(catalog_path=args.catalog_path)
        else:
            total_rows = combine_partial_results(args.tmp_path, args.catalog_path)
        step_progress.update(1)
        partition_info = PartitionInfo.read_from_dir(args.catalog_path)
        catalog_info = args.to_table_properties(
            total_rows, partition_info.get_highest_order(), partition_info.calculate_fractional_coverage()
        )
        catalog_info.to_properties_file(args.catalog_path)
        step_progress.update(1)
        ## TODO - optionally write out arguments file
        step_progress.update(1)
        resume_plan.clean_resume_files()
        step_progress.update(1)
