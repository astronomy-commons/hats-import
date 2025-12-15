from hats.catalog import PartitionInfo
from hats.io import parquet_metadata

from hats_import.nest_light_curves.arguments import NestLightCurveArguments
from hats_import.nest_light_curves.map_reduce import _perform_nest
from hats_import.nest_light_curves.resume_plan import NestLightCurvePlan


def run(args, client):
    """Run the association pipeline"""
    if not args:
        raise TypeError("args is required and should be type NestLightCurveArguments")
    if not isinstance(args, NestLightCurveArguments):
        raise TypeError("args must be type NestLightCurveArguments")

    resume_plan = NestLightCurvePlan(args)
    if not resume_plan.is_counting_done():
        futures = []
        for source_pixel, object_pixels, _ in resume_plan.count_keys:
            futures.append(
                client.submit(
                    _perform_nest,
                    args=args,
                    object_pixel=source_pixel,
                    source_pixels=object_pixels,
                )
            )

        resume_plan.wait_for_counting(futures)

    # All done - write out the metadata
    with resume_plan.print_progress(total=3, stage_name="Finishing") as step_progress:
        total_rows = resume_plan.combine_partial_results(args.catalog_path)
        md_total_rows = parquet_metadata.write_parquet_metadata(
            args.catalog_path, create_thumbnail=args.create_thumbnail, create_metadata=args.create_metadata
        )
        if total_rows != md_total_rows:
            raise ValueError("Unexpected total rows.")
        step_progress.update(1)
        partition_info = PartitionInfo.read_from_dir(args.catalog_path)
        catalog_info = args.to_table_properties(
            total_rows, partition_info.get_highest_order(), partition_info.calculate_fractional_coverage()
        )
        catalog_info.to_properties_file(args.catalog_path)
        step_progress.update(1)
        resume_plan.clean_resume_files()
        step_progress.update(1)
