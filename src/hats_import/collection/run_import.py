import hats.io.file_io as io

import hats_import.catalog.run_import as catalog_runner
import hats_import.index.run_index as index_runner
import hats_import.margin_cache.margin_cache as margin_runner
from hats_import.collection.arguments import CollectionArguments
from hats_import.pipeline_resume_plan import print_progress


def run(args, client):
    """Run catalog collection creation pipeline."""
    if not args:
        raise TypeError("args is required and should be type CollectionArguments")
    if not isinstance(args, CollectionArguments):
        raise TypeError("args must be type CollectionArguments")

    catalog_args = args.get_catalog_args()

    if catalog_args:
        catalog_runner.run(catalog_args, client)

    for margin_args in args.get_margin_args():
        margin_runner.generate_margin_cache(margin_args, client)

    for index_args in args.get_index_args():
        index_runner.run(index_args, client)

    ## Finishing collection
    with print_progress(
        total=2,
        stage_name="Finishing",
        use_progress_bar=args.progress_bar,
        simple_progress_bar=args.simple_progress_bar,
    ) as step_progress:
        collection_info = args.to_collection_properties()
        collection_info.to_properties_file(args.catalog_path)
        step_progress.update(1)
        io.remove_directory(args.tmp_path, ignore_errors=True)
        step_progress.update(1)
