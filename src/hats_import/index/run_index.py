"""Create columnar index of hats table using dask for parallelization"""

from hats.io import file_io, parquet_metadata

import hats_import.index.map_reduce as mr
from hats_import.index.arguments import IndexArguments
from hats_import.pipeline_resume_plan import print_progress


def run(args, client):
    """Run index creation pipeline."""
    if not args:
        raise TypeError("args is required and should be type IndexArguments")
    if not isinstance(args, IndexArguments):
        raise TypeError("args must be type IndexArguments")
    rows_written = mr.create_index(args, client)

    # All done - write out the metadata
    with print_progress(
        total=3,
        stage_name="Finishing",
        pipeline_name="index",
        use_progress_bar=args.progress_bar,
        simple_progress_bar=args.simple_progress_bar,
        tqdm_kwargs=args.tqdm_kwargs,
    ) as step_progress:
        index_catalog_info = args.to_table_properties(int(rows_written))
        index_catalog_info.to_properties_file(args.catalog_path)
        step_progress.update(1)
        file_io.remove_directory(args.tmp_path, ignore_errors=True)
        step_progress.update(1)
        parquet_metadata.write_parquet_metadata(args.catalog_path, order_by_healpix=False)
        step_progress.update(1)
