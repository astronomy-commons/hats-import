import pytest

from hats_import.collection.arguments import CollectionArguments
from hats_import.collection.run_import import run


@pytest.mark.dask
def test_import_collection(
    dask_client,
    small_sky_source_dir,
    tmp_path,
):
    args = (
        CollectionArguments(
            output_artifact_name="small_sky",
            output_path=tmp_path,
            progress_bar=False,
        )
        .catalog(
            input_path=small_sky_source_dir,
            file_reader="csv",
            catalog_type="source",
            ra_column="source_ra",
            dec_column="source_dec",
            sort_columns="source_id",
            highest_healpix_order=2,
        )
        .add_margin(output_artifact_name="small_sky_5arcs", margin_threshold=5.0)
        .add_margin(output_artifact_name="small_sky_15arcs", margin_threshold=15.0)
    )
    run(args, dask_client)
