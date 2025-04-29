import pytest
from hats import read_hats

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
        .add_margin(margin_threshold=5.0)
        .add_margin(margin_threshold=50.0)
        .add_index(indexing_column="object_id", include_healpix_29=False)
        .add_index(indexing_column="source_id", include_healpix_29=False)
    )
    run(args, dask_client)

    collection = read_hats(tmp_path / "small_sky")
    assert collection.collection_path == args.catalog_path

    assert collection.all_margins == ["small_sky_5arcs", "small_sky_50arcs"]
    assert collection.all_indexes == {"object_id": "small_sky_object_id", "source_id": "small_sky_source_id"}

    catalog = read_hats(tmp_path / "small_sky" / "small_sky")
    assert catalog.on_disk
    assert catalog.catalog_info.ra_column == "source_ra"
    assert catalog.catalog_info.dec_column == "source_dec"

    catalog = read_hats(tmp_path / "small_sky" / "small_sky_5arcs")
    assert catalog.on_disk
    assert len(catalog.get_healpix_pixels()) == 2
    assert len(catalog) == 4

    catalog = read_hats(tmp_path / "small_sky" / "small_sky_50arcs")
    assert catalog.on_disk
    assert len(catalog.get_healpix_pixels()) == 2
    assert len(catalog) == 17

    catalog = read_hats(tmp_path / "small_sky" / "small_sky_object_id")
    ## This isn't 131, because one object's sources spans two pixels.
    assert catalog.catalog_info.total_rows == 132

    catalog = read_hats(tmp_path / "small_sky" / "small_sky_source_id")
    assert catalog.catalog_info.total_rows == 17161
