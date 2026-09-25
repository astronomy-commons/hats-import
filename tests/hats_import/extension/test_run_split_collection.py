"""Tests of splitting a catalog collection into a collection of core and extension"""

import shutil

import pandas as pd
import pytest
from hats import read_hats
from hats.catalog import CatalogCollection, CatalogType, CollectionProperties
from hats.io import paths
from hats.io.validation import is_valid_collection

import hats_import.extension.run_split_import as runner
from hats_import.extension.arguments import ExtensionArguments

# pylint: disable=redefined-outer-name

MAIN_NAME = "small_sky_order1"
MARGIN_NAME = "small_sky_order1_margin"
EMPTY_MARGIN_NAME = "small_sky_order1_margin_10arcs"
INDEX_NAME = "small_sky_order1_id_index"
EXTENSION_NAME = "small_sky_order1_errors"


def split_args(input_collection, tmp_path, **kwargs):
    """Arguments for splitting the input collection, with any overrides."""
    arguments = {
        "input_catalog_path": input_collection,
        "extension_columns": ["ra_error", "dec_error"],
        "primary_column": "id",
        "join_column": "object_id",
        "extension_name": "errors",
        "output_path": tmp_path / "output",
        "output_artifact_name": "small_sky_collection",
        "progress_bar": False,
    }
    return ExtensionArguments(**(arguments | kwargs))


@pytest.fixture
def split_collection(small_sky_o1_collection, tmp_path, dask_client):
    """Split the on-disk collection, and return the collection it was written to."""
    runner.run(split_args(small_sky_o1_collection, tmp_path), dask_client)
    return tmp_path / "output" / "small_sky_collection"


@pytest.mark.dask
def test_split_collection(small_sky_o1_collection, split_collection):
    """The output is one collection, holding the core, its margins, its index and the extension."""
    original = read_hats(small_sky_o1_collection)

    collection = read_hats(split_collection)
    assert isinstance(collection, CatalogCollection)
    assert is_valid_collection(split_collection, strict=True)

    ## The core and the collection members of the input keep their names.
    assert collection.collection_properties.hats_primary_table_url == MAIN_NAME
    assert collection.all_margins == [MARGIN_NAME, EMPTY_MARGIN_NAME]
    assert collection.default_margin == MARGIN_NAME
    assert collection.all_indexes == {"id": INDEX_NAME}
    assert collection.default_index_field == "id"
    ## Only the extension is new.
    assert collection.all_extensions == [EXTENSION_NAME]

    core = collection.main_catalog
    assert core.catalog_info.catalog_name == MAIN_NAME
    assert core.catalog_info.catalog_type == CatalogType.OBJECT
    assert core.schema.names == ["_healpix_29", "id", "ra", "dec"]
    assert core.get_healpix_pixels() == original.main_catalog.get_healpix_pixels()


@pytest.mark.dask
def test_split_collection_extension(split_collection):
    """The extension is a collection of its own, inside the one this run writes."""
    extension_collection = read_hats(split_collection / EXTENSION_NAME)
    assert isinstance(extension_collection, CatalogCollection)
    assert is_valid_collection(split_collection / EXTENSION_NAME, strict=True)
    assert extension_collection.collection_properties.hats_primary_table_url == EXTENSION_NAME
    assert extension_collection.all_margins == [f"{EXTENSION_NAME}_margin", f"{EXTENSION_NAME}_margin_10arcs"]
    assert extension_collection.default_margin == f"{EXTENSION_NAME}_margin"
    ## The index belongs to the parent collection.
    assert extension_collection.all_indexes is None

    extension = extension_collection.main_catalog
    properties = extension.catalog_info
    assert properties.catalog_name == EXTENSION_NAME
    assert properties.catalog_type == CatalogType.EXTENSION
    assert properties.extension_columns == ["ra_error", "dec_error"]
    assert extension.schema.names == ["_healpix_29", "object_id", "ra", "dec", "ra_error", "dec_error"]

    ## References are relative to the directory holding the collection this run writes.
    assert properties.primary_catalog == f"small_sky_collection/{MAIN_NAME}"
    assert properties.join_catalog == f"small_sky_collection/{EXTENSION_NAME}/{EXTENSION_NAME}"
    core_reference = split_collection.parent / properties.primary_catalog
    assert core_reference == split_collection / MAIN_NAME
    assert read_hats(core_reference).catalog_info.catalog_name == MAIN_NAME


@pytest.mark.dask
def test_split_collection_margins(small_sky_o1_collection, split_collection):
    """Each margin holds the columns of the catalog it belongs to, over the margin's pixels."""
    original_margin = read_hats(small_sky_o1_collection / MARGIN_NAME)
    core = read_hats(split_collection / MAIN_NAME)
    extension = read_hats(split_collection / EXTENSION_NAME / EXTENSION_NAME)
    core_margin = read_hats(split_collection / MARGIN_NAME)
    extension_margin = read_hats(split_collection / EXTENSION_NAME / f"{EXTENSION_NAME}_margin")

    for margin in (core_margin, extension_margin):
        assert margin.catalog_info.catalog_type == CatalogType.MARGIN
        assert margin.catalog_info.total_rows == 47
        assert margin.catalog_info.margin_threshold == original_margin.catalog_info.margin_threshold
        assert margin.get_healpix_pixels() == original_margin.get_healpix_pixels()

    assert core_margin.schema.names == core.schema.names
    assert extension_margin.schema.names == extension.schema.names
    assert core_margin.catalog_info.catalog_name == MARGIN_NAME
    assert extension_margin.catalog_info.catalog_name == f"{EXTENSION_NAME}_margin"
    assert core_margin.catalog_info.primary_catalog == f"small_sky_collection/{MAIN_NAME}"
    assert extension_margin.catalog_info.primary_catalog == (
        f"small_sky_collection/{EXTENSION_NAME}/{EXTENSION_NAME}"
    )

    ## Margin rows are split the same way as the main catalog's rows.
    pixel = original_margin.get_healpix_pixels()[3]
    original_data = pd.read_parquet(paths.pixel_catalog_file(original_margin.catalog_path, pixel))
    margin_data = pd.read_parquet(paths.pixel_catalog_file(extension_margin.catalog_path, pixel))
    assert margin_data.columns.tolist() == extension.schema.names
    pd.testing.assert_frame_equal(
        margin_data[["_healpix_29", "ra_error", "dec_error"]],
        original_data[["_healpix_29", "ra_error", "dec_error"]],
    )


@pytest.mark.dask
def test_split_collection_empty_margin(split_collection):
    """A margin that no data falls into is still split, as an empty margin."""
    for margin_path in (
        split_collection / EMPTY_MARGIN_NAME,
        split_collection / EXTENSION_NAME / f"{EXTENSION_NAME}_margin_10arcs",
    ):
        margin = read_hats(margin_path)
        assert margin.catalog_info.catalog_type == CatalogType.MARGIN
        assert margin.catalog_info.total_rows == 0
        assert margin.catalog_info.margin_threshold == 10.0
        assert len(margin.get_healpix_pixels()) == 0

    ## The schema is written, even though there are no partitions.
    core = read_hats(split_collection / MAIN_NAME)
    assert read_hats(split_collection / EMPTY_MARGIN_NAME).schema.names == core.schema.names


@pytest.mark.dask
def test_split_collection_index(small_sky_o1_collection, split_collection):
    """An index over a column that stays in the core is carried over, unchanged."""
    original_index = read_hats(small_sky_o1_collection / INDEX_NAME)
    index = read_hats(split_collection / INDEX_NAME)

    assert index.catalog_info.catalog_type == CatalogType.INDEX
    assert index.catalog_info.catalog_name == INDEX_NAME
    assert index.catalog_info.indexing_column == "id"
    assert index.catalog_info.total_rows == original_index.catalog_info.total_rows
    assert index.catalog_info.primary_catalog == f"small_sky_collection/{MAIN_NAME}"

    original_data = pd.read_parquet(original_index.catalog_path / "dataset")
    index_data = pd.read_parquet(index.catalog_path / "dataset")
    pd.testing.assert_frame_equal(index_data, original_data)


@pytest.mark.dask
def test_split_collection_index_over_moved_column(small_sky_o1_collection, tmp_path, dask_client):
    """An index over a column that moves to the extension cannot be carried over."""
    input_collection = tmp_path / "input_collection"
    shutil.copytree(small_sky_o1_collection, input_collection)
    CollectionProperties(
        name="input_collection",
        hats_primary_table_url=MAIN_NAME,
        all_margins=[MARGIN_NAME],
        all_indexes={"ra_error": INDEX_NAME},
    ).to_properties_file(input_collection)

    with pytest.warns(UserWarning, match="not carried over"):
        args = split_args(input_collection, tmp_path)
    assert len(args.indexes) == 0

    runner.run(args, dask_client)
    collection = read_hats(tmp_path / "output" / "small_sky_collection")
    assert collection.all_indexes is None
    assert collection.all_margins == [MARGIN_NAME]
