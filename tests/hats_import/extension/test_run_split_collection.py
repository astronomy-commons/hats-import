"""Tests of splitting a catalog collection into a collection of core and extension"""

import shutil

import nested_pandas as npd
import pytest
from hats import read_hats
from hats.catalog import CatalogCollection, CatalogExtension, CatalogType, CollectionProperties
from hats.io import paths
from hats.io.validation import is_valid_catalog, is_valid_collection
from hats.pixel_math import HealpixPixel

import hats_import.extension.run_split_import as runner
from hats_import.extension.arguments import ExtensionArguments


def split_args(input_collection, tmp_path, **kwargs):
    """Arguments for splitting the input collection, with any overrides."""
    arguments = {
        "input_catalog_path": input_collection,
        "extension_columns": ["ra_error", "dec_error"],
        "primary_column": "id",
        "join_column": "object_id",
        "extension_name": "errors",
        "output_path": tmp_path / "output",
        "output_artifact_name": "small_sky_with_extension",
        "progress_bar": False,
    }
    return ExtensionArguments(**(arguments | kwargs))


@pytest.mark.dask
def test_split_collection(small_sky_o1_collection, tmp_path, dask_client):
    """The output is a collection, holding the core, its margins, its index and the extension."""
    runner.run(split_args(small_sky_o1_collection, tmp_path), dask_client)
    split_collection = tmp_path / "output" / "small_sky_with_extension"
    original = read_hats(small_sky_o1_collection)

    collection = read_hats(split_collection)
    assert isinstance(collection, CatalogCollection)
    assert is_valid_collection(split_collection, strict=True)

    # The core and the collection members of the input keep their names.
    assert collection.collection_properties.hats_primary_table_url == "small_sky_order1"
    assert collection.all_margins == ["small_sky_order1_margin", "small_sky_order1_margin_10arcs"]
    assert collection.default_margin == "small_sky_order1_margin"
    assert collection.all_indexes == {"id": "small_sky_order1_id_index"}
    assert collection.default_index_field == "id"
    # Only the extension is new.
    assert collection.all_extensions == ["small_sky_order1_errors"]

    core = collection.main_catalog
    assert core.catalog_info.catalog_name == "small_sky_order1"
    assert core.catalog_info.catalog_type == CatalogType.OBJECT
    assert core.schema.names == ["_healpix_29", "id", "ra", "dec"]
    assert core.get_healpix_pixels() == original.main_catalog.get_healpix_pixels()

    # The extension is a collection of its own, holding its margins, which are named after it.
    extension_path = split_collection / "small_sky_order1_errors"
    extension_collection = read_hats(extension_path)
    assert isinstance(extension_collection, CatalogCollection)
    assert is_valid_collection(extension_path, strict=True)
    assert extension_collection.collection_properties.hats_primary_table_url == "small_sky_order1_errors"
    assert extension_collection.all_margins == [
        "small_sky_order1_errors_margin",
        "small_sky_order1_errors_margin_10arcs",
    ]
    assert extension_collection.default_margin == "small_sky_order1_errors_margin"
    # The index stays in the core's collection, as it indexes a column that stays in the core.
    assert extension_collection.all_indexes is None

    extension = extension_collection.main_catalog
    assert extension.catalog_info.catalog_name == "small_sky_order1_errors"
    assert extension.catalog_info.catalog_type == CatalogType.OBJECT
    assert extension.schema.names == ["_healpix_29", "object_id", "ra", "dec", "ra_error", "dec_error"]
    assert extension.get_healpix_pixels() == original.main_catalog.get_healpix_pixels()

    # Extension properties: all that is needed to load the extension and join it to the core.
    catalog_extension = read_hats(split_collection / "small_sky_order1_errors.properties")
    assert isinstance(catalog_extension, CatalogExtension)
    properties = catalog_extension.extension_info
    assert properties.name == "small_sky_order1_errors"
    assert properties.catalog_type == CatalogType.EXTENSION
    assert properties.primary_catalog == "small_sky_with_extension"
    assert properties.primary_column == "id"
    assert properties.join_catalog == "small_sky_order1_errors"
    assert properties.join_column == "object_id"
    assert properties.extension_columns == ["ra_error", "dec_error"]
    assert properties.extension_join_style == "left"
    assert properties.shares_primary_coordinates is True
    assert catalog_extension.join_catalog_dir.path == extension_path.as_posix()


@pytest.mark.dask
def test_split_collection_margins(small_sky_o1_collection, tmp_path, dask_client):
    """Each margin holds the columns of the catalog it belongs to, over the margin's pixels."""
    runner.run(split_args(small_sky_o1_collection, tmp_path), dask_client)
    split_collection = tmp_path / "output" / "small_sky_with_extension"
    original_margin = read_hats(small_sky_o1_collection / "small_sky_order1_margin")
    core_margin = read_hats(split_collection / "small_sky_order1_margin")
    extension_margin = read_hats(
        split_collection / "small_sky_order1_errors" / "small_sky_order1_errors_margin"
    )

    for margin in (core_margin, extension_margin):
        assert margin.catalog_info.catalog_type == CatalogType.MARGIN
        assert margin.catalog_info.total_rows == 47
        assert margin.catalog_info.margin_threshold == original_margin.catalog_info.margin_threshold
        assert margin.get_healpix_pixels() == original_margin.get_healpix_pixels()

    assert core_margin.schema.names == ["_healpix_29", "id", "ra", "dec"]
    assert extension_margin.schema.names == ["_healpix_29", "object_id", "ra", "dec", "ra_error", "dec_error"]
    assert core_margin.catalog_info.catalog_name == "small_sky_order1_margin"
    assert extension_margin.catalog_info.catalog_name == "small_sky_order1_errors_margin"
    assert core_margin.catalog_info.primary_catalog == "small_sky_with_extension/small_sky_order1"
    assert extension_margin.catalog_info.primary_catalog == (
        "small_sky_with_extension/small_sky_order1_errors/small_sky_order1_errors"
    )


@pytest.mark.dask
def test_split_collection_margin_data(small_sky_o1_collection, tmp_path, dask_client):
    """The rows of a margin pixel are split the same way as the main catalog's rows."""
    runner.run(split_args(small_sky_o1_collection, tmp_path), dask_client)
    split_collection = tmp_path / "output" / "small_sky_with_extension"
    original_margin = read_hats(small_sky_o1_collection / "small_sky_order1_margin")
    pixel = HealpixPixel(1, 44)
    original_data = npd.read_parquet(paths.pixel_catalog_file(original_margin.catalog_path, pixel))

    # The core margin keeps the columns that stay in the core.
    core_margin_path = split_collection / "small_sky_order1_margin"
    core_data = npd.read_parquet(paths.pixel_catalog_file(core_margin_path, pixel))
    assert core_data.columns.tolist() == ["_healpix_29", "id", "ra", "dec"]
    assert core_data.equals(original_data[["_healpix_29", "id", "ra", "dec"]])

    # The extension margin holds the extension columns, with the join key renamed.
    extension_margin_path = split_collection / "small_sky_order1_errors" / "small_sky_order1_errors_margin"
    extension_data = npd.read_parquet(paths.pixel_catalog_file(extension_margin_path, pixel))
    assert extension_data.columns.tolist() == [
        "_healpix_29",
        "object_id",
        "ra",
        "dec",
        "ra_error",
        "dec_error",
    ]
    assert extension_data.rename(columns={"object_id": "id"}).equals(
        original_data[["_healpix_29", "id", "ra", "dec", "ra_error", "dec_error"]],
    )


@pytest.mark.dask
def test_split_collection_empty_margin(small_sky_o1_collection, tmp_path, dask_client):
    """An empty margin is also split."""
    runner.run(split_args(small_sky_o1_collection, tmp_path), dask_client)
    split_collection = tmp_path / "output" / "small_sky_with_extension"
    extension_path = split_collection / "small_sky_order1_errors"
    for catalog_path, margin_path in (
        (split_collection / "small_sky_order1", split_collection / "small_sky_order1_margin_10arcs"),
        (
            extension_path / "small_sky_order1_errors",
            extension_path / "small_sky_order1_errors_margin_10arcs",
        ),
    ):
        margin = read_hats(margin_path)
        assert margin.catalog_info.catalog_type == CatalogType.MARGIN
        assert margin.catalog_info.total_rows == 0
        assert margin.catalog_info.margin_threshold == 10.0
        assert len(margin.get_healpix_pixels()) == 0
        # The schema of its catalog is written, even though there are no partitions.
        assert margin.schema == read_hats(catalog_path).schema


@pytest.mark.dask
def test_split_collection_index(small_sky_o1_collection, tmp_path, dask_client):
    """An index over a column that stays in the core is carried over, unchanged."""
    runner.run(split_args(small_sky_o1_collection, tmp_path), dask_client)
    split_collection = tmp_path / "output" / "small_sky_with_extension"
    original_index = read_hats(small_sky_o1_collection / "small_sky_order1_id_index")
    index = read_hats(split_collection / "small_sky_order1_id_index")

    assert index.catalog_info.catalog_type == CatalogType.INDEX
    assert index.catalog_info.catalog_name == "small_sky_order1_id_index"
    assert index.catalog_info.indexing_column == "id"
    assert index.catalog_info.total_rows == original_index.catalog_info.total_rows
    assert index.catalog_info.primary_catalog == "small_sky_with_extension/small_sky_order1"

    original_data = npd.read_parquet(original_index.catalog_path / "dataset")
    index_data = npd.read_parquet(index.catalog_path / "dataset")
    assert index_data.equals(original_data)


@pytest.mark.dask
def test_split_collection_index_over_moved_column(small_sky_o1_collection, tmp_path, dask_client):
    """An index over a column that moves to the extension goes to the extension's collection."""
    input_collection = tmp_path / "input_collection"
    shutil.copytree(small_sky_o1_collection, input_collection)
    # The files of the "id" index stand in for an index over "ra_error": only where it goes is checked.
    CollectionProperties(
        name="input_collection",
        hats_primary_table_url="small_sky_order1",
        all_indexes={"ra_error": "small_sky_order1_id_index"},
        default_index="ra_error",
    ).to_properties_file(input_collection)

    args = split_args(input_collection, tmp_path)
    assert len(args.core.indexes) == 0
    assert list(args.extension.indexes) == ["ra_error"]

    runner.run(args, dask_client)
    collection = read_hats(args.catalog_path)
    assert collection.all_indexes is None
    assert collection.default_index_field is None

    extension_collection = read_hats(args.extension.collection_path)
    # The index is renamed after the extension.
    assert extension_collection.all_indexes == {"ra_error": "small_sky_order1_errors_id_index"}
    assert extension_collection.default_index_field == "ra_error"

    # The index now points at the extension catalog.
    index_path = args.extension.collection_path / "small_sky_order1_errors_id_index"
    assert is_valid_catalog(index_path)
    index = read_hats(index_path)
    assert index.catalog_info.catalog_name == "small_sky_order1_errors_id_index"
    assert index.catalog_info.primary_catalog == (
        "small_sky_with_extension/small_sky_order1_errors/small_sky_order1_errors"
    )
