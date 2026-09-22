"""Tests of splitting a single catalog into a collection of core and extension"""

import shutil

import pandas as pd
import pyarrow.parquet as pq
import pytest
from hats import read_hats
from hats.catalog import CatalogCollection, ExtensionCatalog
from hats.io import paths
from hats.io.validation import is_valid_collection
from hats.pixel_math import HealpixPixel

import hats_import.extension.run_split_import as runner
from hats_import.extension.arguments import ExtensionArguments

CORE_NAME = "small_sky_order1"
EXTENSION_NAME = "small_sky_order1_errors"
PIXELS = [HealpixPixel(1, pixel) for pixel in range(44, 48)]


def split_args(small_sky_order1_catalog, tmp_path, **kwargs):
    """Arguments for splitting the small sky catalog, with any overrides."""
    arguments = {
        "input_catalog_path": small_sky_order1_catalog,
        "extension_columns": ["ra_error", "dec_error"],
        "primary_column": "id",
        "join_column": "object_id",
        "extension_name": "errors",
        "output_path": tmp_path,
        "output_artifact_name": "small_sky_with_extension",
        "progress_bar": False,
    }
    return ExtensionArguments(**(arguments | kwargs))


def test_bad_args():
    """Runner should fail with empty/mistyped arguments"""
    with pytest.raises(TypeError, match="ExtensionArguments"):
        runner.run(None, None)
    args = {"output_artifact_name": "bad_arg_type"}
    with pytest.raises(TypeError, match="ExtensionArguments"):
        runner.run(args, None)


@pytest.mark.dask
def test_split_small_sky(small_sky_order1_catalog, tmp_path, dask_client):
    """A single catalog is split into a collection holding the core and the extension."""
    runner.run(split_args(small_sky_order1_catalog, tmp_path), dask_client)
    collection_path = tmp_path / "small_sky_with_extension"

    ## The output is a collection, which lists its extension.
    collection = read_hats(collection_path)
    assert isinstance(collection, CatalogCollection)
    assert is_valid_collection(collection_path, strict=True)
    assert collection.collection_properties.hats_primary_table_url == CORE_NAME
    assert collection.all_extensions == [EXTENSION_NAME]
    ## With no margins to hold, the extension is a catalog, not a collection of its own.
    assert collection.all_margins is None
    assert not (collection_path / EXTENSION_NAME / "collection.properties").exists()

    ## Core catalog: a regular object catalog, with the extension columns removed.
    core = collection.main_catalog
    assert core.catalog_info.catalog_name == CORE_NAME
    assert core.catalog_info.catalog_type == "object"
    assert core.catalog_info.total_rows == 131
    assert core.get_healpix_pixels() == PIXELS
    assert core.schema.names == ["_healpix_29", "id", "ra", "dec"]
    assert (collection_path / CORE_NAME / "skymap.fits").exists()

    ## Extension catalog: the extension columns, plus the join, spatial index and coordinates.
    extension = read_hats(collection_path / EXTENSION_NAME)
    assert isinstance(extension, ExtensionCatalog)
    properties = extension.catalog_info
    assert properties.catalog_name == EXTENSION_NAME
    assert properties.catalog_type == "extension"
    assert properties.total_rows == 131
    assert properties.primary_column == "id"
    assert properties.join_column == "object_id"
    assert properties.extension_columns == ["ra_error", "dec_error"]
    assert properties.extension_join_style == "left"
    assert properties.ra_column == "ra"
    assert properties.dec_column == "dec"
    assert extension.get_healpix_pixels() == PIXELS
    assert extension.schema.names == ["_healpix_29", "object_id", "ra", "dec", "ra_error", "dec_error"]
    assert (collection_path / EXTENSION_NAME / "skymap.fits").exists()

    ## References are relative to the directory holding the collection.
    assert properties.primary_catalog == f"small_sky_with_extension/{CORE_NAME}"
    assert properties.join_catalog == f"small_sky_with_extension/{EXTENSION_NAME}"
    assert read_hats(tmp_path / properties.primary_catalog).catalog_info.catalog_name == CORE_NAME


@pytest.mark.dask
def test_split_small_sky_data(small_sky_order1_catalog, tmp_path, dask_client):
    """Every partition of the core and the extension together holds the input's data."""
    runner.run(split_args(small_sky_order1_catalog, tmp_path), dask_client)
    collection_path = tmp_path / "small_sky_with_extension"
    properties = read_hats(collection_path / EXTENSION_NAME).catalog_info

    for pixel in PIXELS:
        original_data = pd.read_parquet(paths.pixel_catalog_file(small_sky_order1_catalog, pixel))
        core_data = pd.read_parquet(paths.pixel_catalog_file(collection_path / CORE_NAME, pixel))
        extension_data = pd.read_parquet(paths.pixel_catalog_file(collection_path / EXTENSION_NAME, pixel))

        # The healpix and coordinates are copied into the extension.
        pd.testing.assert_frame_equal(
            extension_data[["_healpix_29", "ra", "dec"]], core_data[["_healpix_29", "ra", "dec"]]
        )

        # Joining the extension back to the core recovers the original catalog.
        joined = core_data.merge(
            extension_data[[properties.join_column] + properties.extension_columns],
            left_on=properties.primary_column,
            right_on=properties.join_column,
        ).drop(columns=properties.join_column)[original_data.columns]

        pd.testing.assert_frame_equal(joined, original_data)


@pytest.mark.dask
def test_split_preserves_row_groups(small_sky_order1_catalog, tmp_path, dask_client):
    """The row groups of each input file are kept in both outputs."""
    input_path = tmp_path / "input"

    # Set row groups of size 10 in the input catalog.
    shutil.copytree(small_sky_order1_catalog, input_path)
    for pixel in PIXELS:
        pixel_file = paths.pixel_catalog_file(input_path, pixel)
        pq.write_table(pq.read_table(pixel_file), pixel_file, row_group_size=10)

    # The partitions hold 42, 29, 42 and 18 rows, split with
    # a different number of row groups.
    expected_row_groups = {
        HealpixPixel(1, 44): [10, 10, 10, 10, 2],
        HealpixPixel(1, 45): [10, 10, 9],
        HealpixPixel(1, 46): [10, 10, 10, 10, 2],
        HealpixPixel(1, 47): [10, 8],
    }
    assert _row_group_sizes(input_path) == expected_row_groups

    args = split_args(input_path, tmp_path, output_path=tmp_path / "output")
    runner.run(args, dask_client)

    assert _row_group_sizes(args.core.catalog_path) == expected_row_groups
    assert _row_group_sizes(args.extension.catalog_path) == expected_row_groups


def _row_group_sizes(catalog_path):
    """Row counts of every row group of every partition of a catalog."""
    sizes = {}
    for pixel in PIXELS:
        metadata = pq.ParquetFile(paths.pixel_catalog_file(catalog_path, pixel)).metadata
        sizes[pixel] = [metadata.row_group(index).num_rows for index in range(metadata.num_row_groups)]
    return sizes
