"""Tests of splitting a single catalog into a collection of core and extension"""

import shutil

import nested_pandas as npd
import pyarrow.parquet as pq
import pytest
from hats import read_hats
from hats.catalog import Catalog, CatalogCollection, CatalogExtension, TableProperties
from hats.io import paths
from hats.io.validation import is_valid_collection
from hats.pixel_math import HealpixPixel
from nested_pandas.nestedframe.io import from_pyarrow

import hats_import.extension.run_split_import as runner
from hats_import.extension.arguments import ExtensionArguments
from hats_import.extension.resume_plan import ExtensionSplitPlan


def split_args(catalog_path, tmp_path, **kwargs):
    """Arguments for splitting the given catalog, with any overrides."""
    arguments = {
        "input_catalog_path": catalog_path,
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

    # The output is a collection, which lists its extension.
    collection = read_hats(collection_path)
    assert isinstance(collection, CatalogCollection)
    assert is_valid_collection(collection_path, strict=True)
    assert collection.collection_properties.hats_primary_table_url == "small_sky_order1"
    assert collection.all_extensions == ["small_sky_order1_errors"]
    assert collection.all_margins is None

    # Core catalog: a regular object catalog, with the extension columns removed.
    core = collection.main_catalog
    assert core.catalog_info.catalog_name == "small_sky_order1"
    assert core.catalog_info.catalog_type == "object"
    assert core.catalog_info.total_rows == 131
    assert core.get_healpix_pixels() == [HealpixPixel(1, pixel) for pixel in range(44, 48)]
    assert core.schema.names == ["_healpix_29", "id", "ra", "dec"]

    # Extension: a collection of its own, holding a regular object catalog with the extension
    # columns, plus the join, spatial index and coordinates.
    extension_collection = read_hats(collection_path / "small_sky_order1_errors")
    assert isinstance(extension_collection, CatalogCollection)
    assert extension_collection.all_margins is None
    extension = extension_collection.main_catalog
    assert isinstance(extension, Catalog)
    assert extension.catalog_info.catalog_name == "small_sky_order1_errors"
    assert extension.catalog_info.catalog_type == "object"
    assert extension.catalog_info.total_rows == 131
    assert extension.catalog_info.ra_column == "ra"
    assert extension.catalog_info.dec_column == "dec"
    assert extension.get_healpix_pixels() == core.get_healpix_pixels()
    assert extension.schema.names == ["_healpix_29", "object_id", "ra", "dec", "ra_error", "dec_error"]

    # Extension properties: all that is needed to load the extension and join it to the core.
    extension_properties = read_hats(collection_path / "small_sky_order1_errors.properties")
    assert isinstance(extension_properties, CatalogExtension)
    properties = extension_properties.extension_info
    assert properties.name == "small_sky_order1_errors"
    assert properties.catalog_type == "extension"
    assert properties.primary_catalog == "small_sky_with_extension"
    assert properties.primary_column == "id"
    assert properties.join_catalog == "small_sky_order1_errors"
    assert properties.join_column == "object_id"
    assert properties.extension_columns == ["ra_error", "dec_error"]
    assert properties.extension_join_style == "left"
    assert (
        extension_properties.join_catalog_dir.path == (collection_path / "small_sky_order1_errors").as_posix()
    )


@pytest.mark.dask
def test_split_metadata_files(small_sky_order1_catalog, tmp_path, dask_client):
    """The parquet metadata files and the skymaps are written for both sides."""
    args = split_args(
        small_sky_order1_catalog,
        tmp_path,
        create_thumbnail=True,
        create_per_partition_stats=True,
        skymap_alt_orders=[0],
    )
    runner.run(args, dask_client)

    for catalog_path in (args.core.catalog_path, args.extension.catalog_path):
        assert (catalog_path / "dataset" / "_common_metadata").exists()
        assert (catalog_path / "dataset" / "_metadata").exists()
        assert (catalog_path / "data_thumbnail.parquet").exists()
        assert (catalog_path / "per_partition_statistics.parquet").exists()
        assert (catalog_path / "point_map.fits").exists()
        assert (catalog_path / "skymap.fits").exists()
        assert (catalog_path / "skymap.0.fits").exists()
        catalog_info = read_hats(catalog_path).catalog_info
        assert catalog_info.skymap_order == 1
        assert catalog_info.skymap_alt_orders == [0]


@pytest.mark.dask
def test_split_summary_files(small_sky_order1_catalog, tmp_path, dask_client):
    """The optional visual and summary files are written for both sides."""
    pytest.importorskip("matplotlib.pyplot")

    args = split_args(
        small_sky_order1_catalog,
        tmp_path,
        create_skymap_png=True,
        create_partition_info_png=True,
        create_summary_html=True,
        create_summary_md=True,
    )
    runner.run(args, dask_client)

    for catalog_path in (args.core.catalog_path, args.extension.catalog_path):
        for file_name in ["skymap.png", "partition_info.png", "index.html", "README.md"]:
            assert (catalog_path / file_name).exists()


@pytest.mark.dask
def test_split_npix_as_directory(small_sky_source_npix_dir_catalog, tmp_path, dask_client):
    """The output catalogs use the args suffix, not the input catalog suffix."""
    args = ExtensionArguments(
        input_catalog_path=small_sky_source_npix_dir_catalog,
        extension_columns=["mag", "band"],
        primary_column="source_id",
        extension_name="photometry",
        output_path=tmp_path,
        output_artifact_name="small_sky_collection",
        progress_bar=False,
    )
    assert read_hats(small_sky_source_npix_dir_catalog).catalog_info.npix_suffix == "/"
    runner.run(args, dask_client)

    for catalog_path in (args.core.catalog_path, args.extension.catalog_path):
        catalog = read_hats(catalog_path)
        assert catalog.catalog_info.npix_suffix == args.npix_suffix == ".parquet"
        pixel_file = paths.pixel_catalog_file(catalog_path, catalog.get_healpix_pixels()[0])
        assert pixel_file.is_file()


@pytest.mark.dask
def test_split_default_columns(small_sky_order1_catalog, tmp_path, dask_client):
    """The core keeps the default columns of the input that it holds. The extension has none,
    so all of its columns are loaded by default."""
    input_path = tmp_path / "input"
    shutil.copytree(small_sky_order1_catalog, input_path)
    TableProperties.read_from_dir(input_path).copy_and_update(
        default_columns=["id", "ra", "dec", "ra_error"]
    ).to_properties_file(input_path)

    args = split_args(input_path, tmp_path, output_path=tmp_path / "output")
    runner.run(args, dask_client)

    core = read_hats(args.core.catalog_path)
    assert core.catalog_info.default_columns == ["id", "ra", "dec"]
    extension = read_hats(args.extension.catalog_path)
    assert extension.catalog_info.default_columns is None


def test_split_pixel(small_sky_order1_catalog, tmp_path):
    """One input partition is written to both sides, each with its own columns."""
    args = split_args(small_sky_order1_catalog, tmp_path)
    resume_path = ExtensionSplitPlan(args).tmp_path
    pixel = HealpixPixel(1, 44)

    runner.split_pixel(pixel, args, args.input_catalog, resume_path)

    core_data = npd.read_parquet(paths.pixel_catalog_file(args.core.catalog_path, pixel))
    extension_data = npd.read_parquet(paths.pixel_catalog_file(args.extension.catalog_path, pixel))
    assert list(core_data.columns) == args.core.output_columns
    assert list(extension_data.columns) == args.extension.output_columns
    assert len(core_data) == len(extension_data) == 42


def test_split_row_groups(small_sky_order1_catalog, tmp_path):
    """Each output file keeps the row groups of the input file, unless `row_group_kwargs` is
    given, in which case it is split into row groups of the given size."""
    pixels = [HealpixPixel(1, pixel) for pixel in range(44, 48)]

    # Each input partition is a single row group.
    input_row_groups = {
        HealpixPixel(1, 44): [42],
        HealpixPixel(1, 45): [29],
        HealpixPixel(1, 46): [42],
        HealpixPixel(1, 47): [18],
    }
    assert _row_group_sizes(small_sky_order1_catalog, pixels) == input_row_groups

    # Without `row_group_kwargs`, the row groups do not change.
    args = split_args(small_sky_order1_catalog, tmp_path / "unchanged")
    resume_path = ExtensionSplitPlan(args).tmp_path
    for pixel in pixels:
        runner.split_pixel(pixel, args, args.input_catalog, resume_path)
    assert _row_group_sizes(args.core.catalog_path, pixels) == input_row_groups
    assert _row_group_sizes(args.extension.catalog_path, pixels) == input_row_groups

    # With `row_group_kwargs`, each output file is split into row groups of 10 rows.
    args = split_args(small_sky_order1_catalog, tmp_path / "split", row_group_kwargs={"num_rows": 10})
    resume_path = ExtensionSplitPlan(args).tmp_path
    for pixel in pixels:
        runner.split_pixel(pixel, args, args.input_catalog, resume_path)

    # The partitions hold different numbers of rows, so a different number of row groups.
    expected_row_groups = {
        HealpixPixel(1, 44): [10, 10, 10, 10, 2],
        HealpixPixel(1, 45): [10, 10, 9],
        HealpixPixel(1, 46): [10, 10, 10, 10, 2],
        HealpixPixel(1, 47): [10, 8],
    }
    assert _row_group_sizes(args.core.catalog_path, pixels) == expected_row_groups
    assert _row_group_sizes(args.extension.catalog_path, pixels) == expected_row_groups


def _row_group_sizes(catalog_path, pixels):
    """Row counts of every row group of the given partitions of a catalog."""
    sizes = {}
    for pixel in pixels:
        metadata = pq.ParquetFile(paths.pixel_catalog_file(catalog_path, pixel)).metadata
        sizes[pixel] = [metadata.row_group(index).num_rows for index in range(metadata.num_row_groups)]
    return sizes


def test_split_pixel_failure(small_sky_order1_catalog, tmp_path, capsys):
    """A partition that cannot be split raises an error."""
    args = split_args(small_sky_order1_catalog, tmp_path)
    with pytest.raises(FileNotFoundError):
        runner.split_pixel(HealpixPixel(1, 0), args, args.input_catalog, args.resume_tmp)
    assert "Failed SPLITTING stage" in capsys.readouterr().out


@pytest.mark.dask
def test_split_row_count_mismatch(small_sky_order1_catalog, tmp_path, dask_client):
    """The rows written must add up to the row count of the input table."""
    input_path = tmp_path / "input"
    shutil.copytree(small_sky_order1_catalog, input_path)

    props = TableProperties.read_from_dir(input_path).copy_and_update(total_rows=100)
    props.to_properties_file(input_path)

    args = split_args(input_path, tmp_path, output_path=tmp_path / "output")
    with pytest.raises(ValueError, match="does not match"):
        runner.run(args, dask_client)


@pytest.mark.dask
def test_split_failure_stops_the_run(small_sky_order1_catalog, tmp_path, dask_client):
    """A partition that fails to split on a worker stops the run with its error."""
    input_path = tmp_path / "input"
    shutil.copytree(small_sky_order1_catalog, input_path)
    paths.pixel_catalog_file(input_path, HealpixPixel(1, 44)).unlink()

    args = split_args(input_path, tmp_path, output_path=tmp_path / "output")
    with pytest.raises(FileNotFoundError):
        runner.run(args, dask_client)


@pytest.mark.dask
def test_split_nested_column(small_sky_nested_catalog, tmp_path, dask_client):
    """A nested column moves to the extension whole."""
    args = split_args(
        small_sky_nested_catalog, tmp_path, extension_columns=["lc"], extension_name="lightcurves"
    )
    runner.run(args, dask_client)

    core = read_hats(args.core.catalog_path)
    extension = read_hats(args.extension.catalog_path)
    assert "lc" not in core.schema.names
    extension_frame = from_pyarrow(extension.schema.empty_table())
    assert extension_frame.nested_columns == ["lc"]
    original_frame = from_pyarrow(read_hats(small_sky_nested_catalog).schema.empty_table())
    assert extension_frame.get_subcolumns() == original_frame.get_subcolumns()
