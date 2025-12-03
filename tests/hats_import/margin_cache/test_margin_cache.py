"""Tests of map reduce operations"""

import os

import numpy.testing as npt
import pandas as pd
import pytest
from hats import read_hats
from hats.io import get_parquet_metadata_pointer, paths
from hats.io.file_io import read_parquet_metadata
from hats.pixel_math.healpix_pixel import HealpixPixel

import hats_import.margin_cache.margin_cache as mc
from hats_import.margin_cache.margin_cache_arguments import MarginCacheArguments


@pytest.mark.dask(timeout=150)
@pytest.mark.parametrize(
    "small_sky_source_catalog",
    ["small_sky_source_catalog", "small_sky_source_npix_dir_catalog"],
    indirect=True,
)
def test_margin_cache_gen(small_sky_source_catalog, tmp_path, dask_client):
    """Test that margin cache generation works end to end."""
    args = MarginCacheArguments(
        margin_threshold=180.0,
        input_catalog_path=small_sky_source_catalog,
        output_path=tmp_path,
        output_artifact_name="catalog_cache",
        margin_order=8,
        progress_bar=False,
    )

    assert args.catalog.catalog_info.ra_column == "source_ra"

    mc.generate_margin_cache(args, dask_client)

    norder = 1
    npix = 47

    test_file = paths.pixel_catalog_file(args.catalog_path, HealpixPixel(norder, npix))

    data = pd.read_parquet(test_file)

    assert len(data) == 88

    npt.assert_array_equal(
        data.columns,
        [
            "_healpix_29",
            "source_id",
            "source_ra",
            "source_dec",
            "mjd",
            "mag",
            "band",
            "object_id",
            "object_ra",
            "object_dec",
        ],
    )

    catalog = read_hats(args.catalog_path)
    assert catalog.on_disk
    assert catalog.catalog_path == args.catalog_path

    # Check that the data thumbnail does not exist. It should only exist for
    # main object/source catalogs.
    assert not (args.catalog_path / "dataset" / "data_thumbnail.parquet").exists()


@pytest.mark.dask(timeout=150)
def test_margin_cache_gen_negative_pixels(small_sky_source_catalog, tmp_path, dask_client):
    """Test that margin cache generation can generate a file for a negative pixel."""
    args = MarginCacheArguments(
        margin_threshold=3600.0,
        input_catalog_path=small_sky_source_catalog,
        output_path=tmp_path,
        output_artifact_name="catalog_cache",
        margin_order=3,
        progress_bar=False,
        fine_filtering=False,
    )

    assert args.catalog.catalog_info.ra_column == "source_ra"

    mc.generate_margin_cache(args, dask_client)

    norder = 0
    npix = 7

    negative_test_file = paths.pixel_catalog_file(args.catalog_path, HealpixPixel(norder, npix))

    negative_data = pd.read_parquet(negative_test_file)

    assert len(negative_data) > 0


@pytest.mark.dask(timeout=150)
def test_generate_empty_margin_catalog(small_sky_object_catalog, tmp_path, dask_client):
    """Test that margin cache generation works with empty catalogs."""
    args = MarginCacheArguments(
        margin_threshold=10.0,
        input_catalog_path=small_sky_object_catalog,
        output_path=tmp_path,
        output_artifact_name="catalog_cache",
        progress_bar=False,
    )

    mc.generate_margin_cache(args, dask_client)

    # Verify that an empty catalog was created with the correct metadata
    catalog = read_hats(args.catalog_path)
    object_cat = read_hats(small_sky_object_catalog)
    assert catalog.on_disk
    assert catalog.catalog_path == args.catalog_path
    assert catalog.catalog_info.total_rows == 0
    assert catalog.catalog_info.hats_order == args.catalog.catalog_info.hats_order
    assert len(catalog.get_healpix_pixels()) == 0
    assert len(catalog.pixel_tree) == 0
    assert catalog.catalog_info.ra_column == "ra"
    assert catalog.catalog_info.dec_column == "dec"
    assert catalog.catalog_info.margin_threshold == 10.0
    assert catalog.schema == object_cat.schema

    # Check that the metadata files exist
    assert (args.catalog_path / "partition_info.csv").exists()
    assert (args.catalog_path / "hats.properties").exists()

    metadata_path = args.catalog_path / "dataset" / "_metadata"
    common_metadata_path = args.catalog_path / "dataset" / "_common_metadata"
    assert metadata_path.exists()
    assert common_metadata_path.exists()

    # Verify that the catalog contains no data files
    data_files = [
        f
        for f in (args.catalog_path / "dataset").rglob("*")
        if f.is_file() and f not in (metadata_path, common_metadata_path)
    ]
    assert len(list(data_files)) == 0

    # Check both pyarrow metadata files are correct
    metadata = read_parquet_metadata(metadata_path)
    assert metadata.num_rows == 0
    assert metadata.num_row_groups == 0
    assert metadata.schema.to_arrow_schema() == object_cat.schema

    common_metadata = read_parquet_metadata(common_metadata_path)
    assert common_metadata.num_rows == 0
    assert common_metadata.num_row_groups == 0
    assert common_metadata.schema.to_arrow_schema() == object_cat.schema


@pytest.mark.dask(timeout=150)
def test_margin_gen_nested_catalog(small_sky_nested_catalog, tmp_path, dask_client):
    args = MarginCacheArguments(
        margin_order=8,
        input_catalog_path=small_sky_nested_catalog,
        output_path=tmp_path,
        output_artifact_name="catalog_cache",
        progress_bar=False,
    )

    mc.generate_margin_cache(args, dask_client)

    metadata_path = get_parquet_metadata_pointer(args.catalog_path)
    metadata = read_parquet_metadata(metadata_path)
    assert metadata.num_rows == 13

    norder = 2
    npix = 178

    test_file = paths.pixel_catalog_file(args.catalog_path, HealpixPixel(norder, npix))

    data = pd.read_parquet(test_file)

    assert len(data) == 2

    npt.assert_array_equal(
        data.columns,
        [
            "id",
            "ra",
            "dec",
            "ra_error",
            "dec_error",
            "lc",
            "_healpix_29",
        ],
    )

    catalog = read_hats(args.catalog_path)
    assert catalog.on_disk
    assert catalog.catalog_path == args.catalog_path


@pytest.mark.dask(timeout=150)
def test_margin_cache_omit_metadata(small_sky_source_catalog, tmp_path, dask_client):
    args = MarginCacheArguments(
        margin_threshold=180.0,
        input_catalog_path=small_sky_source_catalog,
        output_path=tmp_path,
        output_artifact_name="catalog_cache",
        margin_order=8,
        progress_bar=False,
        create_metadata=False,
    )
    mc.generate_margin_cache(args, dask_client)

    # Check that the catalog was made correctly, but the metadata file does NOT exist
    catalog = read_hats(args.catalog_path)
    assert catalog.catalog_info.total_rows == 1717
    assert catalog.on_disk
    metadata_filename = os.path.join(args.catalog_path, "dataset", "_metadata")
    assert not os.path.exists(metadata_filename)
    assert catalog.catalog_path == args.catalog_path
