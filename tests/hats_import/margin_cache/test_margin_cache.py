"""Tests of map reduce operations"""

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
from hats import read_hats
from hats.io import paths
from hats.pixel_math.healpix_pixel import HealpixPixel

import hats_import.margin_cache.margin_cache as mc
from hats_import.margin_cache.margin_cache_arguments import MarginCacheArguments


@pytest.mark.dask(timeout=150)
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

    assert all(data[paths.PARTITION_ORDER] == norder)
    assert all(data[paths.PARTITION_PIXEL] == npix)
    assert all(data[paths.PARTITION_DIR] == int(npix / 10_000) * 10_000)

    assert data.dtypes[paths.PARTITION_ORDER] == np.uint8
    assert data.dtypes[paths.PARTITION_PIXEL] == np.uint64
    assert data.dtypes[paths.PARTITION_DIR] == np.uint64

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
            "Norder",
            "Dir",
            "Npix",
            "margin_Norder",
            "margin_Dir",
            "margin_Npix",
        ],
    )

    catalog = read_hats(args.catalog_path)
    assert catalog.on_disk
    assert catalog.catalog_path == args.catalog_path


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
def test_margin_too_small(small_sky_object_catalog, tmp_path, dask_client):
    """Test that margin cache generation works end to end."""
    args = MarginCacheArguments(
        margin_threshold=10.0,
        input_catalog_path=small_sky_object_catalog,
        output_path=tmp_path,
        output_artifact_name="catalog_cache",
        margin_order=8,
        progress_bar=False,
    )

    with pytest.raises(ValueError, match="Margin cache contains no rows"):
        mc.generate_margin_cache(args, dask_client)
