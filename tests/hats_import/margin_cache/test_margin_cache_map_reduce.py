import os

import hats.pixel_math.healpix_shim as hp
import numpy as np
import pandas as pd
import pytest
from hats import pixel_math
from hats.io import paths
from hats.pixel_math.healpix_pixel import HealpixPixel

from hats_import.margin_cache import margin_cache_map_reduce
from hats_import.pipeline_resume_plan import get_pixel_cache_directory

keep_cols = ["weird_ra", "weird_dec"]

drop_cols = [
    "partition_order",
    "partition_pixel",
    "margin_check",
    "margin_pixel",
    "is_trunc",
]


def validate_result_dataframe(df_path, expected_len):
    res_df = pd.read_parquet(df_path)

    assert len(res_df) == expected_len

    cols = res_df.columns.values.tolist()

    for col in keep_cols:
        assert col in cols

    for col in drop_cols:
        assert col not in cols


@pytest.mark.timeout(5)
def test_to_pixel_shard_equator(tmp_path, basic_data_shard_df):
    margin_cache_map_reduce._to_pixel_shard(
        basic_data_shard_df,
        pixel=HealpixPixel(1, 21),
        margin_threshold=360.0,
        output_path=tmp_path,
        ra_column="weird_ra",
        dec_column="weird_dec",
        source_pixel=HealpixPixel(1, 0),
        fine_filtering=True,
    )

    path = tmp_path / "order_1" / "dir_0" / "pixel_21" / "dataset" / "Norder=1" / "Dir=0" / "Npix=0.parquet"

    assert os.path.exists(path)

    validate_result_dataframe(path, 360)


@pytest.mark.timeout(5)
def test_to_pixel_shard_polar(tmp_path, polar_data_shard_df):
    margin_cache_map_reduce._to_pixel_shard(
        polar_data_shard_df,
        pixel=HealpixPixel(2, 15),
        margin_threshold=360.0,
        output_path=tmp_path,
        ra_column="weird_ra",
        dec_column="weird_dec",
        source_pixel=HealpixPixel(2, 0),
        fine_filtering=True,
    )

    path = tmp_path / "order_2" / "dir_0" / "pixel_15" / "dataset" / "Norder=2" / "Dir=0" / "Npix=0.parquet"

    assert os.path.exists(path)

    validate_result_dataframe(path, 360)


def test_map_pixel_shards_error(tmp_path, capsys):
    """Test error behavior on reduce stage. e.g. by not creating the original
    catalog parquet files."""
    with pytest.raises(NotImplementedError):
        margin_cache_map_reduce.map_pixel_shards(
            paths.pixel_catalog_file(tmp_path, HealpixPixel(1, 0)),
            mapping_key="1_21",
            original_catalog_metadata="",
            margin_pair_file="",
            margin_threshold=10,
            output_path=tmp_path,
            margin_order=4,
            ra_column="ra",
            dec_column="dec",
            fine_filtering=True,
        )

    captured = capsys.readouterr()
    assert "Fine filtering temporarily removed" in captured.out


@pytest.mark.skip()
@pytest.mark.timeout(30)
def test_map_pixel_shards_fine(tmp_path, test_data_dir, small_sky_source_catalog):
    """Test basic mapping behavior, with fine filtering enabled."""
    intermediate_dir = tmp_path / "intermediate"
    os.makedirs(intermediate_dir / "mapping")
    margin_cache_map_reduce.map_pixel_shards(
        small_sky_source_catalog / "dataset" / "Norder=1" / "Dir=0" / "Npix=47.parquet",
        mapping_key="1_47",
        original_catalog_metadata=small_sky_source_catalog / "dataset" / "_common_metadata",
        margin_pair_file=test_data_dir / "margin_pairs" / "small_sky_source_pairs.csv",
        margin_threshold=3600,
        output_path=intermediate_dir,
        margin_order=3,
        ra_column="source_ra",
        dec_column="source_dec",
        fine_filtering=True,
    )

    path = (
        intermediate_dir
        / "order_2"
        / "dir_0"
        / "pixel_182"
        / "dataset"
        / "Norder=1"
        / "Dir=0"
        / "Npix=47.parquet"
    )
    assert os.path.exists(path)
    res_df = pd.read_parquet(path)
    assert len(res_df) == 107

    path = (
        intermediate_dir
        / "order_2"
        / "dir_0"
        / "pixel_185"
        / "dataset"
        / "Norder=1"
        / "Dir=0"
        / "Npix=47.parquet"
    )
    assert os.path.exists(path)
    res_df = pd.read_parquet(path)
    assert len(res_df) == 37


@pytest.mark.timeout(15)
def test_map_pixel_shards_coarse(tmp_path, test_data_dir, small_sky_source_catalog):
    """Test basic mapping behavior, without fine filtering enabled."""
    intermediate_dir = tmp_path / "intermediate"
    os.makedirs(intermediate_dir / "mapping")
    margin_cache_map_reduce.map_pixel_shards(
        small_sky_source_catalog / "dataset" / "Norder=1" / "Dir=0" / "Npix=47.parquet",
        mapping_key="1_47",
        original_catalog_metadata=small_sky_source_catalog / "dataset" / "_common_metadata",
        margin_pair_file=test_data_dir / "margin_pairs" / "small_sky_source_pairs.csv",
        margin_threshold=3600,
        output_path=intermediate_dir,
        margin_order=3,
        ra_column="source_ra",
        dec_column="source_dec",
        fine_filtering=False,
    )

    path = (
        intermediate_dir
        / "order_2"
        / "dir_0"
        / "pixel_182"
        / "dataset"
        / "Norder=1"
        / "Dir=0"
        / "Npix=47.parquet"
    )
    assert os.path.exists(path)
    res_df = pd.read_parquet(path)
    assert len(res_df) == 1386

    path = (
        intermediate_dir
        / "order_2"
        / "dir_0"
        / "pixel_185"
        / "dataset"
        / "Norder=1"
        / "Dir=0"
        / "Npix=47.parquet"
    )
    assert os.path.exists(path)
    res_df = pd.read_parquet(path)
    assert len(res_df) == 1978


def test_reduce_margin_shards(tmp_path):
    intermediate_dir = tmp_path / "intermediate"
    partition_dir = get_pixel_cache_directory(intermediate_dir, HealpixPixel(1, 21))
    shard_dir = paths.pixel_directory(partition_dir, 1, 21)

    os.makedirs(shard_dir)
    os.makedirs(intermediate_dir / "reducing")

    first_shard_path = paths.pixel_catalog_file(partition_dir, HealpixPixel(1, 0))
    second_shard_path = paths.pixel_catalog_file(partition_dir, HealpixPixel(1, 1))

    ras = np.arange(0.0, 360.0)
    dec = np.full(360, 0.0)
    norder = np.full(360, 1)
    ndir = np.full(360, 0)
    npix = np.full(360, 0)
    hats_indexes = pixel_math.compute_spatial_index(ras, dec)
    margin_order = np.full(360, 0)
    margin_dir = np.full(360, 0)
    margin_pixels = hp.radec2pix(3, ras, dec)

    test_df = pd.DataFrame(
        data=zip(hats_indexes, ras, dec, norder, ndir, npix, margin_order, margin_dir, margin_pixels),
        columns=[
            "_healpix_29",
            "weird_ra",
            "weird_dec",
            "Norder",
            "Dir",
            "Npix",
            "margin_Norder",
            "margin_Dir",
            "margin_Npix",
        ],
    )

    # Create a schema parquet file.
    schema_path = tmp_path / "metadata.parquet"
    schema_df = test_df.drop(columns=["margin_Norder", "margin_Dir", "margin_Npix"])
    schema_df.to_parquet(schema_path)

    basic_data_shard_df = test_df

    basic_data_shard_df.to_parquet(first_shard_path)
    basic_data_shard_df.to_parquet(second_shard_path)

    margin_cache_map_reduce.reduce_margin_shards(
        intermediate_dir,
        "1_21",
        tmp_path,
        1,
        21,
        original_catalog_metadata=schema_path,
        delete_intermediate_parquet_files=False,
    )

    result_path = paths.pixel_catalog_file(tmp_path, HealpixPixel(1, 21))

    validate_result_dataframe(result_path, 720)
    assert os.path.exists(shard_dir)

    # Run again with delete_intermediate_parquet_files. shard_dir doesn't exist at the end.
    margin_cache_map_reduce.reduce_margin_shards(
        intermediate_dir,
        "1_21",
        tmp_path,
        1,
        21,
        original_catalog_metadata=schema_path,
        delete_intermediate_parquet_files=True,
    )

    result_path = paths.pixel_catalog_file(tmp_path, HealpixPixel(1, 21))

    validate_result_dataframe(result_path, 720)
    assert not os.path.exists(shard_dir)


def test_reduce_margin_shards_error(tmp_path, basic_data_shard_df, capsys):
    """Test error behavior on reduce stage. e.g. by not creating the original
    catalog metadata."""
    intermediate_dir = tmp_path / "intermediate"
    partition_dir = get_pixel_cache_directory(intermediate_dir, HealpixPixel(1, 21))
    shard_dir = paths.pixel_directory(partition_dir, 1, 21)
    os.makedirs(shard_dir)
    os.makedirs(intermediate_dir / "reducing")

    # Don't write anything at the metadata path!
    schema_path = tmp_path / "metadata.parquet"

    basic_data_shard_df.to_parquet(paths.pixel_catalog_file(partition_dir, HealpixPixel(1, 0)))
    basic_data_shard_df.to_parquet(paths.pixel_catalog_file(partition_dir, HealpixPixel(1, 1)))

    with pytest.raises(FileNotFoundError):
        margin_cache_map_reduce.reduce_margin_shards(
            intermediate_dir,
            "1_21",
            tmp_path,
            1,
            21,
            original_catalog_metadata=schema_path,
            delete_intermediate_parquet_files=True,
        )

    captured = capsys.readouterr()
    assert "Parquet file does not exist" in captured.out
