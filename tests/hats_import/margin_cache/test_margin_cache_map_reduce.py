import os

import numpy as np
import pandas as pd
import pyarrow as pa
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
        pa.Table.from_pandas(basic_data_shard_df),
        pixel=HealpixPixel(1, 21),
        output_path=tmp_path,
        source_pixel=HealpixPixel(1, 0),
    )

    path = tmp_path / "order_1" / "dir_0" / "pixel_21" / "dataset" / "Norder=1" / "Dir=0" / "Npix=0.parquet"

    assert os.path.exists(path)

    validate_result_dataframe(path, 360)


@pytest.mark.timeout(5)
def test_to_pixel_shard_polar(tmp_path, polar_data_shard_df):
    margin_cache_map_reduce._to_pixel_shard(
        pa.Table.from_pandas(polar_data_shard_df),
        pixel=HealpixPixel(2, 15),
        output_path=tmp_path,
        source_pixel=HealpixPixel(2, 0),
    )

    path = tmp_path / "order_2" / "dir_0" / "pixel_15" / "dataset" / "Norder=2" / "Dir=0" / "Npix=0.parquet"

    assert os.path.exists(path)

    validate_result_dataframe(path, 360)


def test_map_pixel_shards_error(tmp_path, capsys):
    """Test error behavior on reduce stage. e.g. by not creating the original
    catalog parquet files."""
    with pytest.raises(FileNotFoundError):
        margin_cache_map_reduce.map_pixel_shards(
            paths.pixel_catalog_file(tmp_path, HealpixPixel(1, 0)),
            mapping_key="1_21",
            source_pixel=HealpixPixel(1, 0),
            original_catalog_metadata="",
            margin_pair_file="",
            output_path=tmp_path,
            margin_order=4,
        )

    captured = capsys.readouterr()
    assert "Parquet file does not exist" in captured.out


@pytest.mark.timeout(15)
def test_map_pixel_shards_coarse(tmp_path, test_data_dir, small_sky_source_catalog):
    """Test basic mapping behavior, without fine filtering enabled."""
    intermediate_dir = tmp_path / "intermediate"
    os.makedirs(intermediate_dir / "mapping")
    margin_cache_map_reduce.map_pixel_shards(
        paths.pixel_catalog_file(small_sky_source_catalog, HealpixPixel(1, 47)),
        source_pixel=HealpixPixel(1, 47),
        mapping_key="1_47",
        original_catalog_metadata=small_sky_source_catalog / "dataset" / "_common_metadata",
        margin_pair_file=test_data_dir / "margin_pairs" / "small_sky_source_pairs.csv",
        output_path=intermediate_dir,
        margin_order=3,
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
    hats_indexes = pixel_math.compute_spatial_index(ras, dec)

    basic_data_shard_df = pd.DataFrame(
        data=zip(hats_indexes, ras, dec),
        columns=[
            "_healpix_29",
            "weird_ra",
            "weird_dec",
        ],
    )

    basic_data_shard_df.to_parquet(first_shard_path)
    basic_data_shard_df.to_parquet(second_shard_path)

    margin_cache_map_reduce.reduce_margin_shards(
        intermediate_dir,
        "1_21",
        tmp_path,
        1,
        21,
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
        delete_intermediate_parquet_files=True,
    )

    result_path = paths.pixel_catalog_file(tmp_path, HealpixPixel(1, 21))

    validate_result_dataframe(result_path, 720)
    assert not os.path.exists(shard_dir)
