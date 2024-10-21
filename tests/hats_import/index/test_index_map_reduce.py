"""Tests of map reduce operations"""

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

import hats_import.index.map_reduce as mr
from hats_import.index.arguments import IndexArguments


@pytest.mark.dask
def test_create_index(
    small_sky_object_catalog,
    assert_parquet_file_index,
    tmp_path,
    dask_client,
):
    """Create an index for simple object catalog"""
    args = IndexArguments(
        input_catalog_path=small_sky_object_catalog,
        indexing_column="id",
        output_path=tmp_path,
        output_artifact_name="small_sky_object_index",
        progress_bar=False,
    )
    mr.create_index(args, dask_client)

    output_file = tmp_path / "small_sky_object_index" / "dataset" / "index" / "part.0.parquet"

    expected_ids = [*range(700, 831)]
    assert_parquet_file_index(output_file, expected_ids)

    data_frame = pd.read_parquet(output_file, engine="pyarrow")
    npt.assert_array_equal(
        data_frame.columns,
        ["_healpix_29", "Norder", "Dir", "Npix"],
    )
    assert data_frame.index.name == "id"
    assert (data_frame["Norder"] == 0).all()


@pytest.mark.dask
def test_create_index_no_healpix_29(small_sky_object_catalog, tmp_path, dask_client):
    """Create an index for simple object catalog, without the _healpix_29 field."""
    args = IndexArguments(
        input_catalog_path=small_sky_object_catalog,
        indexing_column="id",
        include_healpix_29=False,
        output_path=tmp_path,
        output_artifact_name="small_sky_object_index",
        progress_bar=False,
    )
    mr.create_index(args, dask_client)

    output_file = tmp_path / "small_sky_object_index" / "dataset" / "index" / "part.0.parquet"

    data_frame = pd.read_parquet(output_file, engine="pyarrow")
    npt.assert_array_equal(data_frame.columns, ["Norder", "Dir", "Npix"])
    assert data_frame.index.name == "id"


@pytest.mark.dask
def test_create_index_no_order_pixel(small_sky_object_catalog, tmp_path, dask_client):
    """Create an index for simple object catalog, without the partitioning columns,
    Norder, Dir, and Npix."""
    args = IndexArguments(
        input_catalog_path=small_sky_object_catalog,
        indexing_column="id",
        include_order_pixel=False,
        output_path=tmp_path,
        output_artifact_name="small_sky_object_index",
        progress_bar=False,
    )
    mr.create_index(args, dask_client)

    output_file = tmp_path / "small_sky_object_index" / "dataset" / "index" / "part.0.parquet"

    data_frame = pd.read_parquet(output_file, engine="pyarrow")
    npt.assert_array_equal(data_frame.columns, ["_healpix_29"])
    assert data_frame.index.name == "id"


@pytest.mark.dask
def test_create_index_source(small_sky_source_catalog, assert_parquet_file_index, tmp_path, dask_client):
    """Create simple index for the source table."""
    args = IndexArguments(
        input_catalog_path=small_sky_source_catalog,
        indexing_column="source_id",
        output_path=tmp_path,
        output_artifact_name="small_sky_source_index",
        progress_bar=False,
    )
    mr.create_index(args, dask_client)

    output_file = tmp_path / "small_sky_source_index" / "dataset" / "index" / "part.0.parquet"

    expected_ids = [*range(70_000, 87_161)]
    assert_parquet_file_index(output_file, expected_ids)

    data_frame = pd.read_parquet(output_file, engine="pyarrow")
    npt.assert_array_equal(
        data_frame.columns,
        ["_healpix_29", "Norder", "Dir", "Npix"],
    )
    assert data_frame.index.name == "source_id"
    assert len(data_frame) == 17161
    assert np.logical_and(data_frame["Norder"] >= 0, data_frame["Norder"] <= 2).all()


@pytest.mark.dask
def test_create_index_with_divisions(
    small_sky_source_catalog,
    assert_parquet_file_index,
    tmp_path,
    dask_client,
):
    """Create an index catalog for the large(r) source catalog, passing
    some divisions hints. This should partition the final output according to
    the `compute_partition_size` and not the provided divisions."""
    divisions = np.arange(start=70_000, stop=87_161, step=5_000)
    divisions = np.append(divisions, 87_161).tolist()

    args = IndexArguments(
        input_catalog_path=small_sky_source_catalog,
        indexing_column="source_id",
        output_path=tmp_path,
        output_artifact_name="small_sky_source_index",
        division_hints=divisions,
        drop_duplicates=False,
        progress_bar=False,
    )
    mr.create_index(args, dask_client)

    output_file = tmp_path / "small_sky_source_index" / "dataset" / "index" / "part.0.parquet"

    expected_ids = [*range(70_000, 87_161)]
    assert_parquet_file_index(output_file, expected_ids)

    data_frame = pd.read_parquet(output_file, engine="pyarrow")
    npt.assert_array_equal(
        data_frame.columns,
        ["_healpix_29", "Norder", "Dir", "Npix"],
    )
    assert data_frame.index.name == "source_id"
    assert len(data_frame) == 17161
    assert np.logical_and(data_frame["Norder"] >= 0, data_frame["Norder"] <= 2).all()


@pytest.mark.dask
def test_create_index_source_by_object(
    small_sky_source_catalog,
    assert_parquet_file_index,
    tmp_path,
    dask_client,
):
    """Create an index for the source table, using the source's object ID
    as the indexing key."""
    args = IndexArguments(
        input_catalog_path=small_sky_source_catalog,
        indexing_column="object_id",
        output_path=tmp_path,
        output_artifact_name="small_sky_source_index",
        progress_bar=False,
    )
    mr.create_index(args, dask_client)

    output_file = tmp_path / "small_sky_source_index" / "dataset" / "index" / "part.0.parquet"

    expected_ids = np.repeat([*range(700, 831)], 131)
    assert_parquet_file_index(output_file, expected_ids)

    data_frame = pd.read_parquet(output_file, engine="pyarrow")
    npt.assert_array_equal(
        data_frame.columns,
        ["_healpix_29", "Norder", "Dir", "Npix"],
    )
    assert data_frame.index.name == "object_id"
    assert len(data_frame) == 17161


@pytest.mark.dask
def test_create_index_extra_columns(
    small_sky_source_catalog,
    assert_parquet_file_index,
    tmp_path,
    dask_client,
):
    """Create an index with some additional payload columns."""
    args = IndexArguments(
        input_catalog_path=small_sky_source_catalog,
        indexing_column="object_id",
        output_path=tmp_path,
        extra_columns=["source_ra"],
        output_artifact_name="small_sky_source_index",
        progress_bar=False,
    )
    mr.create_index(args, dask_client)

    output_file = tmp_path / "small_sky_source_index" / "dataset" / "index" / "part.0.parquet"

    expected_ids = np.repeat([*range(700, 831)], 131)
    assert_parquet_file_index(output_file, expected_ids)

    data_frame = pd.read_parquet(output_file, engine="pyarrow")
    npt.assert_array_equal(
        data_frame.columns,
        ["source_ra", "_healpix_29", "Norder", "Dir", "Npix"],
    )
    assert data_frame.index.name == "object_id"
    assert len(data_frame) == 17161
