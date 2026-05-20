"""Tests of argument validation"""

import re
import sys

import pytest
from packaging.version import Version
import importlib

from hats_import.index.arguments import IndexArguments


@pytest.mark.skipif(
    Version.from_parts(release=sys.version_info[:2]) != Version("3.11"),
    reason="regression only in python 3.11",
)
def test_check_versions():
    # We should still check the environment
    #  - test_lowest_versions will succeed, all other 3.11 CI will expect raises error.
    dask_version = importlib.metadata.version("dask")
    if dask_version.startswith("2025.3"):
        IndexArguments.check_versions()
    else:
        with pytest.raises(RuntimeError, match="dask version"):
            IndexArguments.check_versions()


@pytest.mark.skipif(
    Version.from_parts(release=sys.version_info[:2]) == Version("3.11"),
    reason="dask expr regression with python 3.11",
)
def test_none():
    """No arguments provided. Should error for required args."""
    with pytest.raises(ValueError):
        IndexArguments()


@pytest.mark.skipif(
    Version.from_parts(release=sys.version_info[:2]) == Version("3.11"),
    reason="dask expr regression with python 3.11",
)
def test_empty_required(tmp_path, small_sky_object_catalog):
    """*Most* required arguments are provided."""
    ## Input path is missing
    with pytest.raises(ValueError, match="input_catalog_path"):
        IndexArguments(
            indexing_column="id",
            output_path=tmp_path,
            output_artifact_name="small_sky_object_index",
        )

    ## Input path is bad
    with pytest.raises(ValueError, match="input_catalog_path"):
        IndexArguments(
            input_catalog_path="/foo",
            indexing_column="id",
            output_path=tmp_path,
            output_artifact_name="small_sky_object_index",
        )

    ## Indexing column is required.
    with pytest.raises(ValueError, match="indexing_column "):
        IndexArguments(
            input_catalog_path=small_sky_object_catalog,
            indexing_column="",
            output_path=tmp_path,
            output_artifact_name="small_sky_object_index",
        )


@pytest.mark.skipif(
    Version.from_parts(release=sys.version_info[:2]) == Version("3.11"),
    reason="dask expr regression with python 3.11",
)
def test_invalid_paths(tmp_path, small_sky_object_catalog):
    """Required arguments are provided, but paths aren't found."""
    ## Prove that it works with required args
    IndexArguments(
        input_catalog_path=small_sky_object_catalog,
        indexing_column="id",
        output_path=tmp_path,
        output_artifact_name="small_sky_object_index",
    )
    ## Input path is invalid catalog
    with pytest.raises(ValueError, match="input_catalog_path not a valid catalog"):
        IndexArguments(
            input_catalog_path="path",
            indexing_column="id",
            output_path=f"{tmp_path}/path",
            output_artifact_name="small_sky_object_index",
        )


@pytest.mark.skipif(
    Version.from_parts(release=sys.version_info[:2]) == Version("3.11"),
    reason="dask expr regression with python 3.11",
)
def test_good_paths(tmp_path, small_sky_object_catalog):
    """Required arguments are provided, and paths are found."""
    tmp_path_str = str(tmp_path)
    args = IndexArguments(
        input_catalog_path=small_sky_object_catalog,
        indexing_column="id",
        output_path=tmp_path,
        output_artifact_name="small_sky_object_index",
    )
    assert args.input_catalog_path == small_sky_object_catalog
    assert str(args.output_path) == tmp_path_str
    assert str(args.tmp_path).startswith(tmp_path_str)


@pytest.mark.skipif(
    Version.from_parts(release=sys.version_info[:2]) == Version("3.11"),
    reason="dask expr regression with python 3.11",
)
def test_column_inclusion_args(tmp_path, small_sky_object_catalog):
    """Test errors for healpix partitioning arguments"""
    with pytest.raises(ValueError, match="one of"):
        IndexArguments(
            input_catalog_path=small_sky_object_catalog,
            indexing_column="id",
            output_path=tmp_path,
            output_artifact_name="small_sky_object_index",
            include_healpix_29=False,
            include_order_pixel=False,
        )
    _ = IndexArguments(
        input_catalog_path=small_sky_object_catalog,
        indexing_column="id",
        output_path=tmp_path,
        output_artifact_name="small_sky_object_index",
        include_healpix_29=True,
        include_order_pixel=True,
    )

    _ = IndexArguments(
        input_catalog_path=small_sky_object_catalog,
        indexing_column="id",
        output_path=tmp_path,
        output_artifact_name="small_sky_object_index",
        include_healpix_29=True,
        include_order_pixel=False,
    )
    _ = IndexArguments(
        input_catalog_path=small_sky_object_catalog,
        indexing_column="id",
        output_path=tmp_path,
        output_artifact_name="small_sky_object_index",
        include_healpix_29=False,
        include_order_pixel=True,
    )


@pytest.mark.skipif(
    Version.from_parts(release=sys.version_info[:2]) == Version("3.11"),
    reason="dask expr regression with python 3.11",
)
def test_extra_columns(tmp_path, small_sky_object_catalog):
    args = IndexArguments(
        input_catalog_path=small_sky_object_catalog,
        indexing_column="id",
        output_path=tmp_path,
        output_artifact_name="small_sky_object_index",
        extra_columns=["_healpix_29"],
    )
    assert args.extra_columns == ["_healpix_29"]

    args = IndexArguments(
        input_catalog_path=small_sky_object_catalog,
        indexing_column="id",
        output_path=tmp_path,
        output_artifact_name="small_sky_object_index",
        include_radec=True,
    )
    assert args.extra_columns == ["ra", "dec"]

    with pytest.raises(ValueError, match=re.escape("not in input catalog (mag_r)")):
        IndexArguments(
            input_catalog_path=small_sky_object_catalog,
            indexing_column="id",
            output_path=tmp_path,
            output_artifact_name="small_sky_object_index",
            extra_columns=["mag_r"],
        )


@pytest.mark.skipif(
    Version.from_parts(release=sys.version_info[:2]) == Version("3.11"),
    reason="dask expr regression with python 3.11",
)
def test_compute_partition_size(tmp_path, small_sky_object_catalog):
    """Test validation of compute_partition_size."""
    with pytest.raises(ValueError, match="compute_partition_size"):
        IndexArguments(
            input_catalog_path=small_sky_object_catalog,
            indexing_column="id",
            output_path=tmp_path,
            output_artifact_name="small_sky_object_index",
            compute_partition_size=10,  ## not a valid join option
        )


@pytest.mark.skipif(
    Version.from_parts(release=sys.version_info[:2]) == Version("3.11"),
    reason="dask expr regression with python 3.11",
)
def test_to_table_properties(small_sky_object_catalog, tmp_path):
    """Verify creation of catalog parameters for index to be created."""
    args = IndexArguments(
        input_catalog_path=small_sky_object_catalog,
        indexing_column="id",
        output_path=tmp_path,
        output_artifact_name="small_sky_object_index",
        include_healpix_29=True,
        include_order_pixel=True,
    )
    catalog_info = args.to_table_properties(total_rows=10)
    assert catalog_info.catalog_name == args.output_artifact_name
    assert catalog_info.total_rows == 10
