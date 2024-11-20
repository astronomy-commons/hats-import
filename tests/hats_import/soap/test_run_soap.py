"""Test full execution of SOAP."""

import os

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from hats import read_hats

import hats_import.soap.run_soap as runner
from hats_import.soap.arguments import SoapArguments


def test_empty_args():
    """Runner should fail with empty arguments"""
    with pytest.raises(TypeError):
        runner.run(None, None)


def test_bad_args():
    """Runner should fail with mis-typed arguments"""
    args = {"output_catalog_name": "bad_arg_type"}
    with pytest.raises(TypeError):
        runner.run(args, None)


@pytest.mark.dask
def test_object_to_source(dask_client, small_sky_soap_args):
    """Test creating association between object and source catalogs."""
    runner.run(small_sky_soap_args, dask_client)

    ## Check that the association data can be parsed as a valid association catalog.
    catalog = read_hats(small_sky_soap_args.catalog_path)
    assert catalog.on_disk
    assert catalog.catalog_path == small_sky_soap_args.catalog_path
    assert len(catalog.get_join_pixels()) == 14
    assert catalog.catalog_info.total_rows == 17161
    assert not catalog.catalog_info.contains_leaf_files


@pytest.mark.dask
def test_object_to_self(dask_client, tmp_path, small_sky_object_catalog):
    """Test creating association between object and source catalogs."""
    small_sky_soap_args = SoapArguments(
        object_catalog_dir=small_sky_object_catalog,
        object_id_column="id",
        source_catalog_dir=small_sky_object_catalog,
        source_object_id_column="id",
        output_path=tmp_path,
        progress_bar=False,
        source_id_column="id",
        output_artifact_name="small_sky_object_to_source",
    )
    runner.run(small_sky_soap_args, dask_client)

    ## Check that the association data can be parsed as a valid association catalog.
    catalog = read_hats(small_sky_soap_args.catalog_path)
    assert catalog.on_disk
    assert catalog.catalog_path == small_sky_soap_args.catalog_path
    assert len(catalog.get_join_pixels()) == 1
    assert catalog.catalog_info.total_rows == 131
    assert not catalog.catalog_info.contains_leaf_files


@pytest.mark.dask
def test_object_to_source_with_leaves(
    dask_client, tmp_path, small_sky_object_catalog, small_sky_source_catalog
):
    """Test creating association between object and source catalogs."""
    small_sky_soap_args = SoapArguments(
        object_catalog_dir=small_sky_object_catalog,
        object_id_column="id",
        source_catalog_dir=small_sky_source_catalog,
        source_object_id_column="object_id",
        output_path=tmp_path,
        progress_bar=False,
        write_leaf_files=True,
        source_id_column="source_id",
        output_artifact_name="small_sky_object_to_source",
    )
    runner.run(small_sky_soap_args, dask_client)

    ## Check that the association data can be parsed as a valid association catalog.
    catalog = read_hats(small_sky_soap_args.catalog_path)
    assert catalog.on_disk
    assert catalog.catalog_path == small_sky_soap_args.catalog_path
    assert len(catalog.get_join_pixels()) == 14
    assert catalog.catalog_info.total_rows == 17161
    assert catalog.catalog_info.contains_leaf_files

    parquet_file_name = (
        small_sky_soap_args.catalog_path / "dataset" / "Norder=0" / "Dir=0" / "Npix=11.parquet"
    )
    assert os.path.exists(parquet_file_name), f"file not found [{parquet_file_name}]"

    parquet_file = pq.ParquetFile(parquet_file_name)
    assert parquet_file.metadata.num_row_groups == 14
    assert parquet_file.metadata.num_rows == 17161
    assert parquet_file.metadata.num_columns == 8

    exepcted_schema = pa.schema(
        [
            pa.field("object_id", pa.int64()),
            pa.field("source_id", pa.int64()),
            pa.field("Norder", pa.uint8()),
            pa.field("Dir", pa.uint64()),
            pa.field("Npix", pa.uint64()),
            pa.field("join_Norder", pa.uint8()),
            pa.field("join_Dir", pa.uint64()),
            pa.field("join_Npix", pa.uint64()),
        ]
    )
    assert parquet_file.metadata.schema.to_arrow_schema().equals(exepcted_schema)


@pytest.mark.dask
def test_object_to_source_with_leaves_drop_duplicates(
    dask_client, tmp_path, small_sky_object_catalog, small_sky_source_catalog
):
    """Test creating association between object and source catalogs."""
    small_sky_soap_args = SoapArguments(
        object_catalog_dir=small_sky_object_catalog,
        object_id_column="id",
        source_catalog_dir=small_sky_source_catalog,
        source_object_id_column="object_id",
        output_path=tmp_path,
        progress_bar=False,
        write_leaf_files=True,
        source_id_column="object_id",
        output_artifact_name="small_sky_object_to_source",
    )
    runner.run(small_sky_soap_args, dask_client)

    ## Check that the association data can be parsed as a valid association catalog.
    catalog = read_hats(small_sky_soap_args.catalog_path)
    assert catalog.on_disk
    assert catalog.catalog_path == small_sky_soap_args.catalog_path
    assert len(catalog.get_join_pixels()) == 14
    assert catalog.catalog_info.total_rows == 148
    assert catalog.catalog_info.contains_leaf_files

    parquet_file_name = (
        small_sky_soap_args.catalog_path / "dataset" / "Norder=0" / "Dir=0" / "Npix=11.parquet"
    )
    assert os.path.exists(parquet_file_name), f"file not found [{parquet_file_name}]"

    parquet_file = pq.ParquetFile(parquet_file_name)
    assert parquet_file.metadata.num_row_groups == 14
    assert parquet_file.metadata.num_rows == 148
    assert parquet_file.metadata.num_columns == 8

    exepcted_schema = pa.schema(
        [
            pa.field("object_id", pa.int64()),
            pa.field("source_id", pa.int64()),
            pa.field("Norder", pa.uint8()),
            pa.field("Dir", pa.uint64()),
            pa.field("Npix", pa.uint64()),
            pa.field("join_Norder", pa.uint8()),
            pa.field("join_Dir", pa.uint64()),
            pa.field("join_Npix", pa.uint64()),
        ]
    )
    assert parquet_file.metadata.schema.to_arrow_schema().equals(exepcted_schema)
