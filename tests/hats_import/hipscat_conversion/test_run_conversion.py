"""test stuff."""

import os

import hats
import numpy.testing as npt
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import hats_import.hipscat_conversion.run_conversion as runner
from hats_import.hipscat_conversion.arguments import ConversionArguments


def test_empty_args():
    """Runner should fail with empty arguments"""
    with pytest.raises(TypeError, match="ConversionArguments"):
        runner.run(None, None)


def test_bad_args():
    """Runner should fail with mis-typed arguments"""
    args = {"output_artifact_name": "bad_arg_type"}
    with pytest.raises(TypeError, match="ConversionArguments"):
        runner.run(args, None)


@pytest.mark.dask
def test_run_conversion_object(
    test_data_dir,
    tmp_path,
    assert_parquet_file_ids,
    dask_client,
):
    """Test appropriate metadata is written"""

    input_catalog_dir = test_data_dir / "hipscat" / "small_sky_object_catalog"

    args = ConversionArguments(
        input_catalog_path=input_catalog_dir,
        output_path=tmp_path,
        output_artifact_name="small_sky_object_hats",
        progress_bar=False,
    )
    runner.run(args, dask_client)

    # Check that the catalog metadata file exists
    catalog = hats.read_hats(args.catalog_path)
    assert catalog.on_disk
    assert catalog.catalog_path == args.catalog_path
    assert int(catalog.catalog_info.__pydantic_extra__["hats_estsize"]) > 0

    # Check that the catalog parquet file exists and contains correct object IDs
    output_file = os.path.join(args.catalog_path, "Norder=0", "Dir=0", "Npix=11.parquet")

    expected_ids = [*range(700, 831)]
    assert_parquet_file_ids(output_file, "id", expected_ids)

    # Check that the schema is correct for leaf parquet and _metadata files
    expected_parquet_schema = pa.schema(
        [
            pa.field("_healpix_29", pa.int64()),
            pa.field("id", pa.int64()),
            pa.field("ra", pa.float64()),
            pa.field("dec", pa.float64()),
            pa.field("ra_error", pa.int64()),
            pa.field("dec_error", pa.int64()),
            pa.field("Norder", pa.int8()),
            pa.field("Dir", pa.int64()),
            pa.field("Npix", pa.int64()),
        ]
    )
    schema = pq.read_metadata(output_file).schema.to_arrow_schema()
    assert schema.equals(expected_parquet_schema, check_metadata=False)
    schema = pq.read_metadata(os.path.join(args.catalog_path, "_metadata")).schema.to_arrow_schema()
    assert schema.equals(expected_parquet_schema, check_metadata=False)
    schema = pq.read_metadata(os.path.join(args.catalog_path, "_common_metadata")).schema.to_arrow_schema()
    assert schema.equals(expected_parquet_schema, check_metadata=False)


@pytest.mark.dask
def test_run_conversion_source(
    test_data_dir,
    tmp_path,
    dask_client,
):
    """Test appropriate metadata is written"""

    input_catalog_dir = test_data_dir / "hipscat" / "small_sky_source_catalog"

    args = ConversionArguments(
        input_catalog_path=input_catalog_dir,
        output_path=tmp_path,
        output_artifact_name="small_sky_source_hats",
        progress_bar=False,
    )
    runner.run(args, dask_client)

    # Check that the catalog metadata file exists
    catalog = hats.read_hats(args.catalog_path)
    assert catalog.on_disk
    assert catalog.catalog_path == args.catalog_path

    output_file = os.path.join(args.catalog_path, "Norder=2", "Dir=0", "Npix=185.parquet")

    source_columns = [
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
    ]
    schema = pq.read_metadata(output_file).schema
    npt.assert_array_equal(schema.names, source_columns)
    schema = pq.read_metadata(os.path.join(args.catalog_path, "_metadata")).schema
    npt.assert_array_equal(schema.names, source_columns)
    schema = pq.read_metadata(os.path.join(args.catalog_path, "_common_metadata")).schema
    npt.assert_array_equal(schema.names, source_columns)
