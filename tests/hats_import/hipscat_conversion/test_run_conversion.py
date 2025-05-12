"""test stuff."""

import hats
import numpy.testing as npt
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from hats.io.file_io import file_io
from pyarrow.parquet import ParquetFile

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


# pylint: disable=unused-import
try:
    import healpy as hp

    HAVE_HEALPY = True
except ImportError:
    HAVE_HEALPY = False


@pytest.mark.skipif(not HAVE_HEALPY, reason="healpy is not installed")
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
    assert len(catalog.get_healpix_pixels()) == 1
    assert int(catalog.catalog_info.__pydantic_extra__["hats_estsize"]) > 0

    # Check that the catalog parquet file exists and contains correct object IDs
    output_file = args.catalog_path / "dataset" / "Norder=0" / "Dir=0" / "Npix=11.parquet"

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
        ]
    )
    schema = pq.read_metadata(output_file).schema.to_arrow_schema()
    assert schema.equals(expected_parquet_schema)
    assert schema.metadata is None
    schema = pq.read_metadata(args.catalog_path / "dataset" / "_metadata").schema.to_arrow_schema()
    assert schema.equals(expected_parquet_schema)
    assert schema.metadata is None
    schema = pq.read_metadata(args.catalog_path / "dataset" / "_common_metadata").schema.to_arrow_schema()
    assert schema.equals(expected_parquet_schema)
    assert schema.metadata is None

    data = file_io.read_parquet_file_to_pandas(
        output_file,
        columns=["id", "ra", "dec", "_healpix_29"],
    )
    assert "_healpix_29" in data.columns
    assert data.index.name is None

    # Check that the data thumbnail exists
    data_thumbnail_pointer = args.catalog_path / "dataset" / "data_thumbnail.parquet"
    assert data_thumbnail_pointer.exists()
    thumbnail = ParquetFile(data_thumbnail_pointer)
    thumbnail_schema = thumbnail.metadata.schema.to_arrow_schema()
    assert thumbnail_schema.equals(expected_parquet_schema)
    # The thumbnail has 1 row because the catalog has only 1 pixel
    assert len(thumbnail.read()) == 1


@pytest.mark.skipif(not HAVE_HEALPY, reason="healpy is not installed")
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
    assert len(catalog.get_healpix_pixels()) == 14

    output_file = args.catalog_path / "dataset" / "Norder=2" / "Dir=0" / "Npix=185.parquet"

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
    ]
    schema = pq.read_metadata(output_file).schema
    npt.assert_array_equal(schema.names, source_columns)
    assert schema.to_arrow_schema().metadata is None
    schema = pq.read_metadata(args.catalog_path / "dataset" / "_metadata").schema
    npt.assert_array_equal(schema.names, source_columns)
    assert schema.to_arrow_schema().metadata is None
    schema = pq.read_metadata(args.catalog_path / "dataset" / "_common_metadata").schema
    npt.assert_array_equal(schema.names, source_columns)
    assert schema.to_arrow_schema().metadata is None

    # Check that the data thumbnail exists
    data_thumbnail_pointer = args.catalog_path / "dataset" / "data_thumbnail.parquet"
    assert data_thumbnail_pointer.exists()
    thumbnail = ParquetFile(data_thumbnail_pointer)
    thumbnail_schema = thumbnail.metadata.schema.to_arrow_schema()
    assert thumbnail_schema.equals(schema.to_arrow_schema())
    # The thumbnail has 14 rows because the catalog has 14 pixels
    assert len(thumbnail.read()) == 14
