"""test stuff."""

import sys

import numpy as np
import numpy.testing as npt
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from hats import read_hats
from hats.catalog.index.index_catalog import IndexCatalog
from hats.pixel_math import HealpixPixel

import hats_import.index.run_index as runner
from hats_import.index.arguments import IndexArguments

pytestmark = pytest.mark.skipif(
    (3, 11) <= sys.version_info < (3, 12), reason="dask expr regression with python 3.11"
)


def test_empty_args():
    """Runner should fail with empty arguments"""
    with pytest.raises(TypeError, match="IndexArguments"):
        runner.run(None, None)


def test_bad_args():
    """Runner should fail with mis-typed arguments"""
    args = {"output_artifact_name": "bad_arg_type"}
    with pytest.raises(TypeError, match="IndexArguments"):
        runner.run(args, None)


@pytest.mark.dask
def test_run_index(
    small_sky_object_catalog,
    tmp_path,
    dask_client,
):
    """Test appropriate metadata is written"""

    args = IndexArguments(
        input_catalog_path=small_sky_object_catalog,
        indexing_column="id",
        output_path=tmp_path,
        output_artifact_name="small_sky_object_index",
        progress_bar=False,
    )
    runner.run(args, dask_client)

    # Check that the catalog metadata file exists
    catalog = read_hats(args.catalog_path)
    assert catalog.on_disk
    assert catalog.catalog_path == args.catalog_path

    basic_index_parquet_schema = pa.schema(
        [
            pa.field("_healpix_29", pa.int64()),
            pa.field("Norder", pa.uint8()),
            pa.field("Npix", pa.uint64()),
            pa.field("id", pa.int64()),
        ]
    )

    outfile = args.catalog_path / "dataset" / "index" / "part.0.parquet"
    schema = pq.read_metadata(outfile).schema.to_arrow_schema()
    assert schema.equals(basic_index_parquet_schema)

    schema = pq.read_metadata(args.catalog_path / "dataset" / "_metadata").schema.to_arrow_schema()
    assert schema.equals(basic_index_parquet_schema)

    schema = pq.read_metadata(args.catalog_path / "dataset" / "_common_metadata").schema.to_arrow_schema()
    assert schema.equals(basic_index_parquet_schema)


@pytest.mark.dask
def test_run_index_write_table_kwargs(
    small_sky_object_catalog,
    tmp_path,
    dask_client,
):
    """Test that write_table_kwargs are passed through to the parquet writer."""

    args = IndexArguments(
        input_catalog_path=small_sky_object_catalog,
        indexing_column="id",
        output_path=tmp_path,
        output_artifact_name="small_sky_object_index",
        write_table_kwargs={"compression": "snappy", "version": "1.0"},
        progress_bar=False,
    )
    runner.run(args, dask_client)

    outfile = args.catalog_path / "dataset" / "index" / "part.0.parquet"
    file_metadata = pq.ParquetFile(outfile).metadata
    assert file_metadata.format_version == "1.0"
    assert file_metadata.row_group(0).column(0).compression == "SNAPPY"


@pytest.mark.dask
def test_run_index_default_compression(
    small_sky_object_catalog,
    tmp_path,
    dask_client,
):
    """Test that leaf files are written with the default ZSTD compression,
    like all the other import pipelines."""

    args = IndexArguments(
        input_catalog_path=small_sky_object_catalog,
        indexing_column="id",
        output_path=tmp_path,
        output_artifact_name="small_sky_object_index",
        progress_bar=False,
    )
    runner.run(args, dask_client)

    outfile = args.catalog_path / "dataset" / "index" / "part.0.parquet"
    file_metadata = pq.ParquetFile(outfile).metadata
    assert file_metadata.row_group(0).column(0).compression == "ZSTD"


@pytest.mark.dask
def test_run_index_metadata_options(
    small_sky_object_catalog,
    tmp_path,
    dask_client,
):
    """Test that create_metadata is respected, and that per-partition statistics
    are never generated (they are keyed by healpix pixel, which is meaningless
    for index tables)."""

    with pytest.warns(UserWarning, match="create_per_partition_stats"):
        args = IndexArguments(
            input_catalog_path=small_sky_object_catalog,
            indexing_column="id",
            output_path=tmp_path,
            output_artifact_name="small_sky_object_index",
            create_metadata=False,
            create_per_partition_stats=True,
            progress_bar=False,
        )
    runner.run(args, dask_client)

    assert not (args.catalog_path / "dataset" / "_metadata").exists()
    assert (args.catalog_path / "dataset" / "_common_metadata").exists()
    assert not (args.catalog_path / "per_partition_statistics.parquet").exists()


@pytest.mark.dask
@pytest.mark.parametrize(
    "small_sky_source_catalog",
    ["small_sky_source_catalog", "small_sky_source_npix_dir_catalog"],
    indirect=True,
)
def test_run_index_on_source(
    small_sky_source_catalog,
    tmp_path,
    dask_client,
):
    """Test appropriate metadata is written, when primary catalog covers multiple pixels."""

    args = IndexArguments(
        input_catalog_path=small_sky_source_catalog,
        indexing_column="source_id",
        extra_columns=["mag", "band"],
        output_path=tmp_path,
        output_artifact_name="small_sky_source_id_index",
        progress_bar=False,
    )
    runner.run(args, dask_client)

    # Check that the catalog metadata file exists
    catalog = read_hats(args.catalog_path)
    assert catalog.on_disk
    assert catalog.catalog_path == args.catalog_path
    assert catalog.catalog_info.extra_columns == ["mag", "band"]

    basic_index_parquet_schema = pa.schema(
        [
            pa.field("mag", pa.float64()),
            pa.field("band", pa.string()),
            pa.field("_healpix_29", pa.int64()),
            pa.field("Norder", pa.uint8()),
            pa.field("Npix", pa.uint64()),
            pa.field("source_id", pa.int64()),
        ]
    )

    outfile = args.catalog_path / "dataset" / "index" / "part.0.parquet"
    schema = pq.read_metadata(outfile).schema.to_arrow_schema()
    assert schema.equals(basic_index_parquet_schema)

    schema = pq.read_metadata(args.catalog_path / "dataset" / "_metadata").schema.to_arrow_schema()
    assert schema.equals(basic_index_parquet_schema)

    schema = pq.read_metadata(args.catalog_path / "dataset" / "_common_metadata").schema.to_arrow_schema()
    assert schema.equals(basic_index_parquet_schema)


@pytest.mark.dask
def test_run_index_on_source_object_id(
    small_sky_source_catalog,
    dask_client,
    tmp_path,
    assert_parquet_file_index,
):
    """Test appropriate metadata is written."""

    args = IndexArguments(
        input_catalog_path=small_sky_source_catalog,
        indexing_column="object_id",
        output_path=tmp_path,
        output_artifact_name="small_sky_source_object_id_index",
        include_healpix_29=False,
        progress_bar=False,
    )
    runner.run(args, dask_client)

    # Check that the catalog metadata file exists
    catalog = read_hats(args.catalog_path)
    assert catalog.on_disk
    assert catalog.catalog_path == args.catalog_path

    basic_index_parquet_schema = pa.schema(
        [
            pa.field("Norder", pa.uint8()),
            pa.field("Npix", pa.uint64()),
            pa.field("object_id", pa.int64()),
        ]
    )

    outfile = args.catalog_path / "dataset" / "index" / "part.0.parquet"
    schema = pq.read_metadata(outfile).schema.to_arrow_schema()
    assert schema.equals(basic_index_parquet_schema)

    id_range = np.arange(700, 831)
    ## Some of the objects have sources that span two source partitions.
    doubled_up = [706, 707, 716, 726, 730, 736, 740, 779, 780, 784, 787, 789, 790, 792, 797, 818, 820]
    doubled_up.extend(id_range)

    assert_parquet_file_index(outfile, doubled_up)

    schema = pq.read_metadata(args.catalog_path / "dataset" / "_metadata").schema.to_arrow_schema()
    assert schema.equals(basic_index_parquet_schema)

    schema = pq.read_metadata(args.catalog_path / "dataset" / "_common_metadata").schema.to_arrow_schema()
    assert schema.equals(basic_index_parquet_schema)


@pytest.mark.dask
def test_run_index_on_nested_source_id(
    small_sky_nested_catalog,
    dask_client,
    tmp_path,
):
    """Test appropriate metadata is written."""

    args = IndexArguments(
        input_catalog_path=small_sky_nested_catalog,
        indexing_column="lc.source_id",
        output_path=tmp_path,
        output_artifact_name="small_sky_nested_source_id_index",
        include_healpix_29=False,
        progress_bar=False,
    )
    runner.run(args, dask_client)

    # Check that the catalog metadata file exists
    catalog = read_hats(args.catalog_path)
    assert catalog.on_disk
    assert catalog.catalog_path == args.catalog_path

    basic_index_parquet_schema = pa.schema(
        [
            pa.field("Norder", pa.uint8()),
            pa.field("Npix", pa.uint64()),
            pa.field("lc.source_id", pa.int64()),
        ]
    )

    outfile = args.catalog_path / "dataset" / "index" / "part.0.parquet"
    schema = pq.read_metadata(outfile).schema.to_arrow_schema()
    assert schema.equals(basic_index_parquet_schema)

    schema = pq.read_metadata(args.catalog_path / "dataset" / "_metadata").schema.to_arrow_schema()
    assert schema.equals(basic_index_parquet_schema)

    schema = pq.read_metadata(args.catalog_path / "dataset" / "_common_metadata").schema.to_arrow_schema()
    assert schema.equals(basic_index_parquet_schema)

    assert isinstance(catalog, IndexCatalog)
    npt.assert_array_equal(catalog.loc_partitions([70003]), [HealpixPixel(2, 183)])
