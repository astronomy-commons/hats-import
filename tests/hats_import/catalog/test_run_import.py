"""Functional tests for catalog import"""

import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from hats import read_hats
from hats.pixel_math.sparse_histogram import SparseHistogram

import hats_import.catalog.run_import as runner
from hats_import.catalog.arguments import ImportArguments
from hats_import.catalog.file_readers import CsvReader
from hats_import.catalog.resume_plan import ResumePlan


def test_empty_args():
    """Runner should fail with empty arguments"""
    with pytest.raises(ValueError, match="args is required"):
        runner.run(None, None)


def test_bad_args():
    """Runner should fail with mis-typed arguments"""
    args = {"output_artifact_name": "bad_arg_type"}
    with pytest.raises(ValueError, match="ImportArguments"):
        runner.run(args, None)


@pytest.mark.dask
def test_resume_dask_runner(
    dask_client,
    small_sky_parts_dir,
    resume_dir,
    tmp_path,
    assert_parquet_file_ids,
):
    """Test execution in the presence of some resume files."""
    ## First, copy over our intermediate files.
    ## This prevents overwriting source-controlled resume files.
    intermediate_dir = tmp_path / "resume_catalog" / "intermediate"
    shutil.copytree(resume_dir / "intermediate", intermediate_dir)
    ## Now set up our resume files to match previous work.
    resume_tmp = tmp_path / "tmp" / "resume_catalog"
    ResumePlan(tmp_path=resume_tmp, progress_bar=False)
    histogram = SparseHistogram.make_from_counts([11], [131], 0)
    empty = SparseHistogram.make_empty(0)
    for file_index in range(0, 5):
        ResumePlan.touch_key_done_file(resume_tmp, ResumePlan.SPLITTING_STAGE, f"split_{file_index}")
        histogram_file = ResumePlan.partial_histogram_file(
            tmp_path=resume_tmp, mapping_key=f"map_{file_index}"
        )
        if file_index == 0:
            histogram.to_file(histogram_file)
        else:
            empty.to_file(histogram_file)

    ResumePlan.touch_key_done_file(resume_tmp, ResumePlan.REDUCING_STAGE, "0_11")

    shutil.copytree(resume_dir / "Norder=0", tmp_path / "resume_catalog" / "dataset" / "Norder=0")

    args = ImportArguments(
        output_artifact_name="resume_catalog",
        input_path=small_sky_parts_dir,
        file_reader="csv",
        output_path=tmp_path,
        dask_tmp=tmp_path,
        tmp_dir=tmp_path,
        resume_tmp=tmp_path / "tmp",
        highest_healpix_order=0,
        pixel_threshold=1000,
        progress_bar=False,
    )

    runner.run(args, dask_client)

    # Check that the catalog metadata file exists
    catalog = read_hats(args.catalog_path)
    assert catalog.on_disk
    assert catalog.catalog_path == args.catalog_path
    assert catalog.catalog_info.ra_column == "ra"
    assert catalog.catalog_info.dec_column == "dec"
    assert catalog.catalog_info.total_rows == 131
    assert len(catalog.get_healpix_pixels()) == 1

    # Check that the catalog parquet file exists and contains correct object IDs
    output_file = Path(args.catalog_path) / "dataset" / "Norder=0" / "Dir=0" / "Npix=11.parquet"

    expected_ids = [*range(700, 831)]
    assert_parquet_file_ids(output_file, "id", expected_ids)

    ## Re-running the pipeline with fully done intermediate files
    ## should result in no changes to output files.
    shutil.copytree(resume_dir / "intermediate", resume_tmp)

    args = ImportArguments(
        output_artifact_name="resume",
        input_path=small_sky_parts_dir,
        file_reader="csv",
        output_path=tmp_path,
        dask_tmp=tmp_path,
        tmp_dir=tmp_path,
        highest_healpix_order=0,
        pixel_threshold=1000,
        progress_bar=False,
    )

    runner.run(args, dask_client)

    catalog = read_hats(args.catalog_path)
    assert catalog.on_disk
    assert catalog.catalog_path == args.catalog_path
    assert catalog.catalog_info.ra_column == "ra"
    assert catalog.catalog_info.dec_column == "dec"
    assert catalog.catalog_info.total_rows == 131
    assert len(catalog.get_healpix_pixels()) == 1
    assert_parquet_file_ids(output_file, "id", expected_ids)


@pytest.mark.dask
def test_resume_dask_runner_diff_pixel_order(
    dask_client,
    small_sky_parts_dir,
    resume_dir,
    tmp_path,
    assert_parquet_file_ids,
):
    """Test execution in the presence of histogram files that are not compatible
    with the current HEALPix order."""
    ## First, copy over our intermediate files.
    ## This prevents overwriting source-controlled resume files.
    intermediate_dir = tmp_path / "resume_catalog" / "intermediate"
    shutil.copytree(resume_dir / "intermediate", intermediate_dir)

    ## Now set up our resume files to match previous work.
    resume_tmp = tmp_path / "tmp" / "resume_catalog"
    ResumePlan(tmp_path=resume_tmp, progress_bar=False)
    SparseHistogram.make_from_counts([11], [131], 0).to_dense_file(resume_tmp / "mapping_histogram.npz")
    for file_index in range(0, 5):
        ResumePlan.touch_key_done_file(resume_tmp, ResumePlan.SPLITTING_STAGE, f"split_{file_index}")

    shutil.copytree(resume_dir / "Norder=0", tmp_path / "resume_catalog" / "Norder=0")

    with pytest.raises(ValueError, match="incompatible with the highest healpix order"):
        args = ImportArguments(
            output_artifact_name="resume_catalog",
            input_path=small_sky_parts_dir,
            file_reader="csv",
            output_path=tmp_path,
            dask_tmp=tmp_path,
            tmp_dir=tmp_path,
            resume_tmp=tmp_path / "tmp",
            constant_healpix_order=1,
            pixel_threshold=1000,
            progress_bar=False,
        )
        runner.run(args, dask_client)

    # Running with resume set to "False" will start the pipeline from scratch
    args = ImportArguments(
        output_artifact_name="resume_catalog",
        input_path=small_sky_parts_dir,
        file_reader="csv",
        output_path=tmp_path,
        dask_tmp=tmp_path,
        tmp_dir=tmp_path,
        resume_tmp=tmp_path / "tmp",
        constant_healpix_order=1,
        pixel_threshold=1000,
        progress_bar=False,
        resume=False,
    )
    runner.run(args, dask_client)

    # Check that the catalog metadata file exists
    catalog = read_hats(args.catalog_path)
    assert catalog.on_disk
    assert catalog.catalog_path == args.catalog_path
    assert catalog.catalog_info.ra_column == "ra"
    assert catalog.catalog_info.dec_column == "dec"
    assert catalog.catalog_info.total_rows == 131
    assert len(catalog.get_healpix_pixels()) == 4

    for n_pix in range(44, 48):
        output_filepath = args.catalog_path / "dataset" / "Norder=1" / "Dir=0" / f"Npix={n_pix}.parquet"
        expected_filepath = resume_dir / "Norder=1" / "Dir=0" / f"Npix={n_pix}.parquet"
        expected_file = pd.read_parquet(expected_filepath, engine="pyarrow")
        assert_parquet_file_ids(output_filepath, "id", expected_file["id"].to_numpy())


@pytest.mark.dask
def test_resume_dask_runner_histograms_diff_size(
    dask_client,
    small_sky_parts_dir,
    tmp_path,
):
    """Tests that the pipeline errors if the partial histograms have different sizes."""
    resume_tmp = tmp_path / "tmp" / "resume_catalog"
    ResumePlan(tmp_path=resume_tmp, progress_bar=False)

    # We'll create mock partial histograms of size 0 and 2
    histogram = SparseHistogram.make_empty(0)
    wrong_histogram = SparseHistogram.make_empty(2)

    for file_index in range(0, 5):
        ResumePlan.touch_key_done_file(resume_tmp, ResumePlan.SPLITTING_STAGE, f"split_{file_index}")

        histogram_file = ResumePlan.partial_histogram_file(
            tmp_path=resume_tmp, mapping_key=f"map_{file_index}"
        )
        if file_index == 2:
            histogram.to_file(histogram_file)
        else:
            wrong_histogram.to_file(histogram_file)

    with pytest.raises(ValueError, match="histogram partials have incompatible sizes"):
        args = ImportArguments(
            output_artifact_name="resume_catalog",
            input_path=small_sky_parts_dir,
            file_reader="csv",
            output_path=tmp_path,
            dask_tmp=tmp_path,
            tmp_dir=tmp_path,
            resume_tmp=tmp_path / "tmp",
            constant_healpix_order=1,
            pixel_threshold=1000,
            progress_bar=False,
        )
        runner.run(args, dask_client)


@pytest.mark.dask
def test_dask_runner(
    dask_client,
    small_sky_parts_dir,
    assert_parquet_file_ids,
    tmp_path,
):
    """Test basic execution and the types of the written data."""
    args = ImportArguments(
        output_artifact_name="small_sky_object_catalog",
        input_path=small_sky_parts_dir,
        file_reader=CsvReader(
            type_map={
                "ra": np.float32,
                "dec": np.float32,
                "ra_error": np.float32,
                "dec_error": np.float32,
            }
        ),
        output_path=tmp_path,
        dask_tmp=tmp_path,
        highest_healpix_order=1,
        progress_bar=False,
    )

    runner.run(args, dask_client)

    # Check that the catalog metadata file exists
    catalog = read_hats(args.catalog_path)
    assert catalog.on_disk
    assert catalog.catalog_path == args.catalog_path
    assert catalog.catalog_info.ra_column == "ra"
    assert catalog.catalog_info.dec_column == "dec"
    assert catalog.catalog_info.total_rows == 131
    assert len(catalog.get_healpix_pixels()) == 1

    # Check that the catalog parquet file exists and contains correct object IDs
    output_file = os.path.join(args.catalog_path, "dataset", "Norder=0", "Dir=0", "Npix=11.parquet")

    expected_ids = [*range(700, 831)]
    assert_parquet_file_ids(output_file, "id", expected_ids)

    # Check that the schema is correct for leaf parquet and _metadata files
    expected_parquet_schema = pa.schema(
        [
            pa.field("_healpix_29", pa.int64()),
            pa.field("id", pa.int64()),
            pa.field("ra", pa.float32()),
            pa.field("dec", pa.float32()),
            pa.field("ra_error", pa.float32()),
            pa.field("dec_error", pa.float32()),
            pa.field("Norder", pa.uint8()),
            pa.field("Dir", pa.uint64()),
            pa.field("Npix", pa.uint64()),
        ]
    )
    schema = pq.read_metadata(output_file).schema.to_arrow_schema()
    assert schema.equals(expected_parquet_schema)
    schema = pq.read_metadata(args.catalog_path / "dataset" / "_metadata").schema.to_arrow_schema()
    assert schema.equals(expected_parquet_schema)

    # Check that, when re-loaded as a pandas dataframe, the appropriate numeric types are used.
    data_frame = pd.read_parquet(output_file, engine="pyarrow")
    expected_dtypes = pd.Series(
        {
            "_healpix_29": np.int64,
            "id": np.int64,
            "ra": np.float32,
            "dec": np.float32,
            "ra_error": np.float32,
            "dec_error": np.float32,
            "Norder": np.uint8,
            "Dir": np.uint64,
            "Npix": np.uint64,
        }
    )
    assert data_frame.dtypes.equals(expected_dtypes)
    assert data_frame.index.dtype == np.int64


@pytest.mark.dask
def test_dask_runner_stats_only(dask_client, small_sky_parts_dir, tmp_path):
    """Test basic execution, without generating catalog parquet outputs."""
    args = ImportArguments(
        output_artifact_name="small_sky",
        input_path=small_sky_parts_dir,
        file_reader="csv",
        output_path=tmp_path,
        dask_tmp=tmp_path,
        highest_healpix_order=1,
        progress_bar=False,
        debug_stats_only=True,
    )

    runner.run(args, dask_client)

    metadata_filename = os.path.join(args.catalog_path, "properties")
    assert os.path.exists(metadata_filename)

    # Check that the catalog parquet file DOES NOT exist
    output_file = os.path.join(args.catalog_path, "dataset", "Norder=0", "Dir=0", "Npix=11.parquet")

    assert not os.path.exists(output_file)

    catalog = read_hats(args.catalog_path)
    assert catalog.on_disk
    assert catalog.catalog_path == args.catalog_path
    assert catalog.catalog_info.ra_column == "ra"
    assert catalog.catalog_info.dec_column == "dec"
    assert len(catalog.get_healpix_pixels()) == 1


@pytest.mark.dask
def test_import_mismatch_expectation(
    dask_client,
    small_sky_parts_dir,
    tmp_path,
):
    """Test that the pipeline execution fails if the number of rows does not match
    explicit (but wrong) expectation."""
    args = ImportArguments(
        output_artifact_name="small_sky",
        input_path=small_sky_parts_dir,
        file_reader="csv",
        output_path=tmp_path,
        dask_tmp=tmp_path,
        highest_healpix_order=1,
        progress_bar=False,
        expected_total_rows=1_000,
    )

    with pytest.raises(ValueError, match="does not match expectation"):
        runner.run(args, dask_client)
