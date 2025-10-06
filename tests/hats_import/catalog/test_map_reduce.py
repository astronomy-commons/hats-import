"""Tests of map reduce operations"""

import os
import shutil
from io import StringIO

import cloudpickle
import hats.pixel_math as hist
import hats.pixel_math.healpix_shim as hp
import numpy as np
import numpy.testing as npt
import pandas as pd
import pyarrow as pa
import pytest
from hats.pixel_math.sparse_histogram import SparseHistogram

import hats_import.catalog.map_reduce as mr
from hats_import.catalog.file_readers import get_file_reader
from hats_import.catalog.resume_plan import ResumePlan


def pickle_file_reader(tmp_path, file_reader) -> str:
    """Utility method to pickle a file reader, and return path to pickle."""
    pickled_reader_file = tmp_path / "reader.pickle"
    with open(pickled_reader_file, "wb") as pickle_file:
        cloudpickle.dump(file_reader, pickle_file)
    return pickled_reader_file


def test_read_empty_filename():
    """Empty file name"""
    with pytest.raises(FileNotFoundError):
        mr.map_to_pixels(
            input_file="",
            pickled_reader_file="",
            highest_order=10,
            ra_column="ra",
            dec_column="dec",
            resume_path="",
            mapping_key="map_0",
        )


def test_read_wrong_fileformat(small_sky_file0, tmp_path):
    """CSV file attempting to be read as parquet"""
    with pytest.raises(pa.lib.ArrowInvalid):
        mr.map_to_pixels(
            input_file=small_sky_file0,
            pickled_reader_file=pickle_file_reader(tmp_path, get_file_reader("parquet")),
            highest_order=0,
            ra_column="ra_mean",
            dec_column="dec_mean",
            resume_path="",
            mapping_key="map_0",
        )


def test_read_directory(test_data_dir, tmp_path):
    """Provide directory, not file"""
    with pytest.raises(FileNotFoundError):
        mr.map_to_pixels(
            input_file=test_data_dir,
            pickled_reader_file=pickle_file_reader(tmp_path, get_file_reader("parquet")),
            highest_order=0,
            ra_column="ra",
            dec_column="dec",
            resume_path="",
            mapping_key="map_0",
        )


def test_read_bad_fileformat(blank_data_file, capsys, tmp_path):
    """Unsupported file format"""
    with pytest.raises(NotImplementedError):
        mr.map_to_pixels(
            input_file=blank_data_file,
            pickled_reader_file=pickle_file_reader(tmp_path, None),
            highest_order=0,
            ra_column="ra",
            dec_column="dec",
            resume_path="",
            mapping_key="map_0",
        )
    captured = capsys.readouterr()
    assert "No file reader implemented" in captured.out


def read_partial_histogram(tmp_path, mapping_key):
    """Helper to read in the former result of a map operation."""
    histogram_file = tmp_path / "histograms" / f"{mapping_key}.npz"
    hist = SparseHistogram.from_file(histogram_file)
    return hist.to_array()


def test_read_single_fits(tmp_path, formats_fits):
    """Success case - fits file that exists being read as fits"""
    (tmp_path / "histograms").mkdir(parents=True)
    mr.map_to_pixels(
        input_file=formats_fits,
        pickled_reader_file=pickle_file_reader(tmp_path, get_file_reader("fits")),
        highest_order=0,
        ra_column="ra",
        dec_column="dec",
        resume_path=tmp_path,
        mapping_key="map_0",
    )

    result = read_partial_histogram(tmp_path, "map_0")
    assert len(result) == 12
    expected = hist.empty_histogram(0)
    expected[11] = 131
    npt.assert_array_equal(result, expected)


def test_map_headers_wrong(formats_headers_csv, tmp_path):
    """Test loading the a file with non-default headers (without specifying right headers)"""
    with pytest.raises(ValueError, match="columns expected but not found"):
        mr.map_to_pixels(
            input_file=formats_headers_csv,
            pickled_reader_file=pickle_file_reader(tmp_path, get_file_reader("csv")),
            highest_order=0,
            ra_column="ra",
            dec_column="dec",
            resume_path="",
            mapping_key="map_0",
        )


def test_map_headers(tmp_path, formats_headers_csv):
    """Test loading the a file with non-default headers"""
    (tmp_path / "histograms").mkdir(parents=True)
    mr.map_to_pixels(
        input_file=formats_headers_csv,
        pickled_reader_file=pickle_file_reader(tmp_path, get_file_reader("csv")),
        highest_order=0,
        ra_column="ra_mean",
        dec_column="dec_mean",
        resume_path=tmp_path,
        mapping_key="map_0",
    )

    result = read_partial_histogram(tmp_path, "map_0")

    assert len(result) == 12

    expected = hist.empty_histogram(0)
    expected[11] = 8
    npt.assert_array_equal(result, expected)
    assert (result == expected).all()


def test_map_with_healpix_29(tmp_path, formats_dir, small_sky_single_file):
    (tmp_path / "histograms").mkdir(parents=True)
    input_file = formats_dir / "spatial_index.csv"
    mr.map_to_pixels(
        input_file=input_file,
        pickled_reader_file=pickle_file_reader(tmp_path, get_file_reader("csv")),
        highest_order=0,
        ra_column="NOPE",
        dec_column="NOPE",
        use_healpix_29=True,  # radec don't matter. just use existing index
        resume_path=tmp_path,
        mapping_key="map_0",
    )

    expected = hist.empty_histogram(0)
    expected[11] = 131

    result = read_partial_histogram(tmp_path, "map_0")
    npt.assert_array_equal(result, expected)

    with pytest.raises(ValueError, match="columns expected but not found"):
        mr.map_to_pixels(
            input_file=small_sky_single_file,
            pickled_reader_file=pickle_file_reader(tmp_path, get_file_reader("csv")),
            highest_order=0,
            ra_column="NOPE",
            dec_column="NOPE",
            use_healpix_29=True,  # no pre-existing index! expect failure.
            resume_path=tmp_path,
            mapping_key="map_0",
        )


def test_map_with_schema(tmp_path, mixed_schema_csv_dir, mixed_schema_csv_parquet):
    """Test loading the a file when using a parquet schema file for dtypes"""
    (tmp_path / "histograms").mkdir(parents=True)
    input_file = mixed_schema_csv_dir / "input_01.csv"
    mr.map_to_pixels(
        input_file=input_file,
        pickled_reader_file=pickle_file_reader(
            tmp_path,
            get_file_reader(
                "csv",
                schema_file=mixed_schema_csv_parquet,
            ),
        ),
        highest_order=0,
        ra_column="ra",
        dec_column="dec",
        resume_path=tmp_path,
        mapping_key="map_0",
    )

    result = read_partial_histogram(tmp_path, "map_0")

    assert len(result) == 12

    expected = hist.empty_histogram(0)
    expected[11] = 4
    npt.assert_array_equal(result, expected)
    assert (result == expected).all()


def test_map_small_sky_order0(tmp_path, small_sky_single_file):
    """Test loading the small sky catalog and partitioning each object into the same large bucket"""
    (tmp_path / "histograms").mkdir(parents=True)
    mr.map_to_pixels(
        input_file=small_sky_single_file,
        pickled_reader_file=pickle_file_reader(tmp_path, get_file_reader("csv")),
        highest_order=0,
        ra_column="ra",
        dec_column="dec",
        resume_path=tmp_path,
        mapping_key="map_0",
    )

    result = read_partial_histogram(tmp_path, "map_0")

    assert len(result) == 12

    expected = hist.empty_histogram(0)
    expected[11] = 131
    npt.assert_array_equal(result, expected)
    assert (result == expected).all()


def test_map_small_sky_part_order1(tmp_path, small_sky_file0):
    """
    Test loading a small portion of the small sky catalog and
    partitioning objects into four smaller buckets
    """
    (tmp_path / "histograms").mkdir(parents=True)
    mr.map_to_pixels(
        input_file=small_sky_file0,
        pickled_reader_file=pickle_file_reader(tmp_path, get_file_reader("csv")),
        highest_order=1,
        ra_column="ra",
        dec_column="dec",
        resume_path=tmp_path,
        mapping_key="map_0",
    )

    result = read_partial_histogram(tmp_path, "map_0")

    assert len(result) == 48

    expected = hist.empty_histogram(1)
    filled_pixels = [5, 7, 11, 2]
    expected[44:] = filled_pixels[:]
    npt.assert_array_equal(result, expected)
    assert (result == expected).all()


def test_split_pixels_bad_format(blank_data_file, tmp_path, capsys):
    """Test error behavior, e.g. when alignment file is missing."""
    alignment = np.full(12, None)
    alignment[11] = (0, 11, 131)
    with pytest.raises(FileNotFoundError):
        mr.split_pixels(
            input_file=blank_data_file,
            pickled_reader_file="",
            highest_order=0,
            ra_column="ra_mean",
            dec_column="dec_mean",
            splitting_key="0",
            cache_shard_path=tmp_path,
            resume_path=tmp_path,
            alignment_file="",
        )
    captured = capsys.readouterr()
    assert "No such file or directory" in captured.out


def test_split_pixels_headers(formats_headers_csv, assert_parquet_file_ids, tmp_path):
    """Test loading a file with non-default headers"""
    plan = ResumePlan(tmp_path=tmp_path, progress_bar=False, input_paths=["foo1"])
    raw_histogram = np.full(12, 0)
    raw_histogram[11] = 131
    alignment_file = plan.get_alignment_file(raw_histogram, -1, 0, 0, False, 131, 1_000)
    mr.split_pixels(
        input_file=formats_headers_csv,
        pickled_reader_file=pickle_file_reader(tmp_path, get_file_reader("csv")),
        highest_order=0,
        ra_column="ra_mean",
        dec_column="dec_mean",
        splitting_key="0",
        cache_shard_path=tmp_path,
        resume_path=tmp_path,
        alignment_file=alignment_file,
    )

    file_name = tmp_path / "order_0" / "dir_0" / "pixel_11" / "shard_0_0.parquet"
    expected_ids = [*range(700, 708)]
    assert_parquet_file_ids(file_name, "object_id", expected_ids)

    file_name = tmp_path / "order_0" / "dir_0" / "pixel_1" / "shard_0_0.parquet"
    assert not os.path.exists(file_name)


def test_reduce_idempotent(parquet_shards_dir, assert_parquet_file_ids, tmp_path):
    """Reduce the shards, then reduce them again - make sure no error is thrown the second time."""
    shutil.copytree(parquet_shards_dir, tmp_path / "shards")
    shards_copy = tmp_path / "shards"
    (tmp_path / "reducing").mkdir(parents=True)

    mr.reduce_pixel_shards(
        cache_shard_path=shards_copy,
        resume_path=tmp_path,
        reducing_key="0_11",
        destination_pixel_order=0,
        destination_pixel_number=11,
        destination_pixel_size=131,
        output_path=tmp_path,
        add_healpix_29=True,
        ra_column="ra",
        dec_column="dec",
        sort_columns="id",
        delete_input_files=False,
    )

    output_file = tmp_path / "dataset" / "Norder=0" / "Dir=0" / "Npix=11.parquet"

    expected_ids = [*range(700, 831)]
    assert_parquet_file_ids(output_file, "id", expected_ids)

    ## Confirm idempotency of reduce operation.
    mr.reduce_pixel_shards(
        cache_shard_path=shards_copy,
        resume_path=tmp_path,
        reducing_key="0_11",
        destination_pixel_order=0,
        destination_pixel_number=11,
        destination_pixel_size=131,
        output_path=tmp_path,
        add_healpix_29=True,
        ra_column="ra",
        dec_column="dec",
        sort_columns="id",
        delete_input_files=True,
    )

    output_file = tmp_path / "dataset" / "Norder=0" / "Dir=0" / "Npix=11.parquet"

    expected_ids = [*range(700, 831)]
    assert_parquet_file_ids(output_file, "id", expected_ids)

    ## It's even ok to try to delete the files again!
    mr.reduce_pixel_shards(
        cache_shard_path=shards_copy,
        resume_path=tmp_path,
        reducing_key="0_11",
        destination_pixel_order=0,
        destination_pixel_number=11,
        destination_pixel_size=131,
        output_path=tmp_path,
        add_healpix_29=True,
        ra_column="ra",
        dec_column="dec",
        sort_columns="id",
        delete_input_files=True,
    )

    output_file = tmp_path / "dataset" / "Norder=0" / "Dir=0" / "Npix=11.parquet"

    expected_ids = [*range(700, 831)]
    assert_parquet_file_ids(output_file, "id", expected_ids)


def test_reduce_order0(parquet_shards_dir, assert_parquet_file_ids, tmp_path):
    """Test reducing into one large pixel"""
    (tmp_path / "reducing").mkdir(parents=True)
    mr.reduce_pixel_shards(
        cache_shard_path=parquet_shards_dir,
        resume_path=tmp_path,
        reducing_key="0_11",
        destination_pixel_order=0,
        destination_pixel_number=11,
        destination_pixel_size=131,
        output_path=tmp_path,
        add_healpix_29=True,
        ra_column="ra",
        dec_column="dec",
        sort_columns="id",
        delete_input_files=False,
    )

    output_file = tmp_path / "dataset" / "Norder=0" / "Dir=0" / "Npix=11.parquet"

    expected_ids = [*range(700, 831)]
    assert_parquet_file_ids(output_file, "id", expected_ids)


def test_reduce_healpix_29(parquet_shards_dir, assert_parquet_file_ids, tmp_path):
    """Test reducing with or without a _healpix_29 field"""
    (tmp_path / "reducing").mkdir(parents=True)
    mr.reduce_pixel_shards(
        cache_shard_path=parquet_shards_dir,
        resume_path=tmp_path,
        reducing_key="0_11",
        destination_pixel_order=0,
        destination_pixel_number=11,
        destination_pixel_size=131,
        output_path=tmp_path,
        ra_column="ra",
        dec_column="dec",
        sort_columns="id",
        delete_input_files=False,
    )

    output_file = tmp_path / "dataset" / "Norder=0" / "Dir=0" / "Npix=11.parquet"

    expected_ids = [*range(700, 831)]
    assert_parquet_file_ids(output_file, "id", expected_ids)
    data_frame = pd.read_parquet(output_file, engine="pyarrow")
    npt.assert_array_equal(
        data_frame.columns,
        ["_healpix_29", "id", "ra", "dec", "ra_error", "dec_error"],
    )

    output_file.unlink()

    mr.reduce_pixel_shards(
        cache_shard_path=parquet_shards_dir,
        resume_path=tmp_path,
        reducing_key="0_11",
        destination_pixel_order=0,
        destination_pixel_number=11,
        destination_pixel_size=131,
        output_path=tmp_path,
        add_healpix_29=False,  ## different from above
        ra_column="ra",
        dec_column="dec",
        sort_columns="id",
        delete_input_files=False,
    )

    assert_parquet_file_ids(output_file, "id", expected_ids)
    data_frame = pd.read_parquet(output_file, engine="pyarrow")
    ## No index name.
    assert data_frame.index.name is None
    ## Data fields are the same.
    npt.assert_array_equal(
        data_frame.columns,
        ["id", "ra", "dec", "ra_error", "dec_error"],
    )


def test_reduce_bad_expectation(parquet_shards_dir, tmp_path):
    """Test reducing into one large pixel"""
    with pytest.raises(ValueError, match="Unexpected number of objects"):
        mr.reduce_pixel_shards(
            cache_shard_path=parquet_shards_dir,
            resume_path=tmp_path,
            reducing_key="0_11",
            destination_pixel_order=0,
            destination_pixel_number=11,
            destination_pixel_size=11,  ## should be 131
            output_path=tmp_path,
            ra_column="ra",
            dec_column="dec",
            sort_columns="id",
            delete_input_files=False,
        )


def test_reduce_with_sorting_complex(assert_parquet_file_ids, tmp_path):
    """Test reducing and requesting specific sort columns.

    Logically, the input data has a mix of orderings in files, object IDs, and timestamps.
    Each source is partitioned according to the linked object's radec, and so will be
    ordered within the same spatial_index value.

    First, we take some time to set up these silly data points, then we test out
    reducing them into a single parquet file using a mix of reduction options.
    """
    (tmp_path / "reducing").mkdir(parents=True)
    shard_dir = tmp_path / "reduce_shards" / "order_0" / "dir_0" / "pixel_11"
    shard_dir.mkdir(parents=True)
    output_file = tmp_path / "dataset" / "Norder=0" / "Dir=0" / "Npix=11.parquet"

    file1_string = """source_id,object_id,time,ra,dec
1200,700,3000,282.5,-58.5
1201,700,4000,282.5,-58.5
1402,702,3000,310.5,-27.5
1403,702,3100,310.5,-27.5
1404,702,3200,310.5,-27.5
1505,703,4000,286.5,-69.5"""
    file1_data = pd.read_csv(StringIO(file1_string))
    file1_data.to_parquet(shard_dir / "file_1_shard_1.parquet")

    file2_string = """source_id,object_id,time,ra,dec
1206,700,2000,282.5,-58.5
1307,701,2200,299.5,-48.5
1308,701,2100,299.5,-48.5
1309,701,2000,299.5,-48.5"""
    file2_data = pd.read_csv(StringIO(file2_string))
    file2_data.to_parquet(shard_dir / "file_2_shard_1.parquet")

    combined_data = pd.concat([file1_data, file2_data])
    combined_data["norder19_healpix"] = hp.radec2pix(
        19,
        combined_data["ra"].values,
        combined_data["dec"].values,
    )
    ## Use this to prune generated columns like Norder, Npix, and _healpix_29
    comparison_columns = ["source_id", "object_id", "time", "ra", "dec"]

    ######################## Sort option 1: by source_id
    ## This will sort WITHIN an order 19 healpix pixel. In that ordering, the objects are
    ## (703, 700, 701, 702)
    mr.reduce_pixel_shards(
        cache_shard_path=tmp_path / "reduce_shards",
        resume_path=tmp_path,
        reducing_key="0_11",
        destination_pixel_order=0,
        destination_pixel_number=11,
        destination_pixel_size=10,
        output_path=tmp_path,
        ra_column="ra",
        dec_column="dec",
        sort_columns="source_id",
        delete_input_files=False,
    )

    ## sort order is effectively (norder19 healpix, source_id)
    data_frame = pd.read_parquet(output_file, engine="pyarrow")
    expected_dataframe = combined_data.sort_values(["norder19_healpix", "source_id"], kind="stable")
    pd.testing.assert_frame_equal(
        expected_dataframe[comparison_columns].reset_index(drop=True),
        data_frame[comparison_columns].reset_index(drop=True),
    )
    assert_parquet_file_ids(
        output_file,
        "source_id",
        [1505, 1200, 1201, 1206, 1307, 1308, 1309, 1402, 1403, 1404],
        resort_ids=False,
    )

    assert_parquet_file_ids(
        output_file,
        "object_id",
        [703, 700, 700, 700, 701, 701, 701, 702, 702, 702],
        resort_ids=False,
    )

    ######################## Sort option 2: by object id and time
    ## sort order is effectively (norder19 healpix, object id, time)
    output_file.unlink()
    mr.reduce_pixel_shards(
        cache_shard_path=tmp_path / "reduce_shards",
        resume_path=tmp_path,
        reducing_key="0_11",
        destination_pixel_order=0,
        destination_pixel_number=11,
        destination_pixel_size=10,
        output_path=tmp_path,
        ra_column="ra",
        dec_column="dec",
        sort_columns="object_id,time",
        delete_input_files=False,
    )

    data_frame = pd.read_parquet(output_file, engine="pyarrow")
    expected_dataframe = combined_data.sort_values(["norder19_healpix", "object_id", "time"], kind="stable")
    pd.testing.assert_frame_equal(
        expected_dataframe[comparison_columns].reset_index(drop=True),
        data_frame[comparison_columns].reset_index(drop=True),
    )
    assert_parquet_file_ids(
        output_file,
        "source_id",
        [1505, 1206, 1200, 1201, 1309, 1308, 1307, 1402, 1403, 1404],
        resort_ids=False,
    )
    assert_parquet_file_ids(
        output_file,
        "time",
        [4000, 2000, 3000, 4000, 2000, 2100, 2200, 3000, 3100, 3200],
        resort_ids=False,
    )

    ######################## Sort option 3: by object id and time WITHOUT spatial index.
    ## The 1500 block of ids goes back to the end, because we're not using
    ## spatial properties for sorting, only numeric.
    ## sort order is effectively (object id, time)
    output_file.unlink()
    mr.reduce_pixel_shards(
        cache_shard_path=tmp_path / "reduce_shards",
        resume_path=tmp_path,
        reducing_key="0_11",
        destination_pixel_order=0,
        destination_pixel_number=11,
        destination_pixel_size=10,
        output_path=tmp_path,
        ra_column="ra",
        dec_column="dec",
        sort_columns="object_id,time",
        add_healpix_29=False,
        delete_input_files=False,
    )

    data_frame = pd.read_parquet(output_file, engine="pyarrow")
    expected_dataframe = combined_data.sort_values(["object_id", "time"], kind="stable")
    pd.testing.assert_frame_equal(
        expected_dataframe[comparison_columns].reset_index(drop=True),
        data_frame[comparison_columns].reset_index(drop=True),
    )
    assert_parquet_file_ids(
        output_file,
        "source_id",
        [1206, 1200, 1201, 1309, 1308, 1307, 1402, 1403, 1404, 1505],
        resort_ids=False,
    )


def test_map_reduce_histogram_type(tmp_path):
    """Test that map/reduce operations respect histogram_type and throws error on mismatch.

    Note that this test does not run the full map/reduce, but checks the type logic for
    partitioning. For full integration, see run_import tests.
    """
    # Simulate a ResumePlan with row_count histogram
    plan = ResumePlan(tmp_path=tmp_path, progress_bar=False, input_paths=["foo1"])
    assert plan.histogram_type == "row_count"

    # Simulate a ResumePlan with mem_size histogram by passing byte_pixel_threshold
    class TestArgs:
        input_paths = ["foo1"]
        debug_stats_only = False

        def resume_kwargs_dict(self):
            return {"tmp_path": tmp_path}

        byte_pixel_threshold = 123

    plan_mem = ResumePlan(tmp_path=tmp_path, progress_bar=False, import_args=TestArgs())
    assert plan_mem.histogram_type == "mem_size"

    # Validate that error is raised if wrong type is validated
    import pytest

    with pytest.raises(ValueError, match="histogram type"):
        plan_mem.validate_histogram_type("row_count")
    with pytest.raises(ValueError, match="histogram type"):
        plan.validate_histogram_type("mem_size")

    # If types match, no error
    plan_mem.validate_histogram_type("mem_size")
    plan.validate_histogram_type("row_count")


def test_histogram_file_contents(tmp_path, small_sky_single_file):
    """Test that map_to_pixels writes correct histogram file contents for both row_count and mem_size."""
    (tmp_path / "histograms").mkdir(parents=True)

    # Row count histogram (default)
    mr.map_to_pixels(
        input_file=small_sky_single_file,
        pickled_reader_file=pickle_file_reader(tmp_path, get_file_reader("csv")),
        highest_order=0,
        ra_column="ra",
        dec_column="dec",
        resume_path=tmp_path,
        mapping_key="map_0",
    )
    # Check row_count histogram contents
    histogram_file = tmp_path / "histograms" / "map_0.npz"
    hist_obj = SparseHistogram.from_file(histogram_file)
    arr = hist_obj.to_array()
    assert len(arr) == 12
    assert arr[11] == 131  # known value for this fixture
    assert arr.sum() == 131

    # Now simulate mem_size histogram by passing byte_pixel_threshold
    mr.map_to_pixels(
        input_file=small_sky_single_file,
        pickled_reader_file=pickle_file_reader(tmp_path, get_file_reader("csv")),
        highest_order=0,
        ra_column="ra",
        dec_column="dec",
        resume_path=tmp_path,
        mapping_key="map_1",
        use_byte_threshold_histogram=True,  # triggers mem_size histogram logic
    )
    histogram_file_mem = tmp_path / "histograms" / "map_1.npz"
    hist_obj_mem = SparseHistogram.from_file(histogram_file_mem)
    arr_mem = hist_obj_mem.to_array()
    assert len(arr_mem) == 12
    # For mem_size, the sum should be greater than or equal to row count (since it's bytes)
    assert arr_mem[11] >= 131
    assert arr_mem.sum() >= 131
    # Note: For some datasets, mem_size and row_count histograms may be identical.


def test_get_mem_size_of_chunk(tmp_path, small_sky_single_file):
    """Test the _get_mem_size_of_chunk function for reasonable outputs."""
    # Test with an empty DataFrame
    empty_df = pd.DataFrame(columns=["id", "ra", "dec", "value"])
    mem_sizes_empty = mr._get_mem_size_of_chunk(empty_df)
    assert len(mem_sizes_empty) == 0

    # Test with a small DataFrame
    df = pd.DataFrame(
        {
            "id": [0, 0, 0, 1, 1, 1, 2, 2, 2, 2],
            "ra": [10.0, 10.0, 10.0, 15.0, 15.0, 15.0, 12.1, 12.1, 12.1, 12.1],
            "dec": [0.0, 0.0, 0.0, -1.0, -1.0, -1.0, 0.5, 0.5, 0.5, 0.5],
            "time": [
                60676.0,
                60677.0,
                60678.0,
                60675.0,
                60676.5,
                60677.0,
                60676.6,
                60676.7,
                60676.8,
                60676.9,
            ],
            "brightness": [100.0, 101.0, 99.8, 5.0, 5.01, 4.98, 20.1, 20.5, 20.3, 20.2],
            "band": ["g", "r", "g", "r", "g", "r", "g", "g", "r", "r"],
        }
    )
    mem_sizes = mr._get_mem_size_of_chunk(df)
    # Since we have 10 rows, mem_sizes should have length 10
    assert len(mem_sizes) == 10
    # Each entry should be a positive integer (size in bytes)
    assert all(isinstance(size, int) and size > 0 for size in mem_sizes)

    # Compare to a smaller DataFrame with fewer columns
    df_small = df[["id", "ra", "dec"]]
    mem_sizes_small = mr._get_mem_size_of_chunk(df_small)
    assert len(mem_sizes_small) == 10
    assert all(isinstance(size, int) and size > 0 for size in mem_sizes_small)
    # Each entry in mem_sizes should be > corresponding entry in mem_sizes_small
    assert all(m > s for m, s in zip(mem_sizes, mem_sizes_small, strict=True))

    # Test with a pyarrow Table
    table = pa.Table.from_pandas(df)
    mem_sizes_table = mr._get_mem_size_of_chunk(table)
    assert len(mem_sizes_table) == 10
    assert all(isinstance(size, int) and size > 0 for size in mem_sizes_table)

    # Test with a smaller pyarrow Table
    table_small = pa.Table.from_pandas(df_small)
    mem_sizes_table_small = mr._get_mem_size_of_chunk(table_small)
    assert len(mem_sizes_table_small) == 10
    assert all(isinstance(size, int) and size > 0 for size in mem_sizes_table_small)
    # Each entry in mem_sizes_table should be > corresponding entry in mem_sizes_table_small
    assert all(m > s for m, s in zip(mem_sizes_table, mem_sizes_table_small, strict=True))


def test_get_mem_size_of_chunk_nested(tmp_path):
    """Test the _get_mem_size_of_chunk function with nested data."""
    pytest.importorskip("nested_pandas")
    from nested_pandas.nestedframe import NestedFrame

    # Create a small DataFrame and nest it
    df = pd.DataFrame(
        {
            "id": [0, 0, 0, 1, 1, 1, 2, 2, 2, 2],
            "ra": [10.0, 10.0, 10.0, 15.0, 15.0, 15.0, 12.1, 12.1, 12.1, 12.1],
            "dec": [0.0, 0.0, 0.0, -1.0, -1.0, -1.0, 0.5, 0.5, 0.5, 0.5],
            "time": [
                60676.0,
                60677.0,
                60678.0,
                60675.0,
                60676.5,
                60677.0,
                60676.6,
                60676.7,
                60676.8,
                60676.9,
            ],
            "brightness": [100.0, 101.0, 99.8, 5.0, 5.01, 4.98, 20.1, 20.5, 20.3, 20.2],
            "band": ["g", "r", "g", "r", "g", "r", "g", "g", "r", "r"],
        }
    )
    nf = NestedFrame.from_flat(
        df,
        base_columns=["ra", "dec"],
        nested_columns=["time", "brightness", "band"],
        on="id",
        name="lightcurve",
    )

    # Calculate memory sizes
    mem_sizes = mr._get_mem_size_of_chunk(nf)

    # Since we have only 3 rows once we nest, mem_sizes should have length 3
    assert len(mem_sizes) == 3
    # Each entry should be a positive integer (size in bytes)
    assert all(isinstance(size, int) and size > 0 for size in mem_sizes)
    # The first two entries should be the same, since they have 3 sub-rows each
    assert mem_sizes[0] == mem_sizes[1]
    # The last entry should be the larger than the other two, since it has 4 sub-rows
    assert mem_sizes[2] > mem_sizes[0]
