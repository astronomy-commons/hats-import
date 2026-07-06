"""Tests of per-row memory size estimation for mem_size thresholding."""

import cloudpickle
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from hats.io import size_estimates
from hats.pixel_math.sparse_histogram import SparseHistogram

import hats_import.catalog.map_reduce as mr


def pickle_file_reader(tmp_path, file_reader) -> str:
    """Utility method to pickle a file reader, and return path to pickle."""
    pickled_reader_file = tmp_path / "reader.pickle"
    with open(pickled_reader_file, "wb") as pickle_file:
        cloudpickle.dump(file_reader, pickle_file)
    return pickled_reader_file


def read_partial_histogram(tmp_path, mapping_key, which_histogram="row_count"):
    """Helper to read in the former result of a map operation."""
    histogram_file = tmp_path / f"{which_histogram}_histograms" / f"{mapping_key}.npz"
    histogram = SparseHistogram.from_file(histogram_file)
    return histogram.to_array()


class SingleChunkReader:
    """Test reader that yields a single pre-built chunk, ignoring the input file."""

    # pylint: disable=too-few-public-methods

    def __init__(self, chunk):
        self.chunk = chunk

    def read(self, input_file, read_columns=None):  # pylint: disable=unused-argument
        chunk = self.chunk
        if read_columns is not None:
            chunk = chunk[read_columns] if isinstance(chunk, pd.DataFrame) else chunk.select(read_columns)
        yield chunk


def test_get_cols_in_input_file_pandas(tmp_path):
    """String columns are precomputed alongside fixed-length columns; list columns stay
    variable-length."""
    chunk = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "flag": [True, False, True, False],
            "id_str": ["1001", "1002", "1003", "1004"],
            "list_o_strings": pd.array(["a", "bb", "ccc", "dddd"], dtype="string"),
            "mags": [np.arange(i, dtype=np.float64) for i in range(4)],
        }
    )
    reader_file = pickle_file_reader(tmp_path, SingleChunkReader(chunk))

    var_cols, precomputed_cols, precomputed_row_size = mr._get_cols_in_input_file("unused", reader_file)

    assert var_cols == ["mags"]
    assert precomputed_cols == ["id", "flag", "id_str", "list_o_strings"]
    expected = np.mean(size_estimates.get_mem_size_per_row(chunk, cols=precomputed_cols))
    assert precomputed_row_size == pytest.approx(expected)


def test_get_cols_in_input_file_pyarrow(tmp_path):
    """Same classification for pyarrow tables, including large_string and binary types."""
    chunk = pa.table(
        {
            "id": pa.array([1, 2, 3, 4], type=pa.int64()),
            "id_str": pa.array(["1001", "1002", "1003", "1004"], type=pa.large_string()),
            "blob": pa.array([b"ab", b"cd", b"ef", b"gh"], type=pa.binary()),
            "mags": pa.array([[1.0], [1.0, 2.0], [], [3.0]], type=pa.list_(pa.float64())),
        }
    )
    reader_file = pickle_file_reader(tmp_path, SingleChunkReader(chunk))

    var_cols, precomputed_cols, precomputed_row_size = mr._get_cols_in_input_file("unused", reader_file)

    assert var_cols == ["mags"]
    assert precomputed_cols == ["id", "id_str", "blob"]
    expected = np.mean(size_estimates.get_mem_size_per_row(chunk, cols=precomputed_cols))
    assert precomputed_row_size == pytest.approx(expected)


def test_get_cols_in_input_file_object_dtype_lists(tmp_path):
    """An object-dtype column holding lists is not string-like, and must be
    measured per-row."""
    chunk = pd.DataFrame(
        {
            "id": [1, 2],
            "listed": [[1, 2, 3], [4]],
        }
    )
    reader_file = pickle_file_reader(tmp_path, SingleChunkReader(chunk))

    var_cols, precomputed_cols, _ = mr._get_cols_in_input_file("unused", reader_file)

    assert var_cols == ["listed"]
    assert precomputed_cols == ["id"]


def test_get_cols_in_input_file_inconsistent_strings_pandas(tmp_path):
    """A string column whose value sizes vary wildly (e.g. serialized arrays)
    is demoted to per-row measurement; consistent string columns are not."""
    chunk = pd.DataFrame(
        {
            "id_str": ["1001", "1002", "1003", "1004"],
            "lightcurve": ["1.0", "1.0,2.0", "1.5", ",".join(["19.5"] * 200)],
        }
    )
    reader_file = pickle_file_reader(tmp_path, SingleChunkReader(chunk))

    var_cols, precomputed_cols, precomputed_row_size = mr._get_cols_in_input_file("unused", reader_file)

    assert var_cols == ["lightcurve"]
    assert precomputed_cols == ["id_str"]
    expected = np.mean(size_estimates.get_mem_size_per_row(chunk, cols=["id_str"]))
    assert precomputed_row_size == pytest.approx(expected)


def test_get_cols_in_input_file_inconsistent_strings_pyarrow(tmp_path):
    """Same demotion of wildly-varying string columns for pyarrow tables."""
    chunk = pa.table(
        {
            "id_str": pa.array(["1001", "1002", "1003", "1004"], type=pa.string()),
            "lightcurve": pa.array(["1.0", "1.0,2.0", "1.5", ",".join(["19.5"] * 200)], type=pa.string()),
        }
    )
    reader_file = pickle_file_reader(tmp_path, SingleChunkReader(chunk))

    var_cols, precomputed_cols, _ = mr._get_cols_in_input_file("unused", reader_file)

    assert var_cols == ["lightcurve"]
    assert precomputed_cols == ["id_str"]


def test_string_col_sizes_are_consistent():
    """Direct checks of the consistency guard's boundary behavior."""
    consistent = pd.DataFrame({"strings": ["1001", "1002", "10003", "999"]})
    assert mr._string_col_sizes_are_consistent(consistent, "strings")

    skewed = pd.DataFrame({"strings": ["a", "b", "c", "x" * 500]})
    assert not mr._string_col_sizes_are_consistent(skewed, "strings")


def test_map_to_pixels_mem_size_mixed_columns(tmp_path):
    """map_to_pixels in mem_size mode with both precomputed (numeric, string) and
    variable-length (list) columns: histogram total matches the expected estimate."""
    chunk = pd.DataFrame(
        {
            "ra": [290.0, 300.0, 310.0, 320.0],
            "dec": [-60.0, -50.0, -55.0, -45.0],
            "id_str": ["1001", "1002", "1003", "1004"],
            "mags": [np.arange(i + 1, dtype=np.float64) for i in range(4)],
        }
    )
    reader_file = pickle_file_reader(tmp_path, SingleChunkReader(chunk))
    (tmp_path / "row_count_histograms").mkdir(parents=True)
    (tmp_path / "mem_size_histograms").mkdir(parents=True)

    mr.map_to_pixels(
        input_file="unused",
        pickled_reader_file=reader_file,
        highest_order=0,
        ra_column="ra",
        dec_column="dec",
        resume_path=tmp_path,
        mapping_key="map_0",
        threshold_mode="mem_size",
    )

    result = read_partial_histogram(tmp_path, "map_0", which_histogram="mem_size")
    _, precomputed_cols, precomputed_row_size = mr._get_cols_in_input_file("unused", reader_file)
    assert precomputed_cols == ["ra", "dec", "id_str"]
    # Measure the chunk as the pipeline saw it (after the reader's pickle round-trip),
    # since unpickled ndarray cells can report a different object overhead.
    with open(reader_file, "rb") as pickle_file:
        unpickled_chunk = cloudpickle.load(pickle_file).chunk
    var_sizes = size_estimates.get_mem_size_per_row(unpickled_chunk, cols=["mags"])
    expected_total = sum(var_sizes) + len(chunk) * precomputed_row_size
    # Fractional bytes are truncated as each row is summed into the int64 histogram,
    # so allow up to a byte of loss per row.
    assert expected_total - len(chunk) <= result.sum() <= expected_total
