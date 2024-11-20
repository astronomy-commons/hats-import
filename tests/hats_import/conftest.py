"""Fixtures for testing import tool actions."""

import os
import re
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
from dask.distributed import Client
from hats import pixel_math

# pylint: disable=missing-function-docstring, redefined-outer-name


@pytest.fixture(scope="session", name="dask_client")
def dask_client():
    """Create a single client for use by all unit test cases."""
    client = Client(n_workers=1, threads_per_worker=1)
    yield client
    client.close()


def pytest_collection_modifyitems(items):
    """Modify dask unit tests to
        - ignore event loop deprecation warnings
        - have a longer timeout default timeout (5 seconds instead of 1 second)
        - require use of the `dask_client` fixture, even if it's not requsted

    Individual tests that will be particularly long-running can still override
    the default timeout, by using an annotation like:

        @pytest.mark.dask(timeout=10)
        def test_long_running():
            ...
    """
    first_dask = True
    for item in items:
        timeout = None
        for mark in item.iter_markers(name="dask"):
            timeout = 10
            if "timeout" in mark.kwargs:
                timeout = int(mark.kwargs.get("timeout"))
        if timeout:
            if first_dask:
                ## The first test requires more time to set up the dask client
                timeout += 10
                first_dask = False
            item.add_marker(pytest.mark.timeout(timeout))
            item.add_marker(pytest.mark.usefixtures("dask_client"))
            item.add_marker(pytest.mark.filterwarnings("ignore::DeprecationWarning"))


TEST_DIR = os.path.dirname(__file__)


@pytest.fixture
def test_data_dir():
    return Path(TEST_DIR).parent / "data"


@pytest.fixture
def small_sky_dir(test_data_dir):
    return test_data_dir / "small_sky"


@pytest.fixture
def small_sky_single_file(test_data_dir):
    return test_data_dir / "small_sky" / "catalog.csv"


@pytest.fixture
def small_sky_object_catalog(test_data_dir):
    return test_data_dir / "small_sky_object_catalog"


@pytest.fixture
def small_sky_source_dir(test_data_dir):
    return test_data_dir / "small_sky_source"


@pytest.fixture
def small_sky_source_catalog(test_data_dir):
    return test_data_dir / "small_sky_source_catalog"


@pytest.fixture
def blank_data_dir(test_data_dir):
    return test_data_dir / "blank"


@pytest.fixture
def blank_data_file(test_data_dir):
    return test_data_dir / "blank" / "blank.csv"


@pytest.fixture
def empty_data_dir(test_data_dir):
    return test_data_dir / "empty"


@pytest.fixture
def formats_dir(test_data_dir):
    return test_data_dir / "test_formats"


@pytest.fixture
def formats_headers_csv(test_data_dir):
    return test_data_dir / "test_formats" / "headers.csv"


@pytest.fixture
def formats_pipe_csv(test_data_dir):
    return test_data_dir / "test_formats" / "pipe_delimited.csv"


@pytest.fixture
def formats_fits(test_data_dir):
    return test_data_dir / "test_formats" / "small_sky.fits"


@pytest.fixture
def formats_pandasindex(test_data_dir):
    return test_data_dir / "test_formats" / "pandasindex.parquet"


@pytest.fixture
def indexed_files_dir(test_data_dir):
    return test_data_dir / "indexed_files"


@pytest.fixture
def small_sky_parts_dir(test_data_dir):
    return test_data_dir / "small_sky_parts"


@pytest.fixture
def small_sky_file0(test_data_dir):
    return test_data_dir / "small_sky_parts" / "catalog_00_of_05.csv"


@pytest.fixture
def parquet_shards_dir(test_data_dir):
    return test_data_dir / "parquet_shards"


@pytest.fixture
def soap_intermediate_dir(test_data_dir):
    return test_data_dir / "soap_intermediate"


@pytest.fixture
def parquet_shards_shard_44_0(test_data_dir):
    return test_data_dir / "parquet_shards" / "order_1" / "dir_0" / "pixel_44" / "shard_3_0.parquet"


@pytest.fixture
def mixed_schema_csv_dir(test_data_dir):
    return test_data_dir / "mixed_schema"


@pytest.fixture
def mixed_schema_csv_parquet(test_data_dir):
    return test_data_dir / "mixed_schema" / "schema.parquet"


@pytest.fixture
def resume_dir(test_data_dir):
    return test_data_dir / "resume"


@pytest.fixture
def basic_data_shard_df():
    ras = np.arange(0.0, 360.0)
    dec = np.full(360, 0.0)
    norder = np.full(360, 1)
    npix = np.full(360, 0)
    spatial_indexes = pixel_math.compute_spatial_index(ras, dec)

    test_df = pd.DataFrame(
        data=zip(spatial_indexes, ras, dec, norder, npix),
        columns=[
            "_healpix_29",
            "weird_ra",
            "weird_dec",
            "Norder",
            "Npix",
        ],
    )
    return test_df


@pytest.fixture
def polar_data_shard_df():
    ras = np.arange(0.0, 360.0)
    dec = np.full(360, 89.9)
    norder = np.full(360, 2)
    npix = np.full(360, 0)
    spatial_indexes = pixel_math.compute_spatial_index(ras, dec)

    test_df = pd.DataFrame(
        data=zip(spatial_indexes, ras, dec, norder, npix),
        columns=[
            "_healpix_29",
            "weird_ra",
            "weird_dec",
            "Norder",
            "Npix",
        ],
    )

    return test_df


@pytest.fixture
def assert_text_file_matches():
    def assert_text_file_matches(expected_lines, file_name):
        """Convenience method to read a text file and compare the contents, line for line.

        When file contents get even a little bit big, it can be difficult to see
        the difference between an actual file and the expected contents without
        increased testing verbosity. This helper compares files line-by-line,
        using the provided strings or regular expressions.

        Notes:
            Because we check strings as regular expressions, you may need to escape some
            contents of `expected_lines`.

        Args:
            expected_lines(:obj:`string array`) list of strings, formatted as regular expressions.
            file_name (str): fully-specified path of the file to read
        """
        assert os.path.exists(file_name), f"file not found [{file_name}]"
        with open(
            file_name,
            "r",
            encoding="utf-8",
        ) as metadata_file:
            contents = metadata_file.readlines()

        assert len(expected_lines) == len(
            contents
        ), f"files not the same length ({len(contents)} vs {len(expected_lines)})"
        for i, expected in enumerate(expected_lines):
            assert re.match(expected, contents[i]), (
                f"files do not match at line {i+1} " f"(actual: [{contents[i]}] vs expected: [{expected}])"
            )

    return assert_text_file_matches


@pytest.fixture
def assert_parquet_file_ids():
    def assert_parquet_file_ids(file_name, id_column, expected_ids, resort_ids=True):
        """
        Convenience method to read a parquet file and compare the object IDs to
        a list of expected objects.

        Args:
            file_name (str): fully-specified path of the file to read
            id_column (str): column in the parquet file to read IDs from
            expected_ids (:obj:`int[]`): list of expected ids in `id_column`
        """
        assert os.path.exists(file_name), f"file not found [{file_name}]"

        data_frame = pd.read_parquet(file_name, engine="pyarrow")
        assert id_column in data_frame.columns
        ids = data_frame[id_column].tolist()
        if resort_ids:
            ids.sort()
            expected_ids.sort()

        assert len(ids) == len(
            expected_ids
        ), f"object list not the same size ({len(ids)} vs {len(expected_ids)})"

        npt.assert_array_equal(ids, expected_ids)

    return assert_parquet_file_ids


@pytest.fixture
def assert_parquet_file_index():
    def assert_parquet_file_index(file_name, expected_values):
        """
        Convenience method to read a parquet file and compare the index values to
        a list of expected objects.

        Args:
            file_name (str): fully-specified path of the file to read
            expected_values (:obj:`int[]`): list of expected values in index
        """
        assert os.path.exists(file_name), f"file not found [{file_name}]"

        data_frame = pd.read_parquet(file_name, engine="pyarrow")
        values = data_frame.index.values.tolist()
        values.sort()
        expected_values.sort()

        assert len(values) == len(
            expected_values
        ), f"object list not the same size ({len(values)} vs {len(expected_values)})"

        npt.assert_array_equal(values, expected_values)

    return assert_parquet_file_index
