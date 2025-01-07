"""Test resume file operations"""

from pathlib import Path

import numpy.testing as npt
import pytest

from hats_import.pipeline_resume_plan import PipelineResumePlan, get_formatted_stage_name


def test_done_file(tmp_path):
    """Verify expected behavior of done file"""
    plan = PipelineResumePlan(tmp_path=tmp_path, progress_bar=False)
    done_file = "action_done"

    ## done file should not exist
    assert not plan.done_file_exists(done_file)

    ## mark done and check that we can find it
    plan.touch_stage_done_file(done_file)
    assert plan.done_file_exists(done_file)

    ## mark done is idempotent
    plan.touch_stage_done_file(done_file)
    assert plan.done_file_exists(done_file)

    ## clear all the resume files and verify done file is gone.
    plan.clean_resume_files()

    assert not plan.done_file_exists(done_file)


def test_safe_to_resume(tmp_path):
    """Check that we throw errors when it's not safe to resume."""
    plan = PipelineResumePlan(tmp_path=tmp_path, progress_bar=False, resume=False)
    plan.safe_to_resume()

    ## check is idempotent - intermediate directory exists but does not
    ## contain files.
    plan.safe_to_resume()

    ## take some action and verify that we can no longer resume without
    ## explicitly setting resume=True
    done_file = "action_done"
    plan.touch_stage_done_file(done_file)
    plan.safe_to_resume()

    ## If we mark as a resuming pipeline, we're safe to resume.
    resuming_plan = PipelineResumePlan(tmp_path=tmp_path, progress_bar=False, resume=True)
    resuming_plan.safe_to_resume()

    ## If there are no more intermediate files, we don't need to set resume.
    plan.clean_resume_files()
    plan.safe_to_resume()


@pytest.mark.dask
def test_wait_for_futures(tmp_path, dask_client):
    """Test that we can wait around for futures to complete.

    Additionally test that relevant parts of the traceback are printed to stdout."""
    plan = PipelineResumePlan(tmp_path=tmp_path, progress_bar=False, resume=False)

    def error_on_even(argument):
        """Silly little method used to test futures that fail under predictable conditions"""
        if argument % 2 == 0:
            raise RuntimeError("we are at odds with evens")

    ## Everything is fine if we're all odd
    futures = [dask_client.submit(error_on_even, 1)]
    plan.wait_for_futures(futures, "test")

    ## Throw an even in the mix, and we'll see some stages fail. Should cause whole stage to fail.
    futures = [dask_client.submit(error_on_even, 1), dask_client.submit(error_on_even, 2)]
    with pytest.raises(RuntimeError, match="Some test stages failed"):
        plan.wait_for_futures(futures, "test")


@pytest.mark.dask
def test_wait_for_futures_progress(tmp_path, dask_client, capsys):
    """Test that we can wait around for futures to complete.

    Additionally test that relevant parts of the traceback are printed to stdout."""
    plan = PipelineResumePlan(tmp_path=tmp_path, progress_bar=True, simple_progress_bar=True, resume=False)

    def error_on_even(argument):
        """Silly little method used to test futures that fail under predictable conditions"""
        if argument % 2 == 0:
            raise RuntimeError("we are at odds with evens")

    ## Everything is fine if we're all odd, but use a silly name so it's
    ## clear that the stage name is present, and well-formatted.
    futures = [dask_client.submit(error_on_even, 1)]
    plan.wait_for_futures(futures, "teeeest")

    captured = capsys.readouterr()
    assert "Teeeest" in captured.err
    assert "100%" in captured.err


@pytest.mark.dask
def test_wait_for_futures_fail_fast(tmp_path, dask_client):
    """Test that we can wait around for futures to complete.

    Additionally test that relevant parts of the traceback are printed to stdout."""
    plan = PipelineResumePlan(tmp_path=tmp_path, progress_bar=False, resume=False)

    def error_on_even(argument):
        """Silly little method used to test futures that fail under predictable conditions"""
        if argument % 2 == 0:
            raise RuntimeError("we are at odds with evens")

    futures = [dask_client.submit(error_on_even, 3), dask_client.submit(error_on_even, 4)]
    with pytest.raises(RuntimeError, match="we are at odds with evens"):
        plan.wait_for_futures(futures, "test", fail_fast=True)


def test_formatted_stage_name():
    """Test that we make pretty stage names for presenting in progress bars"""
    formatted = get_formatted_stage_name(None)
    assert formatted == "Progress  "

    formatted = get_formatted_stage_name("")
    assert formatted == "Progress  "

    formatted = get_formatted_stage_name("stage")
    assert formatted == "Stage     "

    formatted = get_formatted_stage_name("very long stage name")
    assert formatted == "Very long stage name"


def test_check_original_input_paths(tmp_path, mixed_schema_csv_dir):
    plan = PipelineResumePlan(tmp_path=tmp_path, progress_bar=False, resume=False)

    input_file_list = [
        Path(mixed_schema_csv_dir) / "input_01.csv",
        Path(mixed_schema_csv_dir) / "input_02.csv",
    ]

    checked_files = plan.check_original_input_paths(input_file_list)

    round_trip_files = plan.check_original_input_paths(checked_files)

    npt.assert_array_equal(checked_files, round_trip_files)


def test_read_markers(test_data_dir):
    plan = PipelineResumePlan(tmp_path=test_data_dir / "markers", progress_bar=False)
    markers = plan.read_markers("mapping")
    assert markers == {"map_001": ["45"], "map_002": ["zippy"]}
