"""Test catalog resume logic"""

from unittest.mock import MagicMock

import numpy as np
import numpy.testing as npt
import pytest
from hats.pixel_math.healpix_pixel import HealpixPixel
from hats.pixel_math.sparse_histogram import SparseHistogram

from hats_import.catalog.resume_plan import ResumePlan


def test_done_checks(tmp_path):
    """Verify that done files imply correct pipeline execution order:
    mapping > splitting > reducing
    """
    plan = ResumePlan(tmp_path=tmp_path, progress_bar=False, resume=True)
    plan.touch_stage_done_file(ResumePlan.REDUCING_STAGE)

    with pytest.raises(ValueError, match="before reducing"):
        plan.gather_plan()

    plan.touch_stage_done_file(ResumePlan.SPLITTING_STAGE)
    with pytest.raises(ValueError, match="before reducing"):
        plan.gather_plan()

    plan.clean_resume_files()

    plan = ResumePlan(tmp_path=tmp_path, progress_bar=False, resume=True)
    plan.touch_stage_done_file(ResumePlan.SPLITTING_STAGE)
    with pytest.raises(ValueError, match="before splitting"):
        plan.gather_plan()


def test_same_input_paths(tmp_path, small_sky_single_file, formats_headers_csv):
    """Test that we can only resume if the input_paths are the same."""
    plan = ResumePlan(
        tmp_path=tmp_path,
        progress_bar=False,
        resume=True,
        input_paths=[small_sky_single_file, formats_headers_csv],
    )
    map_files = plan.map_files
    assert len(map_files) == 2

    with pytest.raises(ValueError, match="Different file set"):
        ResumePlan(
            tmp_path=tmp_path,
            progress_bar=False,
            resume=True,
            input_paths=[small_sky_single_file],
        )

    ## List is the same length, but includes a duplicate
    with pytest.raises(ValueError, match="Different file set"):
        ResumePlan(
            tmp_path=tmp_path,
            progress_bar=False,
            resume=True,
            input_paths=[small_sky_single_file, small_sky_single_file],
        )

    ## Includes a duplicate file, and we don't like that.
    with pytest.raises(ValueError, match="Different file set"):
        ResumePlan(
            tmp_path=tmp_path,
            progress_bar=False,
            resume=True,
            input_paths=[small_sky_single_file, small_sky_single_file, formats_headers_csv],
        )


def test_remaining_map_keys(tmp_path):
    """Test that we can read what we write into a histogram file."""
    num_inputs = 1_000
    input_paths = [f"foo_{i}" for i in range(0, num_inputs)]
    plan = ResumePlan(tmp_path=tmp_path, progress_bar=False, input_paths=input_paths)

    remaining_keys = plan.get_remaining_map_keys()
    assert len(remaining_keys) == num_inputs

    histogram = SparseHistogram([11], [131], 0)
    for i in range(0, num_inputs):
        histogram.to_file(ResumePlan.partial_histogram_file(tmp_path=tmp_path, mapping_key=f"map_{i}"))

    remaining_keys = plan.get_remaining_map_keys()
    assert len(remaining_keys) == 0


def test_read_write_histogram(tmp_path):
    """Test that we can read what we write into a histogram file."""
    plan = ResumePlan(tmp_path=tmp_path, progress_bar=False, input_paths=["foo1"])

    ## We're not ready to read the final histogram - missing partial histograms.
    with pytest.raises(RuntimeError, match="map stages"):
        result = plan.read_histogram(0)

    remaining_keys = plan.get_remaining_map_keys()
    assert remaining_keys == [("map_0", "foo1")]

    histogram = SparseHistogram([11], [131], 0)
    histogram.to_file(ResumePlan.partial_histogram_file(tmp_path=tmp_path, mapping_key="map_0"))

    remaining_keys = plan.get_remaining_map_keys()
    assert len(remaining_keys) == 0
    result = plan.read_histogram(0)
    npt.assert_array_equal(result, histogram.to_array())


def test_get_alignment_file(tmp_path):
    plan = ResumePlan(tmp_path=tmp_path, progress_bar=False, input_paths=["foo1"])
    raw_histogram = np.full(12, 0)
    raw_histogram[11] = 131
    alignment_file = plan.get_alignment_file(raw_histogram, -1, 0, 0, 1_000, True, 131)

    alignment_file2 = plan.get_alignment_file(raw_histogram, -1, 0, 0, 1_000, True, 131)

    assert alignment_file == alignment_file2

    with pytest.raises(ValueError, match="does not match expectation"):
        plan.get_alignment_file(raw_histogram, -1, 0, 0, 1_000, True, 130)


def never_fails():
    """Method never fails, but never marks intermediate success file."""
    return


@pytest.mark.dask
def test_some_map_task_failures(tmp_path, dask_client):
    """Test that we only consider map stage successful if all partial files are written"""
    plan = ResumePlan(tmp_path=tmp_path, progress_bar=False, input_paths=["foo1"])

    ## Method doesn't FAIL, but it doesn't write out the partial histogram either.
    ## Since the intermediate files aren't found, we throw an error.
    futures = [dask_client.submit(never_fails)]
    with pytest.raises(RuntimeError, match="map stages"):
        plan.wait_for_mapping(futures)

    histogram = SparseHistogram([11], [131], 0)
    histogram.to_file(ResumePlan.partial_histogram_file(tmp_path=tmp_path, mapping_key="map_0"))

    ## Method succeeds, *and* partial histogram is present.
    futures = [dask_client.submit(never_fails)]
    plan.wait_for_mapping(futures)


def test_read_write_splitting_keys(tmp_path, small_sky_single_file, formats_headers_csv):
    """Test that we can read what we write into a reducing log file."""
    input_paths = [small_sky_single_file, formats_headers_csv]
    plan = ResumePlan(tmp_path=tmp_path, progress_bar=False, resume=True, input_paths=input_paths)
    split_keys = plan.split_keys
    assert len(split_keys) == 2

    ResumePlan.touch_key_done_file(tmp_path, ResumePlan.SPLITTING_STAGE, "split_0")

    plan.gather_plan()
    split_keys = plan.split_keys
    assert len(split_keys) == 1
    assert split_keys[0][0] == "split_1"

    ResumePlan.touch_key_done_file(tmp_path, ResumePlan.SPLITTING_STAGE, "split_1")
    plan.gather_plan()
    split_keys = plan.split_keys
    assert len(split_keys) == 0

    plan.clean_resume_files()
    plan.gather_plan()
    split_keys = plan.split_keys
    assert len(split_keys) == 2


@pytest.mark.dask
def test_some_split_task_failures(tmp_path, dask_client):
    """Test that we only consider split stage successful if all done files are written"""
    plan = ResumePlan(tmp_path=tmp_path, progress_bar=False, input_paths=["foo1"])

    ## Method doesn't FAIL, but it doesn't write out the done file either.
    ## Since the intermediate files aren't found, we throw an error.
    futures = [dask_client.submit(never_fails)]
    with pytest.raises(RuntimeError, match="split stages"):
        plan.wait_for_splitting(futures)

    ResumePlan.touch_key_done_file(tmp_path, ResumePlan.SPLITTING_STAGE, "split_0")

    ## Method succeeds, and done file is present.
    futures = [dask_client.submit(never_fails)]
    plan.wait_for_splitting(futures)


def test_get_reduce_items(tmp_path):
    """Test generation of remaining reduce items"""
    plan = ResumePlan(tmp_path=tmp_path, progress_bar=False)

    with pytest.raises(RuntimeError, match="destination pixel map"):
        remaining_reduce_items = plan.get_reduce_items()

    with pytest.raises(RuntimeError, match="destination pixel map"):
        remaining_reduce_items = plan.get_destination_pixels()

    plan.destination_pixel_map = {HealpixPixel(0, 11): 131}
    remaining_reduce_items = plan.get_reduce_items()
    assert len(remaining_reduce_items) == 1

    all_pixels = plan.get_destination_pixels()
    assert len(all_pixels) == 1

    ResumePlan.reducing_key_done(tmp_path=tmp_path, reducing_key="0_11")
    remaining_reduce_items = plan.get_reduce_items()
    assert len(remaining_reduce_items) == 0


@pytest.mark.dask
def test_some_reduce_task_failures(tmp_path, dask_client):
    """Test that we only consider reduce stage successful if all done files are written"""
    plan = ResumePlan(tmp_path=tmp_path, progress_bar=False)

    plan.destination_pixel_map = {HealpixPixel(0, 11): 131}
    remaining_reduce_items = plan.get_reduce_items()
    assert len(remaining_reduce_items) == 1

    ## Method doesn't FAIL, but it doesn't write out the done file either.
    ## Since the intermediate files aren't found, we throw an error.
    futures = [dask_client.submit(never_fails)]
    with pytest.raises(RuntimeError, match="reduce stages"):
        plan.wait_for_reducing(futures)

    ResumePlan.touch_key_done_file(tmp_path, ResumePlan.REDUCING_STAGE, "0_11")

    ## Method succeeds, and done file is present.
    futures = [dask_client.submit(never_fails)]
    plan.wait_for_reducing(futures)


def test_run_stages(tmp_path):
    """Verify interaction with user-supplied run stages and done files"""
    shared_args = {
        "tmp_path": tmp_path / "stages",
        "progress_bar": False,
        "resume": True,
    }
    plan = ResumePlan(**shared_args)
    assert plan.should_run_mapping
    assert plan.should_run_splitting
    assert plan.should_run_reducing
    assert plan.should_run_finishing

    with pytest.raises(ValueError, match="before splitting"):
        plan.gather_plan(run_stages=["splitting"])

    plan.touch_stage_done_file(ResumePlan.MAPPING_STAGE)
    with pytest.raises(ValueError, match="before reducing"):
        plan.gather_plan(run_stages=["reducing"])

    plan.touch_stage_done_file(ResumePlan.SPLITTING_STAGE)
    plan.gather_plan()

    assert not plan.should_run_mapping
    assert not plan.should_run_splitting
    assert plan.should_run_reducing
    assert plan.should_run_finishing


def test_resume_plan_with_byte_pixel_threshold(tmp_path):
    """Test ResumePlan initialization with byte_pixel_threshold specified."""
    # Mock import_args with necessary attributes
    import_args = MagicMock()
    import_args.resume_kwargs_dict.return_value = {"tmp_path": tmp_path}
    import_args.debug_stats_only = False
    import_args.byte_pixel_threshold = 100  # Simulate a specified byte_pixel_threshold
    import_args.input_paths = ["file1", "file2"]

    # Initialize ResumePlan with the mocked import_args
    resume_plan = ResumePlan(import_args=import_args)

    # Assert that threshold_mode is set to "mem_size"
    assert resume_plan.threshold_mode == "mem_size"


def test_gather_plan_mem_size_mode(tmp_path):
    """Test gather_plan in mem_size mode."""
    # Mock import_args with necessary attributes
    import_args = MagicMock()
    import_args.resume_kwargs_dict.return_value = {"tmp_path": tmp_path}
    import_args.debug_stats_only = False
    import_args.byte_pixel_threshold = 100  # Simulate a specified byte_pixel_threshold
    import_args.input_paths = ["file1", "file2"]

    # Initialize ResumePlan with the mocked import_args
    plan = ResumePlan(import_args=import_args, tmp_path=tmp_path)

    # Mock methods to avoid actual file operations
    plan.get_remaining_map_keys = MagicMock(return_value=[("map_1", "file1"), ("map_2", "file2")])
    plan.done_file_exists = MagicMock(return_value=False)
    plan.check_original_input_paths = MagicMock(return_value=import_args.input_paths)
    plan.print_progress = MagicMock()

    # Call gather_plan and verify mem_size directory creation
    plan.gather_plan()
    plan.get_remaining_map_keys.assert_any_call(which_histogram="mem_size")


def test_get_remaining_map_keys_mem_size_and_invalid(tmp_path):
    """Test get_remaining_map_keys with mem_size and invalid options."""
    # Create a ResumePlan instance with mem_size threshold_mode
    plan = ResumePlan(tmp_path=tmp_path, progress_bar=False, input_paths=["file1", "file2"])
    plan.threshold_mode = "mem_size"

    # Mock the directory structure to simulate mem_size histogram files
    mem_size_dir = tmp_path / plan.MEM_SIZE_HISTOGRAMS_DIR
    mem_size_dir.mkdir()
    (mem_size_dir / "map_0.npz").touch()

    # Call get_remaining_map_keys with mem_size
    remaining_keys = plan.get_remaining_map_keys(which_histogram="mem_size")
    assert len(remaining_keys) == 1
    assert remaining_keys[0] == ("map_1", "file2")

    # Call get_remaining_map_keys with an invalid option
    with pytest.raises(ValueError, match="Unrecognized which_histogram value"):
        plan.get_remaining_map_keys(which_histogram="invalid_option")

    # Raise error if threshold_mode is not mem_size
    plan.threshold_mode = "row_count"
    with pytest.raises(ValueError, match="when threshold_mode is not 'mem_size'"):
        plan.get_remaining_map_keys(which_histogram="mem_size")


def test_read_histogram_wrong_modes(tmp_path):
    """Test read_histogram with mem_size and invalid options."""
    plan = ResumePlan(tmp_path=tmp_path, progress_bar=False, input_paths=["file1", "file2"])
    assert plan.threshold_mode == "row_count"

    # Raise error for invalid which_histogram option when reading histogram
    with pytest.raises(ValueError, match="Cannot read mem_size histogram"):
        plan.read_histogram(2, which_histogram="mem_size")

    # Raise error for unrecognized which_histogram option
    with pytest.raises(ValueError, match="Unrecognized which_histogram value"):
        plan.read_histogram(2, which_histogram="invalid_option")
