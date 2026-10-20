"""Tests of resuming an interrupted extension split"""

import pytest
from hats import read_hats
from hats.pixel_math import HealpixPixel

import hats_import.extension.run_split_import as runner
from hats_import.extension.arguments import ExtensionArguments
from hats_import.extension.resume_plan import ExtensionSplitPlan


def split_args(catalog_path, tmp_path, **kwargs):
    """Arguments for splitting the given catalog, with a resume directory of its own."""
    arguments = {
        "input_catalog_path": catalog_path,
        "extension_columns": ["ra_error", "dec_error"],
        "primary_column": "id",
        "extension_name": "small_sky_order1_errors",
        "output_path": tmp_path / "output",
        "output_artifact_name": "small_sky_with_extension",
        "tmp_dir": tmp_path / "tmp",
        "progress_bar": False,
    }
    return ExtensionArguments(**(arguments | kwargs))


def test_remaining_pixels(small_sky_order1_catalog, tmp_path):
    """Partitions that are already split are not split again."""
    pixels = [HealpixPixel(1, pixel) for pixel in range(44, 48)]
    args = split_args(small_sky_order1_catalog, tmp_path)
    plan = ExtensionSplitPlan(args)
    assert plan.remaining_pixels(args.input_catalog) == pixels

    runner.split_pixel(HealpixPixel(1, 44), args, args.input_catalog, plan.tmp_path)
    runner.split_pixel(HealpixPixel(1, 46), args, args.input_catalog, plan.tmp_path)

    # A new plan over the same resume directory sees what is left to do.
    resumed = ExtensionSplitPlan(split_args(small_sky_order1_catalog, tmp_path))
    assert resumed.remaining_pixels(args.input_catalog) == [HealpixPixel(1, 45), HealpixPixel(1, 47)]
    assert not resumed.is_splitting_done()


@pytest.mark.dask
def test_resume_interrupted_split(small_sky_order1_catalog, tmp_path, dask_client):
    """A run that resumes an interrupted one writes the partitions that are left."""
    interrupted = split_args(small_sky_order1_catalog, tmp_path)
    plan = ExtensionSplitPlan(interrupted)
    for pixel in [HealpixPixel(1, 44), HealpixPixel(1, 45)]:
        runner.split_pixel(pixel, interrupted, interrupted.input_catalog, plan.tmp_path)

    args = split_args(small_sky_order1_catalog, tmp_path)
    runner.run(args, dask_client)

    core = read_hats(args.core.catalog_path)
    extension = read_hats(args.extension.catalog_path)
    assert core.catalog_info.total_rows == extension.catalog_info.total_rows == 131
    assert core.get_healpix_pixels() == [HealpixPixel(1, pixel) for pixel in range(44, 48)]
    assert not args.tmp_path.exists()


@pytest.mark.dask
def test_resume_false_starts_over(small_sky_order1_catalog, tmp_path, dask_client):
    """With `resume=False`, the resume files of an interrupted run are discarded."""
    interrupted = split_args(small_sky_order1_catalog, tmp_path)
    plan = ExtensionSplitPlan(interrupted)
    runner.split_pixel(HealpixPixel(1, 44), interrupted, interrupted.input_catalog, plan.tmp_path)
    assert len(list((plan.tmp_path / "splitting").glob("*_done"))) == 1

    args = split_args(small_sky_order1_catalog, tmp_path, resume=False)
    plan = ExtensionSplitPlan(args)
    assert plan.remaining_pixels(args.input_catalog) == [HealpixPixel(1, pixel) for pixel in range(44, 48)]

    runner.run(args, dask_client)
    assert read_hats(args.core.catalog_path).catalog_info.total_rows == 131
