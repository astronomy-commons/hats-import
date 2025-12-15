import os
from pathlib import Path

import pytest
from hats import read_hats

from hats_import.nest_light_curves.resume_plan import NestLightCurvePlan, object_neighbor_map


@pytest.mark.timeout(10)
def test_object_to_source_map(small_sky_object_catalog, small_sky_source_catalog, small_sky_ncl_map):
    """Test creating plan map for object and source catalogs."""
    object_catalog = read_hats(small_sky_object_catalog)
    source_catalog = read_hats(small_sky_source_catalog)

    source_to_object = object_neighbor_map(object_catalog, source_catalog)
    assert source_to_object == small_sky_ncl_map


def test_counting_done(small_sky_ncl_args):
    """Verify expected behavior of counting done file"""
    plan = NestLightCurvePlan(small_sky_ncl_args)
    assert not plan.is_counting_done()
    plan.touch_stage_done_file(NestLightCurvePlan.COUNTING_STAGE)
    assert plan.is_counting_done()

    plan.clean_resume_files()
    assert not plan.is_counting_done()


@pytest.mark.timeout(10)
def test_count_keys(small_sky_ncl_args):
    """Verify expected behavior of counting keys file"""
    plan = NestLightCurvePlan(small_sky_ncl_args)
    assert len(plan.count_keys) == 1

    ## Mark one done and check that there's one less key to count later.
    Path(small_sky_ncl_args.tmp_path, "0_11.csv").touch()

    plan.gather_plan(small_sky_ncl_args)
    assert len(plan.count_keys) == 0


@pytest.mark.timeout(10)
def test_cached_map_file(small_sky_ncl_args):
    """Verify that we cache the mapping file for later use.
    This can be expensive to compute for large survey cross-products!"""
    plan = NestLightCurvePlan(small_sky_ncl_args)
    assert len(plan.count_keys) == 1

    ## The source partition mapping should be cached in a file.
    cache_map_file = os.path.join(small_sky_ncl_args.tmp_path, NestLightCurvePlan.SOURCE_MAP_FILE)
    assert os.path.exists(cache_map_file)

    plan = NestLightCurvePlan(small_sky_ncl_args)
    assert len(plan.count_keys) == 1


def test_get_sources_to_count(small_sky_ncl_args):
    """Test generation of remaining count items"""
    plan = NestLightCurvePlan(small_sky_ncl_args)

    remaining_count_items = plan.get_sources_to_count()
    assert len(remaining_count_items) == 1

    ## Use previous value of sources map, and find intermediate file, so there are no
    ## remaining sources to count.
    Path(small_sky_ncl_args.tmp_path, "0_11.csv").touch()
    remaining_count_items = plan.get_sources_to_count()
    assert len(remaining_count_items) == 0

    ## Kind of silly, but clear out the pixel map, since it's populated on init.
    ## Fail to find the remaining sources to count because we don't know the map.
    plan.object_map = None
    with pytest.raises(ValueError, match="object_map"):
        remaining_count_items = plan.get_sources_to_count()


def never_fails():
    """Method never fails, but never marks intermediate success file."""
    return


@pytest.mark.dask
def test_some_counting_task_failures(small_sky_ncl_args, dask_client):
    """Test that we only consider counting stage successful if all done files are written"""
    plan = NestLightCurvePlan(small_sky_ncl_args)

    ## Method doesn't FAIL, but it doesn't write out the intermediate results file either.
    futures = [dask_client.submit(never_fails)]
    with pytest.raises(RuntimeError, match="1 counting stages"):
        plan.wait_for_counting(futures)

    ## Write one intermediate results file. There are fewer unsuccessful stages.
    Path(small_sky_ncl_args.tmp_path, "0_11.csv").touch()
    futures = [dask_client.submit(never_fails)]
    plan.wait_for_counting(futures)
