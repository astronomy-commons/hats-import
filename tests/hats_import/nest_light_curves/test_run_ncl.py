import pytest
from hats import read_hats

import hats_import.nest_light_curves.run_import as runner
from hats_import.nest_light_curves.arguments import NestLightCurveArguments


def test_empty_args():
    """Runner should fail with empty arguments"""
    with pytest.raises(TypeError):
        runner.run(None, None)


def test_bad_args():
    """Runner should fail with mis-typed arguments"""
    args = {"output_catalog_name": "bad_arg_type"}
    with pytest.raises(TypeError):
        runner.run(args, None)


@pytest.mark.dask
def test_object_to_source(dask_client, small_sky_ncl_args):
    """Test creating association between object and source catalogs."""
    runner.run(small_sky_ncl_args, dask_client)

    catalog = read_hats(small_sky_ncl_args.catalog_path)
    assert catalog.on_disk
    assert catalog.catalog_path == small_sky_ncl_args.catalog_path
    assert len(catalog.get_healpix_pixels()) == 4
    assert catalog.catalog_info.total_rows == 131


@pytest.mark.dask
def test_object_to_source_partition_strategy_source(
    dask_client, small_sky_object_catalog, small_sky_source_catalog, tmp_path
):
    """Test creating association between object and source catalogs."""
    small_sky_ncl_args = NestLightCurveArguments(
        object_catalog_dir=small_sky_object_catalog,
        object_id_column="id",
        source_catalog_dir=small_sky_source_catalog,
        source_object_id_column="object_id",
        source_nested_columns=["mjd", "mag", "band"],
        output_artifact_name="small_sky_light_curve",
        partition_strategy="source_count",
        partition_threshold=5000,
        highest_healpix_order=5,
        output_path=tmp_path,
        progress_bar=False,
    )

    runner.run(small_sky_ncl_args, dask_client)

    catalog = read_hats(small_sky_ncl_args.catalog_path)
    assert catalog.on_disk
    assert catalog.catalog_path == small_sky_ncl_args.catalog_path
    assert len(catalog.get_healpix_pixels()) == 10
    assert catalog.catalog_info.total_rows == 131


@pytest.mark.dask
def test_object_to_source_partition_strategy_mem(
    dask_client, small_sky_object_catalog, small_sky_source_catalog, tmp_path
):
    """Test creating association between object and source catalogs."""
    small_sky_ncl_args = NestLightCurveArguments(
        object_catalog_dir=small_sky_object_catalog,
        object_id_column="id",
        source_catalog_dir=small_sky_source_catalog,
        source_object_id_column="object_id",
        source_nested_columns=["mjd", "mag", "band"],
        output_artifact_name="small_sky_light_curve",
        partition_strategy="mem_size",
        partition_threshold=300_000,
        highest_healpix_order=5,
        output_path=tmp_path,
        progress_bar=False,
    )

    runner.run(small_sky_ncl_args, dask_client)

    catalog = read_hats(small_sky_ncl_args.catalog_path)
    assert catalog.on_disk
    assert catalog.catalog_path == small_sky_ncl_args.catalog_path
    assert len(catalog.get_healpix_pixels()) == 16
    assert catalog.catalog_info.total_rows == 131
