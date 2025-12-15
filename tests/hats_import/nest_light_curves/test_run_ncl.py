import pytest
from hats import read_hats

import hats_import.nest_light_curves.run_import as runner


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

    ## Check that the association data can be parsed as a valid association catalog.
    catalog = read_hats(small_sky_ncl_args.catalog_path)
    assert catalog.on_disk
    assert catalog.catalog_path == small_sky_ncl_args.catalog_path
    assert len(catalog.get_healpix_pixels()) == 4
    assert catalog.catalog_info.total_rows == 131

    print("schema")
    print(catalog.schema)
