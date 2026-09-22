"""Tests of argument validation for splitting a catalog into core and extension"""

import pytest

from hats_import.extension.arguments import ExtensionArguments


def make_args(small_sky_order1_catalog, tmp_path, **kwargs):
    """Arguments for splitting the small sky catalog, with any overrides."""
    arguments = {
        "input_catalog_path": small_sky_order1_catalog,
        "extension_columns": ["ra_error", "dec_error"],
        "primary_column": "id",
        "output_path": tmp_path,
        "extension_name": "errors",
        "output_artifact_name": "small_sky_collection",
        "progress_bar": False,
    }
    return ExtensionArguments(**(arguments | kwargs))


def test_good_args(small_sky_order1_catalog, tmp_path):
    """Valid arguments give the expected paths and column lists."""
    args = make_args(small_sky_order1_catalog, tmp_path)
    assert args.join_column == "id"
    assert args.core_columns == ["_healpix_29", "id", "ra", "dec"]
    assert args.extension_input_columns == ["_healpix_29", "id", "ra", "dec", "ra_error", "dec_error"]
    # The extension contains the error columns - the healpix, join_column and ra/dec are copied over.
    assert args.extension_output_columns == args.extension_input_columns
    ## The output is one collection, holding the core and the extension named after it.
    assert args.catalog_path == tmp_path / "small_sky_collection"
    assert args.core.name == "small_sky_order1"
    assert args.core.catalog_path == args.catalog_path / "small_sky_order1"
    assert args.extension.name == "small_sky_order1_errors"
    assert args.extension.catalog_path == args.catalog_path / "small_sky_order1_errors"
    assert args.core.catalog_path.exists()
    assert args.extension.catalog_path.exists()


def test_missing_args(small_sky_order1_catalog, tmp_path):
    """Each required argument is checked."""
    with pytest.raises(ValueError, match="input_catalog_path"):
        make_args(small_sky_order1_catalog, tmp_path, input_catalog_path=None)
    with pytest.raises(ValueError, match="extension_name is required"):
        make_args(small_sky_order1_catalog, tmp_path, extension_name="")
    with pytest.raises(ValueError, match="output_artifact_name is required"):
        make_args(small_sky_order1_catalog, tmp_path, output_artifact_name="")
    with pytest.raises(ValueError, match="extension_columns is required"):
        make_args(small_sky_order1_catalog, tmp_path, extension_columns=[])
    with pytest.raises(ValueError, match="primary_column is required"):
        make_args(small_sky_order1_catalog, tmp_path, primary_column="")


def test_invalid_input_catalog(tmp_path):
    """The input must be a valid catalog or collection."""
    with pytest.raises(ValueError, match="not a valid catalog"):
        make_args(tmp_path, tmp_path)


def test_bad_columns(small_sky_order1_catalog, tmp_path):
    """Columns must exist, and the columns that are copied cannot be moved."""
    with pytest.raises(ValueError, match="do not exist"):
        make_args(small_sky_order1_catalog, tmp_path, extension_columns=["ra_error", "flux"])
    with pytest.raises(ValueError, match="does not exist"):
        make_args(small_sky_order1_catalog, tmp_path, primary_column="object_id")
    ## Another column would be written under the join key's name.
    with pytest.raises(ValueError, match="conflicts"):
        make_args(small_sky_order1_catalog, tmp_path, join_column="ra_error")
    with pytest.raises(ValueError, match="conflicts"):
        make_args(small_sky_order1_catalog, tmp_path, join_column="ra")


def test_copied_columns_cannot_be_moved(small_sky_order1_catalog, tmp_path):
    """The error names the columns that are copied into the extension automatically."""
    with pytest.raises(ValueError, match="copied into the extension") as error:
        make_args(
            small_sky_order1_catalog,
            tmp_path,
            extension_columns=["_healpix_29", "id", "ra", "dec", "ra_error"],
        )
    assert "['_healpix_29', 'id', 'ra', 'dec']" in str(error.value)
