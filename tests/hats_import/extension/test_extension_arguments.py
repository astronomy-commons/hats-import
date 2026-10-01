"""Tests of argument validation for splitting a catalog into core and extension"""

import pytest

from hats_import.extension.arguments import ExtensionArguments


def make_args(catalog_path, tmp_path, **kwargs):
    """Arguments for splitting the given catalog, with any overrides."""
    arguments = {
        "input_catalog_path": catalog_path,
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
    assert args.core.input_columns == ["_healpix_29", "id", "ra", "dec"]
    assert args.core.output_columns == args.core.input_columns
    # The extension contains the error columns - the healpix, join_column and ra/dec are copied over.
    assert args.extension.input_columns == ["_healpix_29", "id", "ra", "dec", "ra_error", "dec_error"]
    assert args.extension.output_columns == args.extension.input_columns
    # The extension is a collection of its own, inside the one this run writes.
    assert args.core.collection_path == args.catalog_path
    assert args.extension.collection_path == args.catalog_path / "small_sky_order1_errors"
    # A renamed join column is written under its new name.
    renamed = make_args(small_sky_order1_catalog, tmp_path / "renamed", join_column="object_id")
    assert renamed.extension.input_columns[1] == "id"
    assert renamed.extension.output_columns[1] == "object_id"
    # The output is one collection, holding the core and the extension named after it.
    assert args.catalog_path == tmp_path / "small_sky_collection"
    assert args.core.name == "small_sky_order1"
    assert args.core.catalog_path == args.catalog_path / "small_sky_order1"
    assert args.extension.name == "small_sky_order1_errors"
    assert args.extension.catalog_path == (
        args.catalog_path / "small_sky_order1_errors" / "small_sky_order1_errors"
    )


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
    with pytest.raises(ValueError, match="extension columns do not exist"):
        make_args(small_sky_order1_catalog, tmp_path, extension_columns=[""])
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
    # Another column would be written under the join key's name.
    with pytest.raises(ValueError, match="conflicts"):
        make_args(small_sky_order1_catalog, tmp_path, join_column="ra_error")
    with pytest.raises(ValueError, match="conflicts"):
        make_args(small_sky_order1_catalog, tmp_path, join_column="ra")


def test_nested_columns(small_sky_nested_catalog, tmp_path):
    """A whole nested column can move to the extension, a single field of one cannot."""
    with pytest.raises(ValueError, match="fields of a nested column"):
        make_args(small_sky_nested_catalog, tmp_path, extension_columns=["lc.mjd", "lc.mag"])
    with pytest.raises(ValueError, match="primary_column 'lc.mjd' is a field of a nested column"):
        make_args(small_sky_nested_catalog, tmp_path, primary_column="lc.mjd")
    args = make_args(small_sky_nested_catalog, tmp_path, extension_columns=["lc"])
    assert args.extension.input_columns == ["_healpix_29", "id", "ra", "dec", "lc"]
    assert "lc" not in args.core.output_columns


def test_copied_columns_cannot_be_moved(small_sky_order1_catalog, tmp_path):
    """The error names the columns that are copied into the extension automatically."""
    with pytest.raises(ValueError, match="copied into the extension") as error:
        make_args(
            small_sky_order1_catalog,
            tmp_path,
            extension_columns=["_healpix_29", "id", "ra", "dec", "ra_error"],
        )
    assert "['_healpix_29', 'id', 'ra', 'dec']" in str(error.value)


def test_input_must_be_object_or_source(small_sky_o1_collection, tmp_path):
    """Only object and source catalogs can be split, and not e.g. a margin."""
    with pytest.raises(ValueError, match="OBJECT or SOURCE"):
        make_args(small_sky_o1_collection / "small_sky_order1_margin", tmp_path)


def test_derived_name(small_sky_order1_catalog, tmp_path):
    """Margins and indexes named after the input catalog are renamed after the side they go to."""
    args = make_args(small_sky_order1_catalog, tmp_path)
    assert args.core.derived_name("small_sky_order1_margin") == "small_sky_order1_margin"
    assert args.extension.derived_name("small_sky_order1_margin") == "small_sky_order1_errors_margin"
    # Any other name is kept as is.
    assert args.extension.derived_name("custom_margin") == "custom_margin"
