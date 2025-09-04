"""Tests of argument validation"""

import pytest

from hats_import.catalog.arguments import ImportArguments, check_healpix_order_range

# pylint: disable=protected-access


def test_none():
    """No arguments provided. Should error for required args."""
    with pytest.raises(ValueError):
        ImportArguments()


def test_empty_required(small_sky_parts_dir, tmp_path):
    """*Most* required arguments are provided."""
    ## File reader is missing
    with pytest.raises(ValueError, match="file_reader"):
        ImportArguments(
            output_artifact_name="catalog",
            file_reader=None,
            input_path=small_sky_parts_dir,
            output_path=tmp_path,
        )
    ## Input path is missing
    with pytest.raises(ValueError, match="input_file"):
        ImportArguments(
            output_artifact_name="catalog",
            file_reader="csv",
            input_path="",
            output_path=tmp_path,
        )

    with pytest.raises(ValueError, match="input_file_list is required"):
        ImportArguments(
            output_artifact_name="catalog",
            file_reader="csv",
            input_file_list=None,
            output_path=tmp_path,
        )

    with pytest.raises(ValueError, match="input_file_list is empty"):
        ImportArguments(
            output_artifact_name="catalog",
            file_reader="csv",
            input_file_list=[],
            output_path=tmp_path,
        )


def test_invalid_paths(blank_data_dir, tmp_path):
    """Required arguments are provided, but paths aren't found."""
    ## Prove that it works with required args
    ImportArguments(
        output_artifact_name="catalog",
        input_path=blank_data_dir,
        file_reader="csv",
        output_path=tmp_path,
        progress_bar=False,
    )

    ## Bad input path
    with pytest.raises(FileNotFoundError):
        ImportArguments(
            output_artifact_name="catalog",
            input_path="path",
            file_reader="csv",
            output_path=tmp_path,
        )

    # Too many paths!
    with pytest.raises(ValueError, match="exactly one of"):
        ImportArguments(
            output_artifact_name="catalog",
            input_path=blank_data_dir,
            input_file_list=[blank_data_dir],
            file_reader="csv",
            output_path=tmp_path,
            progress_bar=False,
        )


def test_missing_paths(tmp_path):
    ## Input path has no files
    with pytest.raises(FileNotFoundError):
        ImportArguments(
            output_artifact_name="catalog",
            file_reader="csv",
            input_path=tmp_path,
            output_path=tmp_path,
        )


def test_good_paths(blank_data_dir, blank_data_file, tmp_path):
    """Required arguments are provided, and paths are found."""
    tmp_path_str = str(tmp_path)
    args = ImportArguments(
        output_artifact_name="catalog",
        input_path=blank_data_dir,
        file_reader="csv",
        output_path=tmp_path_str,
        tmp_dir=tmp_path_str,
        progress_bar=False,
    )
    assert args.input_path == blank_data_dir
    assert len(args.input_paths) == 1
    assert str(blank_data_file) in str(args.input_paths[0])


def test_multiple_files_in_path(small_sky_parts_dir, tmp_path):
    """Required arguments are provided, and paths are found."""
    args = ImportArguments(
        output_artifact_name="catalog",
        input_path=small_sky_parts_dir,
        file_reader="csv",
        output_path=tmp_path,
        progress_bar=False,
    )
    assert args.input_path == small_sky_parts_dir
    assert len(args.input_paths) == 6


def test_single_debug_file(formats_headers_csv, tmp_path):
    """Required arguments are provided, and paths are found."""
    args = ImportArguments(
        output_artifact_name="catalog",
        input_file_list=[formats_headers_csv],
        file_reader="csv",
        output_path=tmp_path,
        progress_bar=False,
    )
    assert len(args.input_paths) == 1
    assert args.input_paths[0] == formats_headers_csv

    args = ImportArguments(
        output_artifact_name="catalog",
        input_file_list=formats_headers_csv,
        file_reader="csv",
        output_path=tmp_path,
        progress_bar=False,
    )
    assert len(args.input_paths) == 1
    assert args.input_paths[0] == formats_headers_csv

    single_file_name = formats_headers_csv.name
    assert isinstance(single_file_name, str)
    args = ImportArguments(
        output_artifact_name="catalog",
        input_file_list=single_file_name,
        file_reader="csv",
        output_path=tmp_path,
        progress_bar=False,
    )
    assert len(args.input_paths) == 1
    assert args.input_paths[0] == single_file_name


def test_healpix_args(blank_data_dir, tmp_path):
    """Test errors for healpix partitioning arguments"""
    with pytest.raises(ValueError, match="highest_healpix_order"):
        ImportArguments(
            output_artifact_name="catalog",
            input_path=blank_data_dir,
            file_reader="csv",
            output_path=tmp_path,
            highest_healpix_order=30,
        )
    with pytest.raises(ValueError, match="pixel_threshold"):
        ImportArguments(
            output_artifact_name="catalog",
            input_path=blank_data_dir,
            file_reader="csv",
            output_path=tmp_path,
            pixel_threshold=3,
        )
    with pytest.raises(ValueError, match="constant_healpix_order"):
        ImportArguments(
            output_artifact_name="catalog",
            input_path=blank_data_dir,
            file_reader="csv",
            output_path=tmp_path,
            constant_healpix_order=30,
        )


def test_catalog_type(blank_data_dir, tmp_path):
    """Test errors for catalog types."""
    with pytest.raises(ValueError, match="catalog_type"):
        ImportArguments(
            output_artifact_name="catalog",
            catalog_type=None,
            input_path=blank_data_dir,
            file_reader="csv",
            output_path=tmp_path,
        )

    with pytest.raises(ValueError, match="catalog_type"):
        ImportArguments(
            output_artifact_name="catalog",
            catalog_type="association",
            input_path=blank_data_dir,
            file_reader="csv",
            output_path=tmp_path,
        )


def test_use_healpix_29(blank_data_dir, tmp_path):
    with pytest.raises(ValueError, match="no sort columns should be added"):
        ImportArguments(
            output_artifact_name="catalog",
            input_path=blank_data_dir,
            file_reader="csv",
            output_path=tmp_path,
            use_healpix_29=True,
            sort_columns="foo",
        )
    ImportArguments(
        output_artifact_name="catalog",
        input_path=blank_data_dir,
        file_reader="csv",
        output_path=tmp_path,
        use_healpix_29=True,
        sort_columns="",  # empty string is ok
    )


def test_to_table_properties(blank_data_dir, tmp_path):
    """Verify creation of catalog parameters for catalog to be created."""
    args = ImportArguments(
        output_artifact_name="catalog",
        input_path=blank_data_dir,
        file_reader="csv",
        output_path=tmp_path,
        tmp_dir=tmp_path,
        progress_bar=False,
        addl_hats_properties={"hats_cols_default": "id, mjd", "obs_regime": "Optical"},
    )
    catalog_info = args.to_table_properties(
        total_rows=10, highest_order=4, moc_sky_fraction=22 / 7, column_names=["id", "mjd", "ra", "dec"]
    )
    assert catalog_info.catalog_name == "catalog"
    assert catalog_info.total_rows == 10
    assert catalog_info.default_columns == ["id", "mjd"]
    assert catalog_info.__pydantic_extra__["obs_regime"] == "Optical"


def test_to_table_properties_missing_default(blank_data_dir, tmp_path):
    """Verify creation of catalog parameters for catalog to be created."""
    args = ImportArguments(
        output_artifact_name="catalog",
        input_path=blank_data_dir,
        file_reader="csv",
        output_path=tmp_path,
        tmp_dir=tmp_path,
        progress_bar=False,
        addl_hats_properties={"hats_cols_default": "id, mjd", "obs_regime": "Optical"},
    )
    with pytest.raises(ValueError, match="Some default columns not found"):
        args.to_table_properties(
            total_rows=10, highest_order=4, moc_sky_fraction=22 / 7, column_names=["id", "ra", "dec"]
        )


def test_check_healpix_order_range():
    """Test method check_healpix_order_range"""
    check_healpix_order_range(5, "order_field")
    check_healpix_order_range(5, "order_field", lower_bound=0, upper_bound=19)

    with pytest.raises(ValueError, match="positive"):
        check_healpix_order_range(5, "order_field", lower_bound=-1)

    with pytest.raises(ValueError, match="29"):
        check_healpix_order_range(5, "order_field", upper_bound=30)

    with pytest.raises(ValueError, match="order_field"):
        check_healpix_order_range(-1, "order_field")
    with pytest.raises(ValueError, match="order_field"):
        check_healpix_order_range(30, "order_field")

    with pytest.raises(TypeError, match="not supported"):
        check_healpix_order_range("two", "order_field")
    with pytest.raises(TypeError, match="not supported"):
        check_healpix_order_range(5, "order_field", upper_bound="ten")


def test_no_import_overwrite(small_sky_object_catalog, parquet_shards_dir):
    """Runner should refuse to overwrite a valid catalog"""
    catalog_dir = small_sky_object_catalog.parent
    catalog_name = small_sky_object_catalog.name
    with pytest.raises(ValueError, match="already contains a valid catalog"):
        ImportArguments(
            input_path=parquet_shards_dir,
            output_path=catalog_dir,
            output_artifact_name=catalog_name,
            file_reader="parquet",
        )
