import hats
import pandas as pd
import pyarrow.parquet as pq
import pytest
from hats import read_hats
from hats.io import paths

import hats_import.catalog.run_import as runner
from hats_import.catalog import ImportArguments
from hats_import.catalog.file_readers import ParquetPyarrowReader


def test_reimport_arguments(tmp_path, small_sky_object_catalog):
    args = ImportArguments.reimport_from_hats(
        small_sky_object_catalog, tmp_path, addl_hats_properties={"obs_regime": "Optical"}
    )
    catalog = hats.read_hats(small_sky_object_catalog)
    file_paths = [
        hats.io.pixel_catalog_file(catalog.catalog_base_dir, p) for p in catalog.get_healpix_pixels()
    ]
    assert args.catalog_type == catalog.catalog_info.catalog_type
    assert args.ra_column == catalog.catalog_info.ra_column
    assert args.dec_column == catalog.catalog_info.dec_column
    assert args.input_paths == file_paths
    assert isinstance(args.file_reader, ParquetPyarrowReader)
    assert args.output_artifact_name == catalog.catalog_name
    assert args.expected_total_rows == catalog.catalog_info.total_rows
    assert args.addl_hats_properties == catalog.catalog_info.extra_dict(by_alias=True) | {
        "hats_cols_default": catalog.catalog_info.default_columns,
        "hats_npix_suffix": catalog.catalog_info.npix_suffix,
        "obs_regime": "Optical",
    }
    assert args.use_healpix_29
    assert not args.add_healpix_29


def test_reimport_arguments_constant(tmp_path, small_sky_object_catalog):
    args = ImportArguments.reimport_from_hats(small_sky_object_catalog, tmp_path, constant_healpix_order=6)
    catalog = hats.read_hats(small_sky_object_catalog)
    file_paths = [
        hats.io.pixel_catalog_file(catalog.catalog_base_dir, p) for p in catalog.get_healpix_pixels()
    ]
    assert args.catalog_type == catalog.catalog_info.catalog_type
    assert args.ra_column == catalog.catalog_info.ra_column
    assert args.dec_column == catalog.catalog_info.dec_column
    assert args.input_paths == file_paths
    assert isinstance(args.file_reader, ParquetPyarrowReader)
    assert args.output_artifact_name == catalog.catalog_name
    assert args.expected_total_rows == catalog.catalog_info.total_rows
    assert args.addl_hats_properties == catalog.catalog_info.extra_dict(by_alias=True) | {
        "hats_cols_default": catalog.catalog_info.default_columns,
        "hats_npix_suffix": catalog.catalog_info.npix_suffix,
    }
    assert args.use_healpix_29
    assert not args.add_healpix_29


def test_reimport_arguments_extra_kwargs(tmp_path, small_sky_object_catalog):
    output_name = "small_sky_higher_order"
    pixel_thresh = 100
    args = ImportArguments.reimport_from_hats(
        small_sky_object_catalog,
        tmp_path,
        pixel_threshold=pixel_thresh,
        output_artifact_name=output_name,
        highest_healpix_order=2,
    )
    catalog = hats.read_hats(small_sky_object_catalog)
    file_paths = [
        hats.io.pixel_catalog_file(catalog.catalog_base_dir, p) for p in catalog.get_healpix_pixels()
    ]
    assert args.catalog_type == catalog.catalog_info.catalog_type
    assert args.ra_column == catalog.catalog_info.ra_column
    assert args.dec_column == catalog.catalog_info.dec_column
    assert args.input_paths == file_paths
    assert isinstance(args.file_reader, ParquetPyarrowReader)
    assert args.output_artifact_name == output_name
    assert args.pixel_threshold == pixel_thresh
    assert args.expected_total_rows == catalog.catalog_info.total_rows
    assert args.addl_hats_properties == catalog.catalog_info.extra_dict(by_alias=True) | {
        "hats_cols_default": catalog.catalog_info.default_columns,
        "hats_npix_suffix": catalog.catalog_info.npix_suffix,
    }
    assert args.use_healpix_29
    assert not args.add_healpix_29


def test_reimport_arguments_empty_dir(tmp_path):
    wrong_input_path = tmp_path / "nonsense"
    with pytest.raises(FileNotFoundError):
        ImportArguments.reimport_from_hats(wrong_input_path, tmp_path)


def test_reimport_arguments_invalid_dir(wrong_files_and_rows_dir, tmp_path):
    with pytest.raises(ValueError):
        ImportArguments.reimport_from_hats(wrong_files_and_rows_dir, tmp_path)


def test_reimport_arguments_catalog_collection(test_data_dir, small_sky_object_catalog, tmp_path):
    wrong_input_path = test_data_dir / "small_sky_collection"
    args = ImportArguments.reimport_from_hats(wrong_input_path, tmp_path)

    catalog = hats.read_hats(small_sky_object_catalog)
    assert len(args.input_paths) == len(catalog.get_healpix_pixels())
    assert args.catalog_type == catalog.catalog_info.catalog_type
    assert args.ra_column == catalog.catalog_info.ra_column
    assert args.dec_column == catalog.catalog_info.dec_column


@pytest.mark.dask(timeout=10)
def test_run_reimport(
    dask_client,
    small_sky_object_catalog,
    tmp_path,
):
    output_name = "small_sky_higher_order"
    pixel_thresh = 100
    args = ImportArguments.reimport_from_hats(
        small_sky_object_catalog,
        tmp_path,
        pixel_threshold=pixel_thresh,
        output_artifact_name=output_name,
        highest_healpix_order=1,
        addl_hats_properties={"obs_regime": "Optical"},
    )

    runner.run(args, dask_client)

    old_cat = read_hats(small_sky_object_catalog)

    # Check that the catalog metadata file exists
    catalog = read_hats(args.catalog_path)
    assert catalog.on_disk
    assert catalog.catalog_path == args.catalog_path
    assert catalog.catalog_info.ra_column == old_cat.catalog_info.ra_column
    assert catalog.catalog_info.dec_column == old_cat.catalog_info.dec_column
    assert catalog.catalog_info.total_rows == old_cat.catalog_info.total_rows
    assert len(old_cat.catalog_info.default_columns) > 0
    assert catalog.catalog_info.default_columns == old_cat.catalog_info.default_columns
    assert catalog.catalog_info.__pydantic_extra__["obs_regime"] == "Optical"
    assert len(catalog.get_healpix_pixels()) == 4

    # Check that the schema is correct for leaf parquet and _metadata files
    original_common_md = paths.get_common_metadata_pointer(old_cat.catalog_base_dir)
    expected_parquet_schema = pq.read_metadata(original_common_md).schema.to_arrow_schema()
    new_schema = paths.get_common_metadata_pointer(catalog.catalog_base_dir)
    schema = pq.read_metadata(new_schema).schema.to_arrow_schema()
    assert schema.equals(expected_parquet_schema)
    schema = pq.read_metadata(args.catalog_path / "dataset" / "_metadata").schema.to_arrow_schema()
    assert schema.equals(expected_parquet_schema)
    output_file = args.catalog_path / "dataset" / "Norder=1" / "Dir=0" / "Npix=44.parquet"
    schema = pq.read_metadata(output_file).schema.to_arrow_schema()
    assert schema.equals(expected_parquet_schema)

    # Check that, when re-loaded as a pandas dataframe, the appropriate numeric types are used.
    data_frame = pd.read_parquet(output_file, engine="pyarrow")
    expected_dtypes = expected_parquet_schema.empty_table().to_pandas().dtypes
    assert data_frame.dtypes.equals(expected_dtypes)
