import hats
import pytest
from hats.io.paths import DATASET_DIR

from hats_import.catalog import ImportArguments
from hats_import.catalog.file_readers import ParquetReader


def test_reimport_arguments(tmp_path, small_sky_object_catalog):
    args = ImportArguments.reimport_from_hats(small_sky_object_catalog, tmp_path)
    catalog = hats.read_hats(small_sky_object_catalog)
    assert args.catalog_type == catalog.catalog_info.catalog_type
    assert args.ra_column == catalog.catalog_info.ra_column
    assert args.dec_column == catalog.catalog_info.dec_column
    assert args.input_path == catalog.catalog_base_dir / DATASET_DIR
    assert isinstance(args.file_reader, ParquetReader)
    assert args.output_artifact_name == catalog.catalog_name
    assert args.use_healpix_29
    assert not args.add_healpix_29


def test_reimport_arguments_extra_kwargs(tmp_path, small_sky_object_catalog):
    output_name = "small_sky_higher_order"
    pixel_thresh = 100
    args = ImportArguments.reimport_from_hats(
        small_sky_object_catalog, tmp_path, pixel_threshold=pixel_thresh, output_artifact_name=output_name
    )
    catalog = hats.read_hats(small_sky_object_catalog)
    assert args.catalog_type == catalog.catalog_info.catalog_type
    assert args.ra_column == catalog.catalog_info.ra_column
    assert args.dec_column == catalog.catalog_info.dec_column
    assert args.input_path == catalog.catalog_base_dir / DATASET_DIR
    assert isinstance(args.file_reader, ParquetReader)
    assert args.output_artifact_name == output_name
    assert args.pixel_threshold == pixel_thresh
    assert args.use_healpix_29
    assert not args.add_healpix_29


def test_reimport_arguments_wrong_dir(tmp_path):
    wrong_input_path = tmp_path / "nonsense"
    with pytest.raises(FileNotFoundError):
        ImportArguments.reimport_from_hats(wrong_input_path, tmp_path)
