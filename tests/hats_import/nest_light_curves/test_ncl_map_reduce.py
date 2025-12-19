import pytest
from hats.catalog.partition_info import PartitionInfo
from hats.pixel_math.healpix_pixel import HealpixPixel

from hats_import.nest_light_curves.arguments import NestLightCurveArguments
from hats_import.nest_light_curves.map_reduce import _perform_nest


def test_perform_nest(small_sky_ncl_args, tmp_path, small_sky_ncl_map):
    for object_pixel, sources in small_sky_ncl_map.items():
        _perform_nest(small_sky_ncl_args, object_pixel, sources)

        result = PartitionInfo.read_from_csv(
            tmp_path
            / "small_sky_light_curve"
            / "intermediate"
            / f"{object_pixel.order}_{object_pixel.pixel}.csv"
        )
        assert len(result) == 4


def test_invalid_pixel(small_sky_ncl_args, capsys):
    with pytest.raises(FileNotFoundError, match="No such file or directory"):
        _perform_nest(small_sky_ncl_args, HealpixPixel(0, 3), [])
    captured = capsys.readouterr()
    assert "No such file or directory" in captured.out


def test_no_matching_sources(small_sky_ncl_args, tmp_path):
    """This is a bit unrealistic - we wouldn't pass empty source list,
    but this tests the case where no sources are found for objects in this partition."""

    object_pixel = HealpixPixel(0, 11)
    _perform_nest(small_sky_ncl_args, object_pixel, [])

    result = PartitionInfo.read_from_csv(
        tmp_path / "small_sky_light_curve" / "intermediate" / f"{object_pixel.order}_{object_pixel.pixel}.csv"
    )
    assert len(result) == 0


def test_perform_nest_partition_strategy_source(
    small_sky_object_catalog, small_sky_source_catalog, tmp_path, small_sky_ncl_map
):
    small_sky_ncl_args = NestLightCurveArguments(
        object_catalog_dir=small_sky_object_catalog,
        object_id_column="id",
        source_catalog_dir=small_sky_source_catalog,
        source_object_id_column="object_id",
        source_nested_columns=["mjd", "mag", "band"],
        nested_column_name="photometry",
        output_artifact_name="small_sky_light_curve",
        partition_strategy="source_count",
        partition_threshold=5000,
        highest_healpix_order=5,
        output_path=tmp_path,
        progress_bar=False,
    )

    for object_pixel, sources in small_sky_ncl_map.items():
        _perform_nest(small_sky_ncl_args, object_pixel, sources)

        result = PartitionInfo.read_from_csv(
            tmp_path
            / "small_sky_light_curve"
            / "intermediate"
            / f"{object_pixel.order}_{object_pixel.pixel}.csv"
        )
        assert len(result) == 10


def test_perform_nest_partition_strategy_mem(
    small_sky_object_catalog, small_sky_source_catalog, tmp_path, small_sky_ncl_map
):
    small_sky_ncl_args = NestLightCurveArguments(
        object_catalog_dir=small_sky_object_catalog,
        object_id_column="id",
        source_catalog_dir=small_sky_source_catalog,
        source_object_id_column="object_id",
        source_nested_columns=["mjd", "mag", "band"],
        nested_column_name="photometry",
        output_artifact_name="small_sky_light_curve",
        partition_strategy="mem_size",
        partition_threshold=250_000,
        highest_healpix_order=5,
        output_path=tmp_path,
        progress_bar=False,
    )

    for object_pixel, sources in small_sky_ncl_map.items():
        _perform_nest(small_sky_ncl_args, object_pixel, sources)

        result = PartitionInfo.read_from_csv(
            tmp_path
            / "small_sky_light_curve"
            / "intermediate"
            / f"{object_pixel.order}_{object_pixel.pixel}.csv"
        )
        assert len(result) == 13
