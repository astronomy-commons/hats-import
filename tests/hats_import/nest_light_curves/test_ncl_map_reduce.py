from hats.catalog.partition_info import PartitionInfo

from hats_import.nest_light_curves.map_reduce import _perform_nest


def test_perform_nest(small_sky_ncl_args, tmp_path, small_sky_ncl_map):
    """Test counting association between object and source catalogs."""
    for object_pixel, sources in small_sky_ncl_map.items():
        _perform_nest(small_sky_ncl_args, object_pixel, sources)

        result = PartitionInfo.read_from_csv(
            tmp_path
            / "small_sky_light_curve"
            / "intermediate"
            / f"{object_pixel.order}_{object_pixel.pixel}.csv"
        )
        assert len(result) == 4
