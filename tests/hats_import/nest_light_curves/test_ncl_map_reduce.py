import os
import shutil
from pathlib import Path

import numpy.testing as npt
import pandas as pd
import pyarrow.parquet as pq
import pytest
from hats.catalog.partition_info import PartitionInfo
from hats.pixel_math.healpix_pixel import HealpixPixel

from hats_import.nest_light_curves.arguments import NestLightCurveArguments
from hats_import.nest_light_curves.map_reduce import count_joins
from hats_import.nest_light_curves.resume_plan import NestLightCurvePlan


def test_count_joins(small_sky_ncl_args, tmp_path, small_sky_ncl_map):
    """Test counting association between object and source catalogs."""
    for object, sources in small_sky_ncl_map.items():
        count_joins(small_sky_ncl_args, object, sources)

        result = PartitionInfo.read_from_csv(
            tmp_path / "small_sky_light_curve" / "intermediate" / f"{object.order}_{object.pixel}.csv"
        )
        assert len(result) == 4
