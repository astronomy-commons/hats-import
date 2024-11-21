from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import hats.pixel_math.healpix_shim as hp
from hats import read_hats
from hats.catalog import TableProperties
from hats.io.validation import is_valid_catalog
from hats.pixel_math.healpix_pixel import HealpixPixel
from upath import UPath

from hats_import.runtime_arguments import RuntimeArguments


@dataclass
class MarginCacheArguments(RuntimeArguments):
    """Container for margin cache generation arguments"""

    margin_threshold: float = 5.0
    """the size of the margin cache boundary, given in arcseconds. setting the
    `margin_threshold` to be greater than the resolution of a `margin_order`
    healpixel will result in a warning, as this may lead to data loss."""
    margin_order: int = -1
    """the order of healpixels that will be used to constrain the margin data before
    doing more precise boundary checking. this value must be greater than the highest
    order of healpix partitioning in the source catalog. if `margin_order` is left
    default or set to -1, then the `margin_order` will be set dynamically to the
    highest partition order plus 1."""
    fine_filtering: bool = False
    """should we perform the precise boundary checking? if false, some results may be
    greater than `margin_threshold` away from the border (but within `margin_order`)."""

    input_catalog_path: str | Path | UPath | None = None
    """the path to the hats-formatted input catalog."""
    debug_filter_pixel_list: List[HealpixPixel] = field(default_factory=list)
    """debug setting. if provided, we will first filter the catalog to the pixels
    provided. this can be useful for creating a margin over a subset of a catalog."""

    def __post_init__(self):
        self._check_arguments()

    def _check_arguments(self):
        super()._check_arguments()
        if not self.input_catalog_path:
            raise ValueError("input_catalog_path is required")
        if not is_valid_catalog(self.input_catalog_path):
            raise ValueError("input_catalog_path not a valid catalog")

        self.catalog = read_hats(self.input_catalog_path)
        if len(self.debug_filter_pixel_list) > 0:
            self.catalog = self.catalog.filter_from_pixel_list(self.debug_filter_pixel_list)
            if len(self.catalog.get_healpix_pixels()) == 0:
                raise ValueError("debug_filter_pixel_list has created empty catalog")

        if self.fine_filtering:
            raise NotImplementedError("Fine filtering temporarily removed.")

        highest_order = int(self.catalog.partition_info.get_highest_order())

        if self.margin_order < 0:
            self.margin_order = hp.margin2order(margin_thr_arcmin=self.margin_threshold / 60.0)

        if self.margin_order < highest_order + 1:
            raise ValueError(
                "margin_order must be of a higher order than the highest order catalog partition pixel."
            )

        margin_pixel_mindist = hp.order2mindist(self.margin_order)
        if margin_pixel_mindist * 60.0 < self.margin_threshold:
            raise ValueError("margin pixels must be larger than margin_threshold")

    def to_table_properties(
        self, total_rows: int, highest_order: int, moc_sky_fraction: float
    ) -> TableProperties:
        """Catalog-type-specific dataset info."""
        info = {
            "catalog_name": self.output_artifact_name,
            "total_rows": total_rows,
            "catalog_type": "margin",
            "ra_column": self.catalog.catalog_info.ra_column,
            "dec_column": self.catalog.catalog_info.dec_column,
            "primary_catalog": str(self.input_catalog_path),
            "margin_threshold": self.margin_threshold,
            "hats_order": highest_order,
            "moc_sky_fraction": f"{moc_sky_fraction:0.5f}",
        } | self.extra_property_dict()
        return TableProperties(**info)
