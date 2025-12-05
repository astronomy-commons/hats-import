from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hats import read_hats
from hats.catalog import TableProperties
from hats.io.validation import is_valid_catalog
from upath import UPath

from hats_import.runtime_arguments import RuntimeArguments


@dataclass
class NestLightCurveArguments(RuntimeArguments):
    """Data class for holding nesting lightcurve (source-object nesting) arguments"""

    ## Input - Object catalog
    object_catalog_dir: str | Path | UPath | None = None
    object_id_column: str = ""

    ## Input - Source catalog
    source_catalog_dir: str | Path | UPath | None = None
    source_object_id_column: str = ""
    source_id_column: str = ""
    source_nested_columns: list[str] = None
    """What columns from the source table should we keep in the nested light curve?
    Note that source tables often include repeated object data, so make sure
    to drop those from this list."""

    ## Output - nested light curves
    nested_column_name: str = "light_curve"
    """The name for the group of nested columns."""
    nested_sort_column: str = ""
    """Field to sort the nested frame on. Typically timestamp."""
    npix_suffix: str = "parquet"
    partition_strategy: str = "object_count"
    partition_threshold: int = 50
    highest_order: int = 3
    lowest_order: int = 0

    def __post_init__(self):
        self._check_arguments()

    def _check_arguments(self):
        super()._check_arguments()
        if not self.object_catalog_dir:
            raise ValueError("object_catalog_dir is required")
        if not self.object_id_column:
            raise ValueError("object_id_column is required")
        if not is_valid_catalog(self.object_catalog_dir):
            raise ValueError("object_catalog_dir not a valid catalog")

        self.object_catalog = read_hats(catalog_path=self.object_catalog_dir)

        ## TODO: check the object_id_column is in the object schema
        ## TODO: check that we have a `_healpix_29` column - we use it!

        if not self.source_catalog_dir:
            raise ValueError("source_catalog_dir is required")
        if not self.source_object_id_column:
            raise ValueError("source_object_id_column is required")
        if not is_valid_catalog(self.source_catalog_dir):
            raise ValueError("source_catalog_dir not a valid catalog")

        self.source_catalog = read_hats(catalog_path=self.source_catalog_dir)

        ## TODO: check the all the columns are in the source schema
        ## TODO: check partition strategy, threshold, and orders.

    def read_source_columns(self):
        if self.source_nested_columns is None:
            ## We intend to read all columns
            return None
        return list(set(self.source_nested_columns) | set([self.source_object_id_column]))

    def to_table_properties(self, total_rows=10, highest_order=4, moc_sky_fraction=22 / 7) -> TableProperties:
        """Catalog-type-specific dataset info."""
        info = {
            "catalog_name": self.output_artifact_name,
            "total_rows": total_rows,
            "hats_order": highest_order,
            "moc_sky_fraction": f"{moc_sky_fraction:0.5f}",
        } | self.extra_property_dict()
        return TableProperties(**info)
