from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from hats import read_hats
from hats.catalog import TableProperties
from hats.io.validation import is_valid_catalog
from upath import UPath

from hats_import.runtime_arguments import RuntimeArguments, check_healpix_order_range


@dataclass
class NestLightCurveArguments(RuntimeArguments):
    """Data class for holding nesting lightcurve (source-object nesting) arguments"""

    ## Input - Object catalog
    object_catalog_dir: str | Path | UPath | None = None
    object_id_column: str = ""

    ## Input - Source catalog
    source_catalog_dir: str | Path | UPath | None = None
    source_object_id_column: str = ""
    source_nested_columns: list[str] = None
    """What columns from the source table should we keep in the nested light curve?
    Note that source tables often include repeated object data, so make sure
    to drop those from this list."""

    ## Output - nested light curves
    nested_column_name: str = "light_curve"
    """The name for the group of nested columns."""
    nested_sort_column: str | None = None
    """Field to sort the nested frame on. Typically timestamp."""
    partition_strategy: str = "object_count"
    partition_threshold: int = 50
    highest_healpix_order: int = 3

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

        object_catalog = read_hats(catalog_path=self.object_catalog_dir)
        self.object_npix_suffix = object_catalog.catalog_info.npix_suffix
        self.object_catalog_schema = object_catalog.schema
        self.object_catalog_info = object_catalog.catalog_info

        if self.object_id_column not in self.object_catalog_schema.names:
            raise ValueError(f"object_id_column ({self.object_id_column}) not found in object catalog")

        if self.addl_hats_properties is None:
            self.addl_hats_properties = {}

        self.addl_hats_properties |= self.object_catalog_info.extra_dict()

        if not self.source_catalog_dir:
            raise ValueError("source_catalog_dir is required")
        if not self.source_object_id_column:
            raise ValueError("source_object_id_column is required")
        if not is_valid_catalog(self.source_catalog_dir):
            raise ValueError("source_catalog_dir not a valid catalog")

        source_catalog = read_hats(catalog_path=self.source_catalog_dir)
        self.source_npix_suffix = source_catalog.catalog_info.npix_suffix

        if self.source_nested_columns is not None:
            if pd.api.types.is_list_like(self.source_nested_columns):
                self.source_nested_columns = set(self.source_nested_columns)
            else:
                self.source_nested_columns = set([self.source_nested_columns])

        source_columns = (
            self.source_nested_columns
            if self.source_nested_columns is not None
            else set() | set([self.source_object_id_column])
        )
        self.source_catalog_schema = source_catalog.schema
        if source_columns:
            missing_columns = source_columns - set(source_catalog.schema.names)
            if len(missing_columns):
                raise ValueError(f"Some columns not found in source catalog: {missing_columns}")

        check_healpix_order_range(self.highest_healpix_order, "highest_healpix_order")

        if self.partition_strategy not in ["object_count", "source_count", "mem_size"]:
            raise ValueError("Unrecognized partition_strategy")

        if not isinstance(self.partition_threshold, int):
            raise TypeError("partition_threshold must be an integer")
        if self.partition_threshold < 0:
            raise ValueError("partition_threshold must be non-negative")

    def read_source_columns(self):
        """Determine list of columns to be read in the source catalog."""
        if self.source_nested_columns is None:
            # We intend to read all columns
            return None
        read_columns = list(set(self.source_nested_columns) | set([self.source_object_id_column]))
        read_columns.sort()
        return read_columns

    def to_table_properties(self, total_rows=10, highest_order=4, moc_sky_fraction=22 / 7) -> TableProperties:
        """Catalog-type-specific dataset info."""
        info = (
            {
                "catalog_name": self.output_artifact_name,
                "catalog_type": "object",
                "total_rows": total_rows,
                "hats_order": highest_order,
                "moc_sky_fraction": f"{moc_sky_fraction:0.5f}",
                "ra_column": self.object_catalog_info.ra_column,
                "dec_column": self.object_catalog_info.dec_column,
                "hats_col_healpix": self.object_catalog_info.healpix_column,
                "hats_col_healpix_order": self.object_catalog_info.healpix_order,
            }
            | self.object_catalog_info.extra_dict()
            | self.extra_property_dict()
        )
        return TableProperties(**info)
