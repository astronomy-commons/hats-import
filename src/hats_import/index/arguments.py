"""Utility to hold all arguments required throughout indexing"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, field
from pathlib import Path

from hats import read_hats
from hats.catalog import Catalog, TableProperties
from hats.io.validation import is_valid_catalog
from packaging.version import Version
from upath import UPath

from hats_import.runtime_arguments import RuntimeArguments


@dataclass
class IndexArguments(RuntimeArguments):
    """Data class for holding indexing arguments"""

    ## Input
    input_catalog_path: str | Path | UPath | None = None
    input_catalog: Catalog | None = None
    indexing_column: str = ""
    extra_columns: list[str] = field(default_factory=list)

    ## Output
    include_healpix_29: bool = True
    """Include the healpix-based hats spatial index."""
    include_order_pixel: bool = True
    """Include partitioning columns, Norder, Dir, and Npix. You probably want to keep these!"""
    include_radec: bool = False
    """Include the ra/dec coordinates of the row."""
    drop_duplicates: bool = True
    """Should we check for duplicate rows (including new indexing column),
    and remove duplicates before writing to new index catalog?
    If you know that your data will not have duplicates (e.g. an index over
    a unique primary key), set to False to avoid unnecessary work."""

    ## Compute parameters
    compute_partition_size: int = 1_000_000_000
    """partition size used when creating leaf parquet files."""
    division_hints: list | None = None
    """Hints used when splitting up the rows by the new index. If you have
    some prior knowledge about the distribution of your indexing_column, 
    providing it here can speed up calculations dramatically. Note that
    these will NOT necessarily be the divisions that the data is partitioned
    along."""

    def __post_init__(self):
        self.check_versions()
        self._check_arguments()

    @classmethod
    def check_versions(cls):
        """Check for version incompatibility.

        There is a significant regression with python 3.11 and dask expr, after v2025.3.0

        No other known combinations cause this problem, and it is destructive to have a global
        pin for these versions.
        """
        python_version = Version.from_parts(release=sys.version_info[:2])
        if python_version == Version("3.11"):
            dask_version = "2025.4.0"
            try:
                dask_version = importlib.metadata.version("dask")
            except Exception:  # pylint: disable=broad-exception-caught # pragma: no cover
                pass
            if not Version("2025.3.0") <= Version(dask_version) < Version("2025.4.0"):
                raise RuntimeError(
                    "dask version must be >=2025.3.0,<2025.4.0, if using python 3.11 "
                    f"(found dask {dask_version} and python {python_version})"
                )

    def _check_arguments(self):
        super()._check_arguments()
        if not self.input_catalog_path:
            raise ValueError("input_catalog_path is required")
        if not self.indexing_column:
            raise ValueError("indexing_column is required")

        if not self.include_healpix_29 and not self.include_order_pixel:
            raise ValueError("At least one of include_healpix_29 or include_order_pixel must be True")

        if not is_valid_catalog(self.input_catalog_path):
            raise ValueError("input_catalog_path not a valid catalog")
        self.input_catalog = read_hats(catalog_path=self.input_catalog_path)
        if self.include_radec:
            catalog_info = self.input_catalog.catalog_info
            self.extra_columns.extend([catalog_info.ra_column, catalog_info.dec_column])
        if len(self.extra_columns) > 0:
            # check that they're in the schema
            schema = self.input_catalog.schema
            missing_fields = [x for x in self.extra_columns if schema.get_field_index(x) == -1]
            if len(missing_fields):
                raise ValueError(f"Some requested columns not in input catalog ({','.join(missing_fields)})")
        # Remove duplicates, preserving order
        extra_columns = []
        for x in self.extra_columns:
            if x not in extra_columns:
                extra_columns.append(x)
        self.extra_columns = extra_columns

        if self.compute_partition_size < 100_000:
            raise ValueError("compute_partition_size must be at least 100_000")

    def to_table_properties(self, total_rows: int) -> TableProperties:
        """Catalog-type-specific dataset info."""
        info = {
            "catalog_name": self.output_artifact_name,
            "catalog_type": "index",
            "total_rows": total_rows,
            "primary_catalog": str(self.input_catalog_path),
            "indexing_column": self.indexing_column,
            "extra_columns": self.extra_columns,
        } | self.extra_property_dict()

        return TableProperties(**info)
