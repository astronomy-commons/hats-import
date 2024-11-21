"""Utility to hold all arguments required throughout partitioning"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from hats.catalog import TableProperties
from hats.pixel_math import spatial_index
from upath import UPath

from hats_import.catalog.file_readers import InputReader, get_file_reader
from hats_import.runtime_arguments import RuntimeArguments, find_input_paths

# pylint: disable=too-many-locals,too-many-arguments,too-many-instance-attributes,too-many-branches,too-few-public-methods


@dataclass
class ImportArguments(RuntimeArguments):
    """Container class for holding partitioning arguments"""

    catalog_type: str = "object"
    """level of catalog data, object (things in the sky) or source (detections)"""
    input_path: str | Path | UPath | None = None
    """path to search for the input data"""
    input_file_list: List[str | Path | UPath] = field(default_factory=list)
    """can be used instead of input_path to import only specified files"""
    input_paths: List[str | Path | UPath] = field(default_factory=list)
    """resolved list of all files that will be used in the importer"""

    ra_column: str = "ra"
    """column for right ascension"""
    dec_column: str = "dec"
    """column for declination"""
    use_healpix_29: bool = False
    """use an existing healpix-based hats spatial index as the position, instead of ra/dec"""
    sort_columns: str | None = None
    """column for survey identifier, or other sortable column. if sorting by multiple columns,
    they should be comma-separated. If `add_healpix_29=True`, `_healpix_29` will be the primary sort key, 
    but the provided sorting will be used for any rows within the same higher-order pixel space."""
    add_healpix_29: bool = True
    """add the healpix-based hats spatial index field alongside the data"""
    use_schema_file: str | Path | UPath | None = None
    """path to a parquet file with schema metadata. this will be used for column
    metadata when writing the files, if specified"""
    expected_total_rows: int = 0
    """number of expected rows found in the dataset. if non-zero, and we find we have 
    a different number of rows, the pipeline will exit."""
    constant_healpix_order: int = -1
    """healpix order to use when mapping. if this is
    a positive number, this will be the order of all final pixels and we
    will not combine pixels according to the threshold"""
    lowest_healpix_order: int = 0
    """when determining bins for the final partitioning, the lowest possible healpix order 
    for resulting pixels. setting this higher than 0 will prevent creating
    partitions with a large area on the sky."""
    highest_healpix_order: int = 10
    """healpix order to use when mapping. this will
    not necessarily be the order used in the final catalog, as we may combine
    pixels that don't meed the threshold"""
    pixel_threshold: int = 1_000_000
    """when determining bins for the final partitioning, the maximum number 
    of rows for a single resulting pixel. we may combine hierarchically until 
    we near the ``pixel_threshold``"""
    drop_empty_siblings: bool = False
    """when determining bins for the final partitioning, should we keep result pixels
    at a higher order (smaller area) if the 3 sibling pixels are empty. setting this to 
    False will result in the same number of result pixels, but they may differ in Norder"""
    mapping_healpix_order: int = -1
    """healpix order to use when mapping. will be
    ``highest_healpix_order`` unless a positive value is provided for
    ``constant_healpix_order``"""
    run_stages: List[str] = field(default_factory=list)
    """list of parallel stages to run. options are ['mapping', 'splitting', 'reducing',
    'finishing']. ['planning', 'binning'] stages are not optional.
    this can be used to force the pipeline to stop after an early stage, to allow the
    user to reset the dask client with different resources for different stages of
    the workflow. if not specified, we will run all pipeline stages."""
    debug_stats_only: bool = False
    """do not perform a map reduce and don't create a new
    catalog. generate the partition info"""
    file_reader: InputReader | str | None = None
    """instance of input reader that specifies arguments necessary for reading
    from your input files"""

    def __post_init__(self):
        self._check_arguments()

    def _check_arguments(self):
        """Check existence and consistency of argument values"""
        super()._check_arguments()

        if self.lowest_healpix_order == self.highest_healpix_order:
            self.constant_healpix_order = self.lowest_healpix_order
        if self.constant_healpix_order >= 0:
            check_healpix_order_range(self.constant_healpix_order, "constant_healpix_order")
            self.mapping_healpix_order = self.constant_healpix_order
        else:
            check_healpix_order_range(self.highest_healpix_order, "highest_healpix_order")
            check_healpix_order_range(
                self.lowest_healpix_order, "lowest_healpix_order", upper_bound=self.highest_healpix_order
            )
            if not 100 <= self.pixel_threshold <= 1_000_000_000:
                raise ValueError("pixel_threshold should be between 100 and 1,000,000,000")
            self.mapping_healpix_order = self.highest_healpix_order

        if self.catalog_type not in ("source", "object", "map"):
            raise ValueError("catalog_type should be one of `source`, `object`, or `map`")

        if self.file_reader is None:
            raise ValueError("file_reader is required")
        if isinstance(self.file_reader, str):
            self.file_reader = get_file_reader(self.file_reader)

        if self.use_healpix_29:
            self.add_healpix_29 = False
            if self.sort_columns:
                raise ValueError("When using _healpix_29 for position, no sort columns should be added")

        # Basic checks complete - make more checks and create directories where necessary
        self.input_paths = find_input_paths(self.input_path, "**/*.*", self.input_file_list)

    def to_table_properties(
        self, total_rows: int, highest_order: int, moc_sky_fraction: float
    ) -> TableProperties:
        """Catalog-type-specific dataset info."""
        info = {
            "catalog_name": self.output_artifact_name,
            "catalog_type": self.catalog_type,
            "total_rows": total_rows,
            "ra_column": self.ra_column,
            "dec_column": self.dec_column,
            "hats_cols_sort": self.sort_columns,
            "hats_max_rows": self.pixel_threshold,
            "hats_order": highest_order,
            "moc_sky_fraction": f"{moc_sky_fraction:0.5f}",
        } | self.extra_property_dict()
        return TableProperties(**info)


def check_healpix_order_range(
    order, field_name, lower_bound=0, upper_bound=spatial_index.SPATIAL_INDEX_ORDER
):
    """Helper method to check if the ``order`` is within the range determined by the
    ``lower_bound`` and ``upper_bound``, inclusive.

    Args:
        order (int): healpix order to check
        field_name (str): field name to use in the error message
        lower_bound (int): lower bound of range
        upper_bound (int): upper bound of range
    Raise:
        ValueError: if the order is outside the specified range, or bounds
            are unreasonable.
    """
    if lower_bound < 0:
        raise ValueError("healpix orders must be positive")
    if upper_bound > spatial_index.SPATIAL_INDEX_ORDER:
        raise ValueError(f"healpix order should be <= {spatial_index.SPATIAL_INDEX_ORDER}")
    if not lower_bound <= order <= upper_bound:
        raise ValueError(f"{field_name} should be between {lower_bound} and {upper_bound}")
