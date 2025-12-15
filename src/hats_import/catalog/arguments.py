"""Utility to hold all arguments required throughout partitioning"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import hats
from hats.catalog import TableProperties
from hats.catalog.catalog_collection import CatalogCollection
from hats.io.file_io import get_upath
from hats.io.paths import DATASET_DIR, HIVE_COLUMNS, PARTITION_ORDER
from hats.io.validation import is_valid_catalog
from hats.pixel_math.spatial_index import SPATIAL_INDEX_COLUMN, SPATIAL_INDEX_ORDER
from typing_extensions import Self
from upath import UPath

from hats_import.catalog.file_readers import InputReader, get_file_reader
from hats_import.catalog.file_readers.parquet import ParquetPyarrowReader
from hats_import.runtime_arguments import RuntimeArguments, check_healpix_order_range, find_input_paths

# pylint: disable=too-many-locals,too-many-arguments,too-many-instance-attributes,too-many-branches,too-few-public-methods


@dataclass
class ImportArguments(RuntimeArguments):
    """Container class for holding arguments for partitioning input
    data into a HATS catalog."""

    catalog_type: str = "object"
    """level of catalog data, object (things in the sky) or source (detections)"""
    allowed_catalog_types: tuple[str] = ("source", "object", "map")
    """possible types of catalog to import with `ImportArguments`"""
    input_path: str | Path | UPath | None = None
    """path to search for the input data"""
    input_file_list: list[str | Path | UPath] = field(default_factory=list)
    """can be used instead of input_path to import only specified files"""
    input_paths: list[str | Path | UPath] = field(default_factory=list)
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
    byte_pixel_threshold: int | None = None
    """when determining bins for the final partitioning, the maximum number
    of rows for a single resulting pixel, expressed in bytes. we may combine hierarchically until
    we near the ``byte_pixel_threshold``. if this is set, it will override
    ``pixel_threshold``."""
    drop_empty_siblings: bool = True
    """when determining bins for the final partitioning, should we keep result pixels
    at a higher order (smaller area) if the 3 sibling pixels are empty. setting this to 
    False will result in the same number of result pixels, but they may differ in Norder"""
    mapping_healpix_order: int = -1
    """healpix order to use when mapping. will be
    ``highest_healpix_order`` unless a positive value is provided for
    ``constant_healpix_order``"""
    run_stages: list[str] = field(default_factory=list)
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
    existing_pixels: Sequence[tuple[int, int]] | None = None
    """the list of HEALPix pixels to include in the alignment"""

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

        if self.existing_pixels:
            max_existing_order = max(p[0] for p in self.existing_pixels)
            if self.mapping_healpix_order < max_existing_order:
                raise ValueError("`highest_order` needs to be >= current `existing_pixels` order")

        if self.catalog_type not in self.allowed_catalog_types:
            raise ValueError(f"catalog_type should be one of {self.allowed_catalog_types}")

        if self.file_reader is None:
            raise ValueError("file_reader is required")
        if isinstance(self.file_reader, str):
            self.file_reader = get_file_reader(self.file_reader)

        if self.use_healpix_29:
            self.add_healpix_29 = False
            if self.sort_columns:
                raise ValueError("When using _healpix_29 for position, no sort columns should be added")

        # Validate byte_pixel_threshold
        if self.byte_pixel_threshold is not None:
            if not isinstance(self.byte_pixel_threshold, int):
                raise TypeError("byte_pixel_threshold must be an integer")
            if self.byte_pixel_threshold < 0:
                raise ValueError("byte_pixel_threshold must be non-negative")

        # Basic checks complete - make more checks and create directories where necessary
        self.input_paths = find_input_paths(self.input_path, "**/*.*", self.input_file_list)

    def to_table_properties(
        self,
        total_rows: int,
        highest_order: int,
        moc_sky_fraction: float,
        column_names=None,
    ) -> TableProperties:
        """Catalog-type-specific dataset info."""
        info = self.extra_property_dict() | {
            "catalog_name": self.output_artifact_name,
            "catalog_type": self.catalog_type,
            "total_rows": total_rows,
            "ra_column": self.ra_column,
            "dec_column": self.dec_column,
            "hats_cols_sort": self.sort_columns,
            "hats_max_rows": self.pixel_threshold,
            "hats_order": highest_order,
            "moc_sky_fraction": f"{moc_sky_fraction:0.5f}",
            "hats_npix_suffix": self.npix_suffix,
        }
        if self.should_write_skymap:
            info.update(
                {
                    "hats_skymap_order": self.mapping_healpix_order,
                    "hats_skymap_alt_orders": self.skymap_alt_orders,
                }
            )
        if self.add_healpix_29:
            info.update(
                {
                    "hats_col_healpix": SPATIAL_INDEX_COLUMN,
                    "hats_col_healpix_order": SPATIAL_INDEX_ORDER,
                }
            )
        properties = TableProperties(**info)

        if properties.default_columns and column_names:
            missing_columns = set(properties.default_columns) - set(column_names)
            if len(missing_columns):
                raise ValueError(f"Some default columns not found in catalog: {missing_columns}")

        return properties

    @classmethod
    def reimport_from_hats(cls, path: str | Path | UPath, output_dir: str | Path | UPath, **kwargs) -> Self:
        """Generate the import arguments to reimport a HATS catalog with different parameters

        Args:
            path (str | Path | UPath): the path to the existing HATS catalog to reimport
            output_dir (str | Path | UPath): the path to output the reimported catalog to
            kwargs: any import arguments to update from the existing catalog. Can be any argument usually
                passed to :func:`ImportArguments`

        Returns:
            A ImportArguments object with the arguments from the existing catalog, and any updates from kwargs
        """

        path = get_upath(path)

        catalog = hats.read_hats(path)

        if isinstance(catalog, CatalogCollection):
            path = catalog.main_catalog_dir
            catalog = catalog.main_catalog
        if not is_valid_catalog(path, strict=True):
            raise ValueError("path not a valid catalog")

        column_names = None
        if catalog.schema is not None:
            column_names = [name for name in catalog.schema.names if name not in HIVE_COLUMNS]

        in_file_paths = list(
            (path / DATASET_DIR).rglob(f"{PARTITION_ORDER}*/**/*{catalog.catalog_info.npix_suffix}")
        )

        addl_hats_properties = catalog.catalog_info.extra_dict(by_alias=True)

        addl_hats_properties.update(
            {
                "hats_cols_default": catalog.catalog_info.default_columns,
                "hats_npix_suffix": catalog.catalog_info.npix_suffix,
            }
        )

        if "addl_hats_properties" in kwargs:
            addl_hats_properties.update(kwargs.pop("addl_hats_properties"))

        existing_pixels = None
        if "existing_pixels" in kwargs:
            existing_pixels = kwargs.pop("existing_pixels")

        import_args = {
            "catalog_type": catalog.catalog_info.catalog_type,
            "ra_column": catalog.catalog_info.ra_column,
            "dec_column": catalog.catalog_info.dec_column,
            "input_file_list": in_file_paths,
            "file_reader": ParquetPyarrowReader(column_names=column_names),
            "output_artifact_name": catalog.catalog_name,
            "output_path": output_dir,
            "use_healpix_29": True,
            "add_healpix_29": False,
            "use_schema_file": hats.io.paths.get_common_metadata_pointer(catalog.catalog_base_dir),
            "expected_total_rows": catalog.catalog_info.total_rows,
            "addl_hats_properties": addl_hats_properties,
            "existing_pixels": existing_pixels,
        }

        import_args.update(**kwargs)
        return cls(**import_args)  # type: ignore
