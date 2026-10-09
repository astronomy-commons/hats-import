"""Utility to hold all arguments required for extension splitting"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from hats import read_hats
from hats.catalog import Catalog, CatalogCollection, CatalogType, MarginCatalog
from hats.io.validation import is_valid_catalog
from hats.pixel_math.spatial_index import SPATIAL_INDEX_COLUMN
from nested_pandas.nestedframe.io import from_pyarrow
from upath import UPath

from hats_import.extension.split_side import SplitSide
from hats_import.runtime_arguments import RuntimeArguments


@dataclass
class ExtensionArguments(RuntimeArguments):
    """Container class for holding arguments for splitting an input HATS Catalog
    into a "core" Catalog and its corresponding Extension.

    The output is always a collection, written to ``<output_path>/<output_artifact_name>``.
    The core keeps the name of the input catalog, and the extension, a collection of its own,
    is named ``extension_name``. Both keep the partitioning of the input, so each extension
    partition can be joined to the core partition with the same HEALPix pixel::

        small_sky_collection/
        ├── small_sky/                        the core
        ├── small_sky_margin/                 a margin of the input, split for the core
        ├── small_sky_id_index/               an index over a column that stays in the core, copied
        ├── small_sky_spectra/                the extension, a collection of its own
        │   ├── small_sky_spectra/
        │   ├── small_sky_spectra_margin/     the same margin, split for the extension
        │   └── collection.properties
        ├── small_sky_spectra.properties      describes the extension, and how to join it to the core
        └── collection.properties             lists the margins, indexes and extensions

    A few things to know about the layout:

    * Each margin of the input is split in two, so the core and the extension each get a margin
      with their own columns.
    * Each index follows the column it indexes. If that column moves to the extension, the index
      moves too (and is renamed after the extension). Otherwise, it stays with the core.
    """

    ## Input
    input_catalog_path: str | Path | UPath | None = None
    """path to the existing HATS catalog to split"""
    input_catalog: Catalog = None
    """the loaded input catalog. For a collection, this is its main catalog"""
    input_collection: CatalogCollection | None = None
    """the loaded input collection, if the input path is a collection"""
    extension_columns: list[str] = field(default_factory=list)
    """the set of column names to move from the input catalog into the extension. 
    They are removed from the core. The spatial index, the join key and the coordinate
    columns are copied into the extension on top of these, and stay in the core, so they
    cannot be listed here."""
    primary_column: str = ""
    """column of the core catalog used to join the extension back to the core"""
    join_column: str = ""
    """name of the column in the extension that matches `primary_column`.
    Defaults to the same name as `primary_column`."""

    ## Output
    extension_name: str = ""
    """name of the extension, e.g. "small_sky_spectra". It names the extension's collection, its
    main catalog and its ``<extension_name>.properties`` file. Its margins and indexes take it in
    place of the input catalog's name: ``small_sky_margin`` becomes ``small_sky_spectra_margin``."""
    join_style: Literal["left", "inner"] = "left"
    """the type of join to use when combining the extension with the core"""
    product_type_served: str | None = None
    """modality of the data that the extension stores (e.g. "spectra", "images")"""

    ## Constructed
    core: SplitSide = None
    """the core side of the split"""
    extension: SplitSide = None
    """the extension side of the split"""
    margins: list[MarginCatalog] = field(default_factory=list)
    """the margins of the input collection, each split into both sides"""
    indexes: dict[str, UPath] = field(default_factory=dict)
    """the index catalogs of the input collection, by the column each one indexes. Each goes to
    the side that keeps its column"""
    default_margin_name: str | None = None
    """name of the input collection's default margin, if it has one"""
    default_index_column: str | None = None
    """column of the input collection's default index, if it has one"""

    def _check_arguments(self):
        super()._check_arguments()
        self._read_input()

        if not self.extension_name:
            raise ValueError("extension_name is required")
        if re.search(r"[^A-Za-z0-9\._\-\\]", self.extension_name):
            raise ValueError("extension_name contains invalid characters")

        core_columns, extension_columns = self._check_columns()
        if self.input_collection is not None:
            self._plan_collection_members()
        self._build_sides(core_columns, extension_columns)
        self._check_extension_name_conflicts()

    def _read_input(self):
        """Read the input path, which may be a single catalog or a whole collection."""
        if not self.input_catalog_path:
            raise ValueError("input_catalog_path is required")
        if not is_valid_catalog(self.input_catalog_path):
            raise ValueError("input_catalog_path not a valid catalog or collection")
        input_dataset = read_hats(self.input_catalog_path)
        if isinstance(input_dataset, CatalogCollection):
            self.input_collection = input_dataset
            self.input_catalog = input_dataset.main_catalog
        else:
            self.input_catalog = input_dataset
        if self.input_catalog.catalog_info.catalog_type not in (CatalogType.OBJECT, CatalogType.SOURCE):
            raise ValueError("Extensions can only be created from OBJECT or SOURCE catalogs")

    def _check_columns(self):
        """Check that the requested columns can be split out of the input catalog.

        Returns the columns of the core and of the extension, mapped to their new names."""
        if not self.extension_columns:
            raise ValueError("extension_columns is required")
        if not self.primary_column:
            raise ValueError("primary_column is required")
        if not self.join_column:
            self.join_column = self.primary_column

        # Remove duplicates and preserve order.
        self.extension_columns = list(dict.fromkeys(self.extension_columns))

        column_names = self.input_catalog.schema.names
        # A whole nested column can move to the extension, but a single field of one cannot.
        subcolumns = from_pyarrow(self.input_catalog.schema.empty_table()).get_subcolumns()
        nested_fields = [col for col in self.extension_columns if col in subcolumns]
        if nested_fields:
            raise ValueError(
                f"The following columns {nested_fields} are fields of a nested column and cannot be "
                f"listed in extension_columns. List the whole nested column instead."
            )
        missing_columns = [col for col in self.extension_columns if col not in column_names]
        if missing_columns:
            raise ValueError(f"Some extension columns do not exist in the input catalog: {missing_columns}")
        if self.primary_column in subcolumns:
            raise ValueError(
                f"primary_column '{self.primary_column}' is a field of a nested column, "
                f"and cannot be used as a join key."
            )
        if self.primary_column not in column_names:
            raise ValueError(f"primary_column '{self.primary_column}' does not exist in the input catalog")

        # The healpix, join key and coordinate columns are copied into the extension, and stay in the core.
        catalog_info = self.input_catalog.catalog_info
        copied_columns = list(
            dict.fromkeys(
                [self.healpix_column, self.primary_column, catalog_info.ra_column, catalog_info.dec_column]
            )
        )
        invalid_extension_columns = [col for col in self.extension_columns if col in copied_columns]
        if invalid_extension_columns:
            raise ValueError(
                f"The following columns {invalid_extension_columns} are copied into the extension and "
                f"cannot be listed in extension_columns."
            )

        # The join key is written under its new name, so nothing else may claim that name.
        extension_input_columns = copied_columns + self.extension_columns
        if self.join_column in [col for col in extension_input_columns if col != self.primary_column]:
            raise ValueError(
                f"join_column '{self.join_column}' conflicts with an existing extension column. "
                f"Please unset join_column or choose a different name."
            )

        core_columns = {col: col for col in column_names if col not in self.extension_columns}
        extension_columns = {
            col: self.join_column if col == self.primary_column else col for col in extension_input_columns
        }
        return core_columns, extension_columns

    def _plan_collection_members(self):
        """Record the margins to split, the index catalogs to carry over, and the default margin and index."""
        collection = self.input_collection
        self.default_margin_name = collection.default_margin
        self.default_index_column = collection.default_index_field
        for margin_name in collection.all_margins or []:
            margin_dir = CatalogCollection.resolve_inner_path(collection.collection_path, margin_name)
            self.margins.append(read_hats(margin_dir))
        for column, index_name in (collection.all_indexes or {}).items():
            self.indexes[column] = CatalogCollection.resolve_inner_path(
                collection.collection_path, index_name
            )

    def _build_sides(self, core_columns: dict[str, str], extension_columns: dict[str, str]):
        """Build the two sides of the split (for the core and extension)."""
        if self.catalog_path is None:  # pragma: no cover (not reachable, but required for mypy)
            raise ValueError("catalog_path is required")
        collection_path = self.catalog_path
        output_path = collection_path.parent

        extension_indexes = {
            column: path for column, path in self.indexes.items() if column in self.extension_columns
        }
        core_indexes = {
            column: path for column, path in self.indexes.items() if column not in extension_indexes
        }

        core_name = self.input_catalog.catalog_info.catalog_name
        self.core = SplitSide(
            name=core_name,
            collection_path=collection_path,
            catalog_path=collection_path / core_name,
            input_columns=list(core_columns),
            output_columns=list(core_columns.values()),
            input_catalog_name=core_name,
            output_path=output_path,
            indexes=core_indexes,
        )
        extension_collection_path = collection_path / self.extension_name
        self.extension = SplitSide(
            name=self.extension_name,
            collection_path=extension_collection_path,
            catalog_path=extension_collection_path / self.extension_name,
            input_columns=list(extension_columns),
            output_columns=list(extension_columns.values()),
            input_catalog_name=core_name,
            output_path=output_path,
            indexes=extension_indexes,
        )

    def _check_extension_name_conflicts(self):
        """Check that the extension's name is not taken by a table of the core. The extension's
        collection sits next to the core's tables, so it cannot share a name with one."""
        core_tables = [self.core.table_path(table).name for table in self.input_tables]
        core_tables += [self.core.derived_name(index_dir.name) for index_dir in self.core.indexes.values()]
        if self.extension_name in core_tables:
            raise ValueError(
                f"extension_name '{self.extension_name}' is already the name of a table in the collection"
            )

    @property
    def sides(self) -> tuple[SplitSide, SplitSide]:
        """The two sides of the split, the core first."""
        return self.core, self.extension

    @property
    def input_tables(self) -> list:
        """The input tables to split: the main catalog, and every margin of a collection."""
        return [self.input_catalog] + self.margins

    @property
    def healpix_column(self) -> str:
        """Name of the spatial index column, kept in both the core and the extension."""
        return self.input_catalog.catalog_info.healpix_column or SPATIAL_INDEX_COLUMN
