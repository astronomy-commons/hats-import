"""Utility to hold all arguments required for extension splitting"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from hats import read_hats
from hats.catalog import Catalog, CatalogCollection, CatalogType, MarginCatalog
from hats.io import file_io
from hats.io.validation import is_valid_catalog
from hats.pixel_math.spatial_index import SPATIAL_INDEX_COLUMN
from upath import UPath

from hats_import.extension.split_side import SplitSide
from hats_import.runtime_arguments import RuntimeArguments


@dataclass
class ExtensionArguments(RuntimeArguments):
    """Container class for holding arguments for splitting an input HATS Catalog
    into a "core" Catalog and its corresponding Extension.

    The output is always a collection, written to ``<output_path>/<output_artifact_name>``.
    The core keeps the name of the input catalog, and the extension is named after it, with
    ``extension_name`` appended. Both keep the partitioning of the input, so each extension
    partition can be joined to the core partition with the same HEALPix pixel::

        small_sky_collection/
        ├── small_sky/                        the core
        ├── small_sky_margin/                 a margin of the input, split for the core
        ├── small_sky_index/                  an index of the input, copied
        ├── small_sky_spectra/                the extension, a collection when it has margins
        │   ├── small_sky_spectra/
        │   ├── small_sky_spectra_margin/     the same margin, split for the extension
        │   └── collection.properties
        └── collection.properties             lists the margins, indexes and extensions

    With no margins to hold, the extension is written as a catalog directory, without a
    collection of its own. A margin holds the same columns as the catalog it belongs to, and
    an index is carried over as long as it indexes a column that stays in the core.
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
    """what this extension provides, e.g. "spectra". The extension catalog is named after
    the input catalog, with this appended: ``small_sky`` gives ``small_sky_spectra``."""
    join_style: Literal["left", "inner"] = "left"
    """the type of join to use when combining the extension with the core"""
    product_type_served: str | None = None
    """modality of the data that the extension stores (e.g. "spectra", "images")"""

    ## Constructed
    copied_columns: list[str] = field(default_factory=list)
    """columns the extension always holds, which stay in the core as well"""
    core_columns: list[str] = field(default_factory=list)
    """columns of the input catalog that are written to the core"""
    extension_input_columns: list[str] = field(default_factory=list)
    """columns to read from an input table, to write the extension"""
    extension_output_columns: list[str] = field(default_factory=list)
    """names to write `extension_input_columns` under, in the same order"""
    core: SplitSide = None
    """the core side of the split"""
    extension: SplitSide = None
    """the extension side of the split"""
    margins: list[MarginCatalog] = field(default_factory=list)
    """the margins of the input collection, each split into both sides"""
    default_margin_name: str | None = None
    """name of the input collection's default margin, if it has one"""
    default_index_column: str | None = None
    """column of the input collection's default index, if it is carried over"""
    indexes: dict[str, UPath] = field(default_factory=dict)
    """the index catalogs to carry over, by the column each one indexes. An index over a column
    that moves to the extension is not carried over"""

    def _check_arguments(self):
        super()._check_arguments()
        self._read_input()

        if not self.extension_name:
            raise ValueError("extension_name is required")

        self._check_columns()
        self._prepare_output_paths()

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
            raise ValueError("Extensions can only be split from object or source catalogs")

    def _check_columns(self):
        """Check that the requested columns can be split out of the input catalog."""
        if not self.extension_columns:
            raise ValueError("extension_columns is required")
        if not self.primary_column:
            raise ValueError("primary_column is required")
        if not self.join_column:
            self.join_column = self.primary_column

        # Remove duplicates, preserving order
        self.extension_columns = list(dict.fromkeys(self.extension_columns))

        column_names = self.input_catalog.schema.names
        missing_columns = [col for col in self.extension_columns if col not in column_names]
        if missing_columns:
            raise ValueError(f"Some extension columns do not exist in the catalog: {missing_columns}")
        if self.primary_column not in column_names:
            raise ValueError(f"primary_column does not exist in the catalog: {self.primary_column}")

        catalog_info = self.input_catalog.catalog_info
        self.copied_columns = list(
            dict.fromkeys(
                [self.healpix_column, self.primary_column, catalog_info.ra_column, catalog_info.dec_column]
            )
        )
        self.extension_input_columns = self.copied_columns + self.extension_columns
        self.extension_output_columns = [
            self.join_column if col == self.primary_column else col for col in self.extension_input_columns
        ]

        listed_copies = [col for col in self.extension_columns if col in self.copied_columns]
        if listed_copies:
            raise ValueError(
                "Columns are automatically copied into the extension and cannot be listed in "
                f"extension_columns: {listed_copies}"
            )
        ## The join key is written under its new name, so nothing else may claim that name.
        renamed_over = [col for col in self.extension_input_columns if col != self.primary_column]
        if self.join_column in renamed_over:
            raise ValueError(f"join_column conflicts with another extension column: {self.join_column}")

        self.core_columns = [col for col in column_names if col not in self.extension_columns]

    def _prepare_output_paths(self):
        """Build the two sides of the split, inside the collection this run writes.

        The base class has already made ``catalog_path``, which is the collection's root. The
        core catalog keeps the input catalog's name, and the extension is named after it. An
        extension with margins to hold is a collection of its own, inside this one.
        """
        if not self.output_path:  # pragma: no cover (not reachable, but required for mypy)
            raise ValueError("output_path is required")
        output_path = file_io.get_upath(self.output_path)
        collection_path = output_path / self.output_artifact_name
        if self.input_collection is not None:
            self._plan_collection_members()

        core_name = self.input_catalog.catalog_info.catalog_name
        self.core = self._build_side(
            name=core_name,
            collection_path=collection_path,
            output_path=output_path,
            columns=(self.core_columns, self.core_columns),
        )
        extension_name = f"{core_name}_{self.extension_name}"
        self.extension = self._build_side(
            name=extension_name,
            ## Margins of the extension need a collection to hold them.
            collection_path=collection_path / extension_name if self.margins else collection_path,
            output_path=output_path,
            columns=(self.extension_input_columns, self.extension_output_columns),
            is_core=False,
        )

    def _build_side(self, name, collection_path, output_path, columns, is_core=True) -> SplitSide:
        """Build one side of the split, and make the directory its main catalog goes in."""
        catalog_path = collection_path / name
        file_io.make_directory(catalog_path, exist_ok=True)
        return SplitSide(
            name=name,
            collection_path=collection_path,
            catalog_path=catalog_path,
            input_columns=columns[0],
            output_columns=columns[1],
            input_catalog_name=self.input_catalog.catalog_info.catalog_name,
            output_path=output_path,
            is_core=is_core,
        )

    def _plan_collection_members(self):
        """Record the margins to split, and the index catalogs to carry over."""
        collection = self.input_collection
        self.default_margin_name = collection.default_margin
        self.default_index_column = collection.default_index_field
        for margin_name in collection.all_margins or []:
            margin_dir = CatalogCollection.resolve_inner_path(collection.collection_path, margin_name)
            self.margins.append(read_hats(margin_dir))
        for indexing_column, index_name in (collection.all_indexes or {}).items():
            if indexing_column not in self.core_columns:
                warnings.warn(
                    f"Index {index_name} is not carried over, as it indexes {indexing_column}, "
                    "which is moved to the extension"
                )
                continue
            self.indexes[indexing_column] = CatalogCollection.resolve_inner_path(
                collection.collection_path, index_name
            )
        if self.default_index_column not in self.indexes:
            self.default_index_column = None

    @property
    def sides(self) -> tuple[SplitSide, SplitSide]:
        """The two sides of the split, the core first."""
        return (self.core, self.extension)

    @property
    def input_tables(self) -> list:
        """The input tables to split: the main catalog, and every margin of a collection."""
        return [self.input_catalog] + self.margins

    @property
    def healpix_column(self) -> str:
        """Name of the spatial index column, kept in both the core and the extension."""
        return self.input_catalog.catalog_info.healpix_column or SPATIAL_INDEX_COLUMN
