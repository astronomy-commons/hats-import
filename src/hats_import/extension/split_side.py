"""One side of an extension split: the core, or the extension"""

from __future__ import annotations

from dataclasses import dataclass

from hats.catalog import CatalogType
from upath import UPath


@dataclass
class SplitSide:
    """One side of the split: the core, or the extension.

    Holds where that side's tables go, and which columns they hold, so that the pipeline can
    write both sides with the same code."""

    name: str
    """name of this side's main catalog"""
    collection_path: UPath
    """directory holding this side's tables. The collection this run writes, or, for an
    extension that has margins, its own collection inside it"""
    catalog_path: UPath
    """directory of this side's main catalog"""
    input_columns: list[str]
    """columns to read from an input table"""
    output_columns: list[str]
    """names to write those columns under, in the same order"""
    input_catalog_name: str
    """name of the input catalog, replaced in the names of derived margins"""
    output_path: UPath
    """directory holding the collection this run writes, which references are relative to"""
    is_core: bool = True
    """whether this side is the core, rather than the extension"""

    @property
    def writes_collection(self) -> bool:
        """Whether this side writes a ``collection.properties`` of its own.

        The core writes the collection this run produces. The extension writes one only when
        it has margins to hold."""
        return self.is_core or self.collection_path != self.output_path / self.top_level_name

    @property
    def top_level_name(self) -> str:
        """Name of the collection this run writes."""
        return self.collection_path.relative_to(self.output_path).parts[0]

    @property
    def catalog_reference(self) -> str:
        """Path of this side's main catalog, for another table to point at."""
        return self.reference(self.catalog_path)

    def reference(self, path: UPath) -> str:
        """Path of a table that this run writes, for another table to point at.

        Like the rest of HATS, a reference is relative to the directory that holds the
        collection, so a pair of tables keeps pointing at each other wherever it is moved."""
        return str(path.relative_to(self.output_path))

    def derived_name(self, member_name: str) -> str:
        """Name of a margin on this side, with the input catalog's name replaced.

        ``small_sky_margin`` becomes ``small_sky_spectra_margin`` for an extension named
        ``small_sky_spectra``, and keeps its name on the core, which is named after the input
        catalog. A name that is not prefixed with the input catalog's name is used as it is.
        """
        prefix = f"{self.input_catalog_name}_"
        if member_name.startswith(prefix):
            return f"{self.name}_{member_name.removeprefix(prefix)}"
        return member_name

    def table_path(self, input_catalog) -> UPath:
        """Directory this side writes its copy of the given input table to."""
        if input_catalog.catalog_info.catalog_type == CatalogType.MARGIN:
            return self.collection_path / self.derived_name(input_catalog.catalog_info.catalog_name)
        return self.catalog_path
