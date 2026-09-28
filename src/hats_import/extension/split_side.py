"""One side of an extension split: the core, or the extension"""

from __future__ import annotations

from dataclasses import dataclass

from hats.catalog import MarginCatalog
from upath import UPath


@dataclass
class SplitSide:
    """One side of the split: the core, or the extension.

    Holds where that side's tables go, which columns they hold, and which indexes go with them,
    so that the pipeline can write both sides with the same code."""

    name: str
    """name of this side's main catalog"""
    collection_path: UPath
    """directory holding this side's tables. The collection this run writes, or, for an
    extension that has margins or indexes, its own collection inside it"""
    catalog_path: UPath
    """directory of this side's main catalog"""
    input_columns: list[str]
    """columns to read from an input table"""
    output_columns: list[str]
    """names to write those columns under, in the same order"""
    input_catalog_name: str
    """name of the input catalog, replaced in the names of derived margins and indexes"""
    output_path: UPath
    """directory holding the collection this run writes, which references are relative to"""
    indexes: dict[str, UPath]
    """index catalogs of the input that go with this side, by the column each one indexes"""
    writes_collection: bool
    """whether this side writes a ``collection.properties`` of its own. The core always writes
    the collection this run produces, and the extension only writes one when it has margins or
    indexes to hold"""

    @property
    def root_path(self) -> UPath:
        """Directory of this side as a whole: its collection when it writes one, or its main
        catalog otherwise."""
        return self.collection_path if self.writes_collection else self.catalog_path

    @property
    def catalog_reference(self) -> str:
        """Path of this side's main catalog, for its margins and indexes to point at.

        As elsewhere in hats-import, it is relative to the directory that holds the collection."""
        return str(self.catalog_path.relative_to(self.output_path))

    def derived_name(self, member_name: str) -> str:
        """Name of a margin or an index on this side, with the input catalog's name replaced.

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
        if isinstance(input_catalog, MarginCatalog):
            return self.collection_path / self.derived_name(input_catalog.catalog_info.catalog_name)
        return self.catalog_path
