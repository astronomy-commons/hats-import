"""One side of an extension split: the core, or the extension"""

from __future__ import annotations

from dataclasses import dataclass

from hats.catalog import MarginCatalog
from upath import UPath


@dataclass
class SplitSide:
    """One side of the split: the core, or the extension. Holds where that side's tables go,
    which columns they hold, and which indexes go with them."""

    name: str
    """name of this side's main catalog"""
    collection_path: UPath
    """directory holding this side's tables: the collection this run writes for the core, and
    the extension's own collection inside it"""
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

    @property
    def catalog_reference(self) -> str:
        """Path of this side's main catalog, for its margins and indexes to point at."""
        return self.catalog_path.relative_to(self.output_path).as_posix()

    def derived_name(self, member_name: str) -> str:
        """Name of a margin or an index on this side, with the input catalog's name replaced.

        For an extension named ``small_sky_spectra``, the margin that is originally named
        ``small_sky_margin`` becomes ``small_sky_spectra_margin``.
        """
        prefix = f"{self.input_catalog_name}_"
        if member_name.startswith(prefix):
            return f"{self.name}_{member_name.removeprefix(prefix)}"
        return member_name

    def table_path(self, input_catalog) -> UPath:
        """Directory where this side's split of the input table is written to."""
        if isinstance(input_catalog, MarginCatalog):
            return self.collection_path / self.derived_name(input_catalog.catalog_info.catalog_name)
        return self.catalog_path
