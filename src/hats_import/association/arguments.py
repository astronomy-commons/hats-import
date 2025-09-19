from dataclasses import dataclass
from pathlib import Path

import hats
from hats.catalog import AssociationCatalog
from hats.io.file_io import get_upath
from typing_extensions import Self
from upath import UPath

from hats_import import ImportArguments


@dataclass
class AssociationArguments(ImportArguments):
    """Container class for holding arguments for partitioning association HATS catalogs."""

    catalog_type: str = "association"
    """level of catalog data, a mapping between catalogs"""
    allowed_catalog_types: tuple[str] = ("association",)
    """possible types of catalog to import with `AssociationArguments`"""
    should_write_skymap: bool = False
    """association catalogs do not contain skymap fits files"""

    @classmethod
    def reimport_from_hats(cls, path: str | Path | UPath, output_dir: str | Path | UPath, **kwargs) -> Self:
        """Generate the import arguments to reimport a HATS association with different parameters

        Args:
            path (str | Path | UPath): the path to the existing association catalog to reimport
            output_dir (str | Path | UPath): the path to output the reimported catalog to
            kwargs: any import arguments to update from the existing catalog. Can be any argument
                usually passed to :func:`AssociationArguments`

        Returns:
            A AssociationArguments object with the arguments from the existing association,
            and any updates from kwargs
        """
        catalog = hats.read_hats(get_upath(path))

        if not isinstance(catalog, AssociationCatalog):
            raise ValueError("The catalog must be of type `association`")

        addl_hats_properties = {
            "primary_catalog": catalog.catalog_info.primary_catalog,
            "primary_column": catalog.catalog_info.primary_column,
            "primary_column_association": catalog.catalog_info.primary_column_association,
            "join_catalog": catalog.catalog_info.join_catalog,
            "join_column": catalog.catalog_info.join_column,
            "join_column_association": catalog.catalog_info.join_column_association,
            "assn_max_separation": catalog.catalog_info.assn_max_separation,
            "contains_leaf_files": catalog.catalog_info.contains_leaf_files,
        }
        return super().reimport_from_hats(
            path, output_dir, addl_hats_properties=addl_hats_properties, **kwargs
        )
