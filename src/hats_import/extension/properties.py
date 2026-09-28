"""Properties of the tables, collections and extension file that an extension split writes"""

from __future__ import annotations

from hats.catalog import (
    CatalogType,
    CollectionProperties,
    ExtensionProperties,
    MarginCatalog,
    TableProperties,
)
from upath import UPath

from hats_import.extension.arguments import ExtensionArguments
from hats_import.extension.split_side import SplitSide


def table_properties(
    args: ExtensionArguments,
    side: SplitSide,
    input_catalog,
    catalog_path: UPath,
    total_rows: int,
    skymap_order: int | None,
) -> TableProperties:
    """Properties of one side's copy of an input table.

    Every table keeps the properties of the table it was split from, and replaces the fields
    that the split changes: its name, its default columns, and what it points at.
    """
    catalog_info = input_catalog.catalog_info

    if isinstance(input_catalog, MarginCatalog):
        overrides = {
            "catalog_name": side.derived_name(catalog_info.catalog_name),
            "catalog_type": CatalogType.MARGIN,
            "primary_catalog": side.catalog_reference,
        }
    else:
        overrides = {"catalog_name": side.name}

    # The core keeps the default columns for the columns it holds.
    # The extension carries no default columns.
    default_columns = None
    if side is args.core and catalog_info.default_columns:
        default_columns = [
            col for col in catalog_info.default_columns if col.split(".")[0] in side.output_columns
        ] or None
    info = (
        catalog_info.explicit_dict()
        | catalog_info.extra_dict()
        | args.extra_property_dict(catalog_path)
        | {
            "total_rows": total_rows,
            "default_columns": default_columns,
            "npix_suffix": args.npix_suffix,
            "skymap_order": skymap_order,
            "skymap_alt_orders": args.skymap_alt_orders if skymap_order is not None else None,
        }
        | overrides
    )
    return TableProperties(**info)


def collection_properties(args: ExtensionArguments) -> CollectionProperties:
    """Properties of the collection which holds the core. It lists the core's
    margins and indexes, and the new extension."""
    return CollectionProperties(
        **(_collection_info(args, args.core) | {"all_extensions": [args.extension.name]})
    )


def extension_collection_properties(args: ExtensionArguments) -> CollectionProperties:
    """Properties of the extension's own collection, which holds its margins and indexes."""
    return CollectionProperties(**_collection_info(args, args.extension))


def _collection_info(args: ExtensionArguments, side: SplitSide) -> dict:
    """Properties that every collection this run writes has: its catalog, its margins, and its
    indexes. The input's default index stays the default in the collection its index goes to."""
    info: dict = {"name": side.collection_path.name, "hats_primary_table_url": side.name}
    if args.margins:
        info["all_margins"] = [side.derived_name(margin.catalog_info.catalog_name) for margin in args.margins]
        if args.default_margin_name:
            info["default_margin"] = side.derived_name(args.default_margin_name)
    if side.indexes:
        info["all_indexes"] = {
            column: side.derived_name(index_dir.name) for column, index_dir in side.indexes.items()
        }
        if args.default_index_column in side.indexes:
            info["default_index"] = args.default_index_column
    return info | args.extra_property_dict(side.collection_path)


def extension_properties(args: ExtensionArguments) -> ExtensionProperties:
    """Properties of the extension, written to ``<extension>.properties`` at the root of the
    collection."""
    extension_path = args.extension.root_path
    info = {
        "name": args.extension.name,
        "primary_catalog": args.output_artifact_name,
        "primary_column": args.primary_column,
        "join_catalog": extension_path.relative_to(args.core.collection_path).as_posix(),
        "join_column": args.join_column,
        "extension_columns": args.extension_columns,
        "extension_join_style": args.join_style,
        "extension_product_type": args.product_type_served,
    }
    return ExtensionProperties(**(info | args.extra_property_dict(extension_path)))
