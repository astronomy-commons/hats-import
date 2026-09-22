"""Properties of the tables that an extension split writes"""

from __future__ import annotations

from hats.catalog import CatalogType, CollectionProperties, MarginCatalog, TableProperties
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
    that the split changes: its name, the columns it holds, and what it points at.
    """
    catalog_info = input_catalog.catalog_info

    if isinstance(input_catalog, MarginCatalog):
        overrides = {
            "catalog_name": side.derived_name(catalog_info.catalog_name),
            "catalog_type": CatalogType.MARGIN,
            "primary_catalog": side.catalog_reference,
        }
    elif side.is_core:
        overrides = {"catalog_name": side.name}
    else:
        overrides = _extension_overrides(args)

    default_columns = None
    if catalog_info.default_columns:
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


def collection_properties(args: ExtensionArguments, side: SplitSide) -> CollectionProperties:
    """Properties of the collection one side writes.

    The core writes the collection this run produces, which lists the margins and indexes of
    the input, and the extension. An extension with margins writes a collection of its own,
    holding only those margins."""
    info = {"name": side.collection_path.name, "hats_primary_table_url": side.name}
    if args.margins:
        info["all_margins"] = [side.derived_name(margin.catalog_info.catalog_name) for margin in args.margins]
        if args.default_margin_name:
            info["default_margin"] = side.derived_name(args.default_margin_name)
    if side.is_core:
        info["all_extensions"] = [args.extension.name]
        if args.indexes:
            info["all_indexes"] = {column: index_dir.name for column, index_dir in args.indexes.items()}
            if args.default_index_column:
                info["default_index"] = args.default_index_column
    return CollectionProperties(**(info | args.extra_property_dict(side.collection_path)))


def _extension_overrides(args: ExtensionArguments) -> dict:
    """Properties that only the extension catalog has, carrying what a reader needs to join
    it to the core."""
    return {
        "catalog_name": args.extension.name,
        "catalog_type": CatalogType.EXTENSION,
        "primary_catalog": args.core.catalog_reference,
        "primary_column": args.primary_column,
        "join_catalog": args.extension.catalog_reference,
        "join_column": args.join_column,
        "extension_columns": args.extension_columns,
        "extension_join_style": args.join_style,
        "extension_product_type": args.product_type_served,
    }
