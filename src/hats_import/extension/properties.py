"""Properties of the tables that an extension split writes"""

from __future__ import annotations

from hats.catalog import CatalogType, CollectionProperties, TableProperties
from hats.pixel_math.spatial_index import SPATIAL_INDEX_ORDER
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

    A margin, on either side, and the core catalog keep the properties of the table they were
    split from. The extension catalog is described from scratch, as it is of another type.
    """
    catalog_info = input_catalog.catalog_info
    if catalog_info.catalog_type == CatalogType.MARGIN:
        return _inherited_properties(
            args,
            side,
            catalog_info,
            catalog_path,
            total_rows,
            skymap_order,
            catalog_name=side.derived_name(catalog_info.catalog_name),
            catalog_type=CatalogType.MARGIN,
            primary_catalog=side.catalog_reference,
        )
    if side.is_core:
        return _inherited_properties(
            args, side, catalog_info, catalog_path, total_rows, skymap_order, catalog_name=side.name
        )
    return _extension_properties(args, catalog_info, total_rows, skymap_order)


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


def _inherited_properties(
    args: ExtensionArguments,
    side: SplitSide,
    input_catalog_info: TableProperties,
    catalog_path: UPath,
    total_rows: int,
    skymap_order: int | None,
    **overrides,
) -> TableProperties:
    """Properties of the input table, with only the fields that the split changes replaced."""
    default_columns = None
    if input_catalog_info.default_columns:
        default_columns = [
            col for col in input_catalog_info.default_columns if col.split(".")[0] in side.output_columns
        ] or None
    info = (
        input_catalog_info.explicit_dict()
        | input_catalog_info.extra_dict()
        | args.extra_property_dict(catalog_path)
        | {
            "total_rows": total_rows,
            "default_columns": default_columns,
            "skymap_order": skymap_order,
            "skymap_alt_orders": args.skymap_alt_orders if skymap_order is not None else None,
        }
        | overrides
    )
    return TableProperties(**info)


def _extension_properties(
    args: ExtensionArguments, input_catalog_info: TableProperties, total_rows: int, skymap_order: int | None
) -> TableProperties:
    """Properties of the extension catalog, carrying what a reader needs to join it to the core."""
    info = {
        "catalog_name": args.extension.name,
        "catalog_type": CatalogType.EXTENSION,
        "total_rows": total_rows,
        "ra_column": input_catalog_info.ra_column,
        "dec_column": input_catalog_info.dec_column,
        "healpix_column": args.healpix_column,
        "healpix_order": input_catalog_info.healpix_order or SPATIAL_INDEX_ORDER,
        "npix_suffix": args.npix_suffix,
        "primary_catalog": args.core.catalog_reference,
        "primary_column": args.primary_column,
        "join_catalog": args.extension.catalog_reference,
        "join_column": args.join_column,
        "extension_columns": args.extension_columns,
        "extension_join_style": args.join_style,
        "extension_product_type": args.product_type_served,
        "skymap_order": skymap_order,
        "skymap_alt_orders": args.skymap_alt_orders if skymap_order is not None else None,
        "hats_order": getattr(input_catalog_info, "hats_order", None),
        "moc_sky_fraction": input_catalog_info.moc_sky_fraction,
        "hats_max_rows": input_catalog_info.hats_max_rows,
    }
    info = {key: value for key, value in info.items() if value is not None}
    return TableProperties(**(info | args.extra_property_dict(args.extension.catalog_path)))
