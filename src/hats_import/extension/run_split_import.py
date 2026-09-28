"""Split an input HATS Catalog into a "core" Catalog and its corresponding Extension"""

import shutil

import hats.pixel_math.healpix_shim as hp
import pyarrow.parquet as pq
from dask.distributed import as_completed
from hats.catalog import PartitionInfo, TableProperties
from hats.io import file_io, paths
from hats.io.parquet_metadata import write_parquet_metadata
from hats.io.skymap import write_skymap
from hats.io.summary_file import write_catalog_summary_file, write_partition_info_png, write_skymap_png
from hats.io.validation import is_valid_collection
from hats.pixel_math.healpix_pixel import HealpixPixel
from upath import UPath

from hats_import.catalog.map_reduce import _split_to_row_groups
from hats_import.extension.arguments import ExtensionArguments
from hats_import.extension.properties import (
    collection_properties,
    extension_collection_properties,
    extension_properties,
    table_properties,
)
from hats_import.extension.split_side import SplitSide
from hats_import.pipeline_resume_plan import print_progress, print_task_failure


def run(args: ExtensionArguments, client):
    """Run the pipeline that splits a catalog into a core catalog and an extension.

    The output is always a collection, holding the core, the extension (a collection of its own
    when it has margins or indexes) and the extension's properties file. Every margin of the input
    is split alongside the main catalog, and every index goes with the column it indexes."""
    if not args:
        raise TypeError("args is required and should be type ExtensionArguments")
    if not isinstance(args, ExtensionArguments):
        raise TypeError("args must be type ExtensionArguments")

    for side in args.sides:
        file_io.make_directory(side.catalog_path, exist_ok=True)

    futures = []
    for input_catalog in args.input_tables:
        for pixel in input_catalog.get_healpix_pixels():
            futures.append(client.submit(split_pixel, pixel=pixel, args=args, input_catalog=input_catalog))

    for future in print_progress(
        as_completed(futures),
        stage_name="Splitting",
        total=len(futures),
        use_progress_bar=args.progress_bar,
        simple_progress_bar=args.simple_progress_bar,
        tqdm_kwargs=args.tqdm_kwargs,
    ):
        if future.status == "error":
            raise future.exception()

    total_steps = 2 * len(args.input_tables) + sum(len(side.indexes) for side in args.sides) + 4

    with print_progress(
        total=total_steps,
        stage_name="Finishing",
        use_progress_bar=args.progress_bar,
        simple_progress_bar=args.simple_progress_bar,
        tqdm_kwargs=args.tqdm_kwargs,
    ) as step_progress:
        for input_catalog in args.input_tables:
            point_map = _read_point_map(args, input_catalog)
            for side in args.sides:
                _write_table_metadata(args, side, input_catalog, point_map)
                step_progress.update(1)
        for side in args.sides:
            for index_dir in side.indexes.values():
                _copy_index_catalog(args, side, index_dir)
                step_progress.update(1)
        extension_properties(args).to_properties_file(args.core.collection_path)
        step_progress.update(1)
        collection_properties(args).to_properties_file(args.core.collection_path)
        step_progress.update(1)
        if args.extension.writes_collection:
            extension_collection_properties(args).to_properties_file(args.extension.collection_path)
        step_progress.update(1)
        for side in args.sides:
            if side.writes_collection:
                assert is_valid_collection(side.collection_path)
        if args.tmp_path:  # pragma: no cover (always set, but required for mypy)
            file_io.remove_directory(args.tmp_path, ignore_errors=True)
        step_progress.update(1)


def split_pixel(pixel: HealpixPixel, args: ExtensionArguments, input_catalog):
    """Split the data of a single input partition into its core and extension files.
    The row group structure of the input file is preserved, unless `row_group_kwargs`
    were provided, in which case each output file is re-split accordingly."""
    try:
        input_file = paths.pixel_catalog_file(
            input_catalog.catalog_path, pixel, npix_suffix=input_catalog.catalog_info.npix_suffix
        )
        table = pq.read_table(input_file.path, filesystem=input_file.fs).replace_schema_metadata()
        for side in args.sides:
            destination_file = paths.new_pixel_catalog_file(
                side.table_path(input_catalog),
                pixel,
                npix_suffix=args.npix_suffix,
                npix_parquet_name=args.npix_parquet_name,
            )
            side_table = table.select(side.input_columns).rename_columns(side.output_columns)
            _write_table(side_table, destination_file, pixel, args)
    except Exception as exception:  # pylint: disable=broad-exception-caught
        print_task_failure(f"Failed SPLITTING stage for pixel: {pixel}", exception)
        raise exception


def _write_table(table, destination_file, pixel, args):
    """Write one output file, as one row group per chunk of the table."""
    if args.row_group_kwargs:
        rowgroups = _split_to_row_groups(table, args.row_group_kwargs, pixel.order)
    else:
        rowgroups = table.to_batches()
    with pq.ParquetWriter(
        destination_file.path,
        table.schema,
        filesystem=destination_file.fs,
        **args.write_table_kwargs,
    ) as writer:
        for rowgroup in rowgroups:
            writer.write(rowgroup)


def _read_point_map(args: ExtensionArguments, input_catalog):
    """Read an input table's point map, which its split copies keep as it is."""
    point_map_file = paths.get_point_map_file_pointer(input_catalog.catalog_path)
    if not args.should_write_skymap or not point_map_file.exists():
        return None
    return file_io.read_fits_image(point_map_file)


def _write_table_metadata(args: ExtensionArguments, side: SplitSide, input_catalog, point_map):
    """Write the metadata of one side's copy of an input table. That is the partition info,
    the parquet metadata, the skymaps and the properties, plus any optional summary files."""
    catalog_path = side.table_path(input_catalog)
    pixels = input_catalog.get_healpix_pixels()

    # An empty table has no data files, so nothing has created its directory yet.
    file_io.make_directory(catalog_path, exist_ok=True)
    PartitionInfo.from_healpix(pixels).write_to_file(paths.get_partition_info_pointer(catalog_path))
    total_rows, skymap_order = _write_parquet_metadata(args, side, input_catalog, catalog_path, point_map)

    properties = table_properties(args, side, input_catalog, catalog_path, total_rows, skymap_order)
    properties.to_properties_file(catalog_path)

    _write_summary_files(args, catalog_path)


def _write_parquet_metadata(
    args: ExtensionArguments, side: SplitSide, input_catalog, catalog_path, point_map
):
    """Write the parquet metadata and skymaps of one output table.

    Returns the number of rows written, and the order of the skymap, if one was written."""
    catalog_info = input_catalog.catalog_info
    if not input_catalog.get_healpix_pixels():
        _write_empty_metadata(side, input_catalog, catalog_path)
        return 0, None
    total_rows = write_parquet_metadata(
        catalog_path,
        create_thumbnail=args.create_thumbnail,
        thumbnail_threshold=catalog_info.hats_max_rows or 1_000_000,
        create_metadata=args.create_metadata,
        create_per_partition_stats=args.create_per_partition_stats,
    )
    if catalog_info.total_rows is not None and total_rows != catalog_info.total_rows:
        raise ValueError(
            f"Number of rows in parquet ({total_rows}) does not match "
            f"input table {catalog_info.catalog_name} ({catalog_info.total_rows})"
        )
    if point_map is None:
        return total_rows, None
    file_io.write_fits_image(point_map, paths.get_point_map_file_pointer(catalog_path))
    write_skymap(histogram=point_map, catalog_dir=catalog_path, orders=args.skymap_alt_orders)
    return total_rows, hp.npix2order(len(point_map))


def _write_empty_metadata(side: SplitSide, input_catalog, catalog_path: UPath):
    """Write the parquet metadata of an output table that has no partitions."""
    schema = (
        input_catalog.schema.empty_table()
        .select(side.input_columns)
        .rename_columns(side.output_columns)
        .schema
    )
    common_metadata_path = paths.get_common_metadata_pointer(catalog_path)
    file_io.make_directory(common_metadata_path.parent, exist_ok=True)
    pq.write_metadata(schema, common_metadata_path.path, filesystem=common_metadata_path.fs)


def _write_summary_files(args: ExtensionArguments, catalog_path: UPath):
    """Write the optional visual and summary files of one output table."""
    if args.create_skymap_png:
        write_skymap_png(catalog_path)
    if args.create_partition_info_png:
        write_partition_info_png(catalog_path)
    if args.create_summary_html:
        write_catalog_summary_file(catalog_path, fmt="html")
    if args.create_summary_md:
        write_catalog_summary_file(catalog_path, fmt="markdown")


def _copy_index_catalog(args: ExtensionArguments, side: SplitSide, index_dir: UPath):
    """Copy an index catalog of the input collection into the collection of the side that holds
    its column. An index that moves to the extension is renamed after it, like its margins."""
    index_name = side.derived_name(index_dir.name)
    catalog_path = side.collection_path / index_name
    for source_file in index_dir.rglob("*"):
        if source_file.is_dir():
            continue
        destination_file = catalog_path / source_file.relative_to(index_dir)
        destination_file.parent.mkdir(parents=True, exist_ok=True)
        with source_file.open("rb") as source, destination_file.open("wb") as destination:
            shutil.copyfileobj(source, destination)
    catalog_info = TableProperties.read_from_dir(catalog_path).copy_and_update(
        catalog_name=index_name,
        primary_catalog=side.catalog_reference,
        **args.extra_property_dict(catalog_path),
    )
    catalog_info.to_properties_file(catalog_path)
