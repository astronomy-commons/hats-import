"""Inner methods for SOAP"""

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from hats.catalog.association_catalog.partition_join_info import PartitionJoinInfo
from hats.io import file_io, paths
from hats.io.parquet_metadata import get_healpix_pixel_from_metadata
from hats.pixel_math.healpix_pixel import HealpixPixel
from hats.pixel_math.healpix_pixel_function import get_pixel_argsort

from hats_import.pipeline_resume_plan import get_pixel_cache_directory, print_task_failure
from hats_import.soap.arguments import SoapArguments
from hats_import.soap.resume_plan import SoapPlan


def _count_joins_for_object(source_data, source_pixel, object_pixel, soap_args):
    object_path = paths.pixel_catalog_file(soap_args.object_catalog_dir, object_pixel)
    object_data = file_io.read_parquet_file_to_pandas(
        object_path,
        columns=[soap_args.object_id_column],
        schema=soap_args.object_catalog.schema,
    ).set_index(soap_args.object_id_column)

    joined_data = source_data.merge(object_data, how="inner", left_index=True, right_index=True)

    rows_written = len(joined_data)
    if not soap_args.write_leaf_files or rows_written == 0:
        return rows_written

    # Prepare the dataframe columns
    prepared_data = pd.DataFrame(
        data={
            "object_id": joined_data.index.values,
            "source_id": (
                joined_data.index.values
                if soap_args.source_object_id_column == soap_args.source_id_column
                else joined_data[soap_args.source_id_column]
            ),
            "Norder": np.full(rows_written, fill_value=object_pixel.order, dtype=np.uint8),
            "Dir": np.full(rows_written, fill_value=object_pixel.dir, dtype=np.uint64),
            "Npix": np.full(rows_written, fill_value=object_pixel.pixel, dtype=np.uint64),
            "join_Norder": np.full(rows_written, fill_value=source_pixel.order, dtype=np.uint8),
            "join_Dir": np.full(rows_written, fill_value=source_pixel.dir, dtype=np.uint64),
            "join_Npix": np.full(rows_written, fill_value=source_pixel.pixel, dtype=np.uint64),
        },
    ).drop_duplicates()

    # Write to parquet file.
    pixel_dir = get_pixel_cache_directory(soap_args.tmp_path, object_pixel)
    file_io.make_directory(pixel_dir, exist_ok=True)
    output_file = file_io.append_paths_to_pointer(
        pixel_dir, f"source_{source_pixel.order}_{source_pixel.pixel}.parquet"
    )
    prepared_data.to_parquet(output_file, index=False)

    return rows_written


def _write_count_results(cache_path, source_healpix, results):
    """Build a nice dataframe with pretty columns and rows"""
    num_results = len(results)
    dataframe = pd.DataFrame(results, columns=["Norder", "Npix", "num_rows"])

    dataframe["Dir"] = [int(order / 10_000) * 10_000 if order >= 0 else -1 for order, _, _ in results]
    dataframe["join_Norder"] = np.full(num_results, fill_value=source_healpix.order, dtype=np.uint8)
    dataframe["join_Dir"] = np.full(num_results, fill_value=source_healpix.dir, dtype=np.uint64)
    dataframe["join_Npix"] = np.full(num_results, fill_value=source_healpix.pixel, dtype=np.uint64)

    ## Reorder columns.
    dataframe = dataframe[["Norder", "Dir", "Npix", "join_Norder", "join_Dir", "join_Npix", "num_rows"]]

    file_io.write_dataframe_to_csv(
        dataframe=dataframe,
        file_pointer=cache_path / f"{source_healpix.order}_{source_healpix.pixel}.csv",
        index=False,
    )


def count_joins(soap_args: SoapArguments, source_pixel: HealpixPixel, object_pixels: list[HealpixPixel]):
    """Count the number of equijoined sources in the object pixels.
    If any un-joined source pixels remain, stretch out to neighboring object pixels.

    Args:
        soap_args(`hats_import.soap.SoapArguments`): set of arguments for pipeline execution
        source_pixel(HealpixPixel): order and pixel for the source catalog single pixel.
        object_pixels(List[HealpixPixel]): set of tuples of order and pixel for the partitions
            of the object catalog to be joined.
    """
    try:
        source_path = paths.pixel_catalog_file(soap_args.source_catalog_dir, source_pixel)
        if soap_args.write_leaf_files and soap_args.source_object_id_column != soap_args.source_id_column:
            read_columns = [soap_args.source_object_id_column, soap_args.source_id_column]
        else:
            read_columns = [soap_args.source_object_id_column]
        source_data = file_io.read_parquet_file_to_pandas(
            source_path,
            columns=read_columns,
            schema=soap_args.source_catalog.schema,
        ).set_index(soap_args.source_object_id_column)

        remaining_sources = len(source_data)
        results = []

        for object_pixel in object_pixels:
            if remaining_sources < 1:
                break
            join_count = _count_joins_for_object(
                source_data,
                source_pixel,
                object_pixel,
                soap_args,
            )
            results.append([object_pixel.order, object_pixel.pixel, join_count])
            remaining_sources -= join_count

        ## mark that some sources were not joined
        if remaining_sources > 0:
            results.append([-1, -1, remaining_sources])

        _write_count_results(soap_args.tmp_path, source_pixel, results)
    except Exception as exception:  # pylint: disable=broad-exception-caught
        print_task_failure(f"Failed COUNTING stage for shard: {source_pixel}", exception)
        raise exception


def combine_partial_results(input_path, output_path) -> int:
    """Combine many partial CSVs into single partition join info.
    Also write out a debug file with counts of unmatched sources, if any.

    Args:
        input_path(str): intermediate directory with partial result CSVs. likely, the
            directory used in the previous `count_joins` call as `cache_path`
        output_path(str): directory to write the combined results CSVs.

    Returns:
        integer that is the sum of all matched num_rows.
    """
    partial_files = file_io.find_files_matching_path(input_path, "*.csv")
    partials = []

    for partial_file in partial_files:
        partials.append(file_io.load_csv_to_pandas(partial_file))

    dataframe = pd.concat(partials)

    matched = dataframe.loc[dataframe["Norder"] != -1]
    matched = matched.loc[matched["num_rows"] > 0]
    unmatched = dataframe.loc[dataframe["Norder"] == -1]

    file_io.write_dataframe_to_csv(
        dataframe=matched, file_pointer=output_path / "partition_join_info.csv", index=False
    )

    if len(unmatched) > 0:
        file_io.write_dataframe_to_csv(
            dataframe=unmatched, file_pointer=output_path / "unmatched_sources.csv", index=False
        )

    primary_only = matched.groupby(["Norder", "Dir", "Npix"])["num_rows"].sum().reset_index()
    file_io.write_dataframe_to_csv(
        dataframe=primary_only, file_pointer=output_path / "partition_info.csv", index=False
    )

    join_info = PartitionJoinInfo(matched)
    join_info.write_to_metadata_files(output_path)

    return primary_only["num_rows"].sum()


def reduce_joins(
    soap_args: SoapArguments, object_pixel: HealpixPixel, object_key: str, delete_input_files: bool = True
):
    """Reduce join tables into one parquet file per object-pixel, with one row-group
    inside per source pixel."""
    try:
        pixel_dir = get_pixel_cache_directory(soap_args.tmp_path, object_pixel)
        # If there's no directory, this implies there were no matches to this object pixel
        # earlier in the pipeline. Move on.
        if not file_io.does_file_or_directory_exist(pixel_dir):
            return
        # Find all of the constituent files / source pixels. Create a list of PyArrow Tables from those
        # parquet files. We need to know the schema before we create the ParquetWriter.
        shard_file_list = file_io.find_files_matching_path(pixel_dir, "source*.parquet")

        if len(shard_file_list) == 0:
            return

        ## We want to order the row groups in a "breadth-first" sorting. Determine our sorting
        ## via the metadata, then read the tables in using that sorting.
        healpix_pixels = []
        for shard_file_name in shard_file_list:
            healpix_pixels.append(
                get_healpix_pixel_from_metadata(pq.read_metadata(shard_file_name), "join_Norder", "join_Npix")
            )

        argsort = get_pixel_argsort(healpix_pixels)
        shard_file_list = np.array(shard_file_list)[argsort]

        shards = []
        for shard_file_name in shard_file_list:
            shards.append(pq.read_table(shard_file_name))

        # Write all of the shards into a single parquet file, one row-group-per-shard.
        destination_dir = paths.pixel_directory(
            soap_args.catalog_path, object_pixel.order, object_pixel.pixel
        )
        file_io.make_directory(destination_dir, exist_ok=True)

        output_file = paths.pixel_catalog_file(soap_args.catalog_path, object_pixel)
        with pq.ParquetWriter(output_file.path, shards[0].schema, filesystem=output_file.fs) as writer:
            for table in shards:
                writer.write_table(table)

        # Delete the intermediate shards.
        if delete_input_files:
            file_io.remove_directory(pixel_dir, ignore_errors=True)

        SoapPlan.reducing_key_done(tmp_path=soap_args.tmp_path, reducing_key=object_key)
    except Exception as exception:  # pylint: disable=broad-exception-caught
        print_task_failure(f"Failed REDUCING stage for shard: {object_pixel}", exception)
        raise exception
