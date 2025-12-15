"""Import a set of non-hats files using dask for parallelization"""

import pickle
import sys
from collections import defaultdict

import cloudpickle
import hats.pixel_math.healpix_shim as hp
import nested_pandas as npd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from hats import pixel_math
from hats.io import file_io, paths
from hats.io.paths import PARTITION_PIXEL
from hats.pixel_math.healpix_pixel import HealpixPixel
from hats.pixel_math.sparse_histogram import HistogramAggregator, SparseHistogram
from hats.pixel_math.spatial_index import SPATIAL_INDEX_COLUMN, spatial_index_to_healpix
from upath import UPath

from hats_import.catalog.resume_plan import ResumePlan
from hats_import.pipeline_resume_plan import get_pixel_cache_directory, print_task_failure

# pylint: disable=too-many-locals,too-many-arguments


def _has_named_index(dataframe):
    """Heuristic to determine if a dataframe has some meaningful index.

    This will reject dataframes with no index name for a single index,
    or empty names for multi-index (e.g. [] or [None]).
    """
    if dataframe.index.name is not None:
        ## Single index with a given name.
        return True
    if len(dataframe.index.names) == 0 or all(name is None for name in dataframe.index.names):
        return False
    return True


def _iterate_input_file(
    input_file: UPath,
    pickled_reader_file: str,
    highest_order,
    ra_column,
    dec_column,
    use_healpix_29=False,
    read_columns=None,
):
    """Helper function to handle input file reading and healpix pixel calculation"""
    with open(pickled_reader_file, "rb") as pickle_file:
        file_reader = cloudpickle.load(pickle_file)
    if not file_reader:
        raise NotImplementedError("No file reader implemented")

    for chunk_number, data in enumerate(file_reader.read(input_file, read_columns=read_columns)):
        if use_healpix_29:
            if isinstance(data, pd.DataFrame) and data.index.name == SPATIAL_INDEX_COLUMN:
                mapped_pixels = spatial_index_to_healpix(data.index, target_order=highest_order)
            else:
                mapped_pixels = spatial_index_to_healpix(
                    data[SPATIAL_INDEX_COLUMN], target_order=highest_order
                )
        else:
            # Set up the pixel data
            if isinstance(data, pd.DataFrame):
                mapped_pixels = hp.radec2pix(
                    highest_order,
                    data[ra_column].to_numpy(copy=False, dtype=float),
                    data[dec_column].to_numpy(copy=False, dtype=float),
                )
            else:
                mapped_pixels = hp.radec2pix(
                    highest_order,
                    data[ra_column].to_numpy(),
                    data[dec_column].to_numpy(),
                )
        yield chunk_number, data, mapped_pixels


def map_to_pixels(
    input_file: UPath,
    pickled_reader_file: str,
    resume_path: UPath,
    mapping_key,
    highest_order,
    ra_column,
    dec_column,
    use_healpix_29=False,
    threshold_mode="row_count",
):
    """Map a file of input objects to their healpix pixels.

    Args:
        input_file (UPath): file to read for catalog data.
        file_reader (hats_import.catalog.file_readers.InputReader): instance of input
            reader that specifies arguments necessary for reading from the input file.
        resume_path (UPath): where to write resume partial results.
        mapping_key (str): unique counter for this input file, used
            when creating intermediate files
        highest_order (int): healpix order to use when mapping
        ra_column (str): where to find right ascension data in the dataframe
        dec_column (str): where to find declation in the dataframe
        threshold_mode (str): mode for thresholding, either "row_count" or "mem_size".

    Returns:
        one-dimensional numpy array of long integers where the value at each index corresponds
        to the number of objects found at the healpix pixel.
    Raises:
        ValueError: if the `ra_column` or `dec_column` cannot be found in the input file.
        FileNotFoundError: if the file does not exist, or is a directory
    """
    try:
        # Always generate the row-count histogram.
        row_count_histo = HistogramAggregator(highest_order)
        mem_size_histo = None
        if threshold_mode == "mem_size":
            mem_size_histo = HistogramAggregator(highest_order)

        # Determine which columns to read from the input file. If we're using
        # the bytewise/mem_size histogram, we need to read all columns to accurately
        # estimate memory usage.
        if threshold_mode == "mem_size":
            read_columns = None
        elif use_healpix_29:
            read_columns = [SPATIAL_INDEX_COLUMN]
        else:
            read_columns = [ra_column, dec_column]

        # Iterate through the input file in chunks, mapping pixels and updating histograms.
        for _, chunk_data, mapped_pixels in _iterate_input_file(
            input_file,
            pickled_reader_file,
            highest_order,
            ra_column,
            dec_column,
            use_healpix_29,
            read_columns,
        ):
            # Always add to row_count histogram.
            mapped_pixel, count_at_pixel = np.unique(mapped_pixels, return_counts=True)
            row_count_histo.add(SparseHistogram(mapped_pixel, count_at_pixel, highest_order))

            # If using bytewise/mem_size thresholding, also add to mem_size histogram.
            if threshold_mode == "mem_size":
                data_mem_sizes = _get_mem_size_of_chunk(chunk_data)
                pixel_mem_sizes: dict[int, int] = defaultdict(int)
                for pixel, mem_size in zip(mapped_pixels, data_mem_sizes, strict=True):
                    pixel_mem_sizes[pixel] += mem_size

                # Turn our dict into two lists, the keys and vals, sorted so the keys are increasing
                mapped_pixel_ids = np.array(list(pixel_mem_sizes.keys()), dtype=np.int64)
                mapped_pixel_mem_sizes = np.array(list(pixel_mem_sizes.values()), dtype=np.int64)

                if mem_size_histo is not None:
                    mem_size_histo.add(
                        SparseHistogram(mapped_pixel_ids, mapped_pixel_mem_sizes, highest_order)
                    )

        # Write row_count histogram to file.
        row_count_histo.to_sparse().to_file(
            ResumePlan.partial_histogram_file(tmp_path=resume_path, mapping_key=mapping_key)
        )
        # If using bytewise/mem_size thresholding, also write mem_size histogram to a separate file.
        if threshold_mode == "mem_size" and mem_size_histo is not None:
            mem_size_histo.to_sparse().to_file(
                ResumePlan.partial_histogram_file(
                    tmp_path=resume_path, mapping_key=f"{mapping_key}", which_histogram="mem_size"
                )
            )
    except Exception as exception:  # pylint: disable=broad-exception-caught
        print_task_failure(f"Failed MAPPING stage with file {input_file}", exception)
        raise exception


def _get_row_mem_size_data_frame(row):
    """Given a pandas dataframe row (as a tuple), return the memory size of that row.

    Args:
        row (tuple): the row from the dataframe

    Returns:
        int: the memory size of the row in bytes
    """
    total = 0

    # Add the memory overhead of the row object itself.
    total += sys.getsizeof(row)

    # Then add the size of each item in the row.
    for item in row:
        if isinstance(item, np.ndarray):
            total += item.nbytes + sys.getsizeof(item)  # object data + object overhead
        else:
            total += sys.getsizeof(item)
    return total


def _get_row_mem_size_pa_table(table, row_index):
    """Given a pyarrow table and a row index, return the memory size of that row.

    Args:
        table (pa.Table): the pyarrow table
        row_index (int): the index of the row to measure

    Returns:
        int: the memory size of the row in bytes
    """
    total = 0

    # Add the memory overhead of the row object itself.
    total += sys.getsizeof(row_index)

    # Then add the size of each item in the row.
    for column in table.itercolumns():
        item = column[row_index]
        if isinstance(item, np.ndarray):
            total += item.nbytes + sys.getsizeof(item)  # object data + object overhead
        else:
            total += sys.getsizeof(item.as_py())
    return total


def _get_mem_size_of_chunk(data):
    """Given a 2D array of data, return a list of memory sizes for each row in the chunk.

    Args:
        data (pd.DataFrame or pa.Table): the data chunk to measure

    Returns:
        list[int]: list of memory sizes for each row in the chunk
    """
    if isinstance(data, pd.DataFrame):
        mem_sizes = [_get_row_mem_size_data_frame(row) for row in data.itertuples(index=False, name=None)]
    elif isinstance(data, pa.Table):
        mem_sizes = [_get_row_mem_size_pa_table(data, i) for i in range(data.num_rows)]
    else:
        raise NotImplementedError(f"Unsupported data type {type(data)} for memory size calculation")
    return mem_sizes


def split_pixels(
    input_file: UPath,
    pickled_reader_file: str,
    splitting_key,
    highest_order,
    ra_column,
    dec_column,
    cache_shard_path: UPath,
    resume_path: UPath,
    alignment_file=None,
    use_healpix_29=False,
):
    """Map a file of input objects to their healpix pixels and split into shards.

    Args:
        input_file (UPath): file to read for catalog data.
        file_reader (hats_import.catalog.file_readers.InputReader): instance
            of input reader that specifies arguments necessary for reading from the input file.
        splitting_key (str): unique counter for this input file, used
            when creating intermediate files
        highest_order (int): healpix order to use when mapping
        ra_column (str): where to find right ascension data in the dataframe
        dec_column (str): where to find declation in the dataframe
        cache_shard_path (UPath): where to write intermediate parquet files.
        resume_path (UPath): where to write resume files.

    Raises:
        ValueError: if the `ra_column` or `dec_column` cannot be found in the input file.
        FileNotFoundError: if the file does not exist, or is a directory
    """
    try:
        with open(alignment_file, "rb") as pickle_file:
            alignment = pickle.load(pickle_file)
        for chunk_number, data, mapped_pixels in _iterate_input_file(
            input_file, pickled_reader_file, highest_order, ra_column, dec_column, use_healpix_29
        ):
            aligned_pixels = alignment[mapped_pixels]
            unique_pixels, unique_inverse = np.unique(aligned_pixels, return_inverse=True, axis=0)

            for unique_index, pixel_alignment_count in enumerate(unique_pixels):
                order = pixel_alignment_count[0]
                pixel = pixel_alignment_count[1]
                pixel_dir = get_pixel_cache_directory(cache_shard_path, HealpixPixel(order, pixel))
                file_io.make_directory(pixel_dir, exist_ok=True)
                output_file = file_io.append_paths_to_pointer(
                    pixel_dir, f"shard_{splitting_key}_{chunk_number}.parquet"
                )
                if isinstance(data, pd.DataFrame):
                    filtered_data = data.iloc[unique_inverse == unique_index]
                    if _has_named_index(filtered_data):
                        filtered_data = filtered_data.reset_index()
                    filtered_data = pa.Table.from_pandas(
                        npd.NestedFrame(filtered_data).to_pandas(), preserve_index=False
                    ).replace_schema_metadata()
                else:
                    filtered_data = data.filter(unique_inverse == unique_index)

                pq.write_table(filtered_data, output_file.path, filesystem=output_file.fs)
                del filtered_data

        ResumePlan.splitting_key_done(tmp_path=resume_path, splitting_key=splitting_key)
    except Exception as exception:  # pylint: disable=broad-exception-caught
        print_task_failure(f"Failed SPLITTING stage with file {input_file}", exception)
        raise exception


# pylint: disable=too-many-positional-arguments,too-many-statements
def reduce_pixel_shards(
    cache_shard_path,
    resume_path,
    reducing_key,
    destination_pixel_order,
    destination_pixel_number,
    destination_pixel_size,
    output_path,
    ra_column,
    dec_column,
    sort_columns: str = "",
    use_healpix_29=False,
    add_healpix_29=True,
    delete_input_files=True,
    use_schema_file="",
    write_table_kwargs=None,
    row_group_kwargs=None,
    npix_suffix=".parquet",
    npix_parquet_name=None,
):
    """Reduce sharded source pixels into destination pixels.

    In addition to combining multiple shards of data into a single
    parquet file, this method will (optionally) add the ``_healpix_29`` column.
    See ``hats.pixel_math.spatial_index`` for more in-depth discussion of this field.

    Args:
        cache_shard_path (UPath): where to read intermediate parquet files.
        resume_path (UPath): where to write resume files.
        reducing_key (str): unique string for this task, used for resume files.
        destination_pixel_order (int): order of the final catalog pixel
        destination_pixel_number (int): pixel number at the above order
        destination_pixel_size (int): expected number of rows to write
            for the catalog's final pixel
        output_path (UPath): where to write the final catalog pixel data
        ra_column (str): where to find right ascension data in the dataframe
        dec_column (str): where to find declination in the dataframe
        sort_columns (str): column for survey identifier, or other sortable column
        use_healpix_29 (bool): should we use a pre-existing _healpix_29 column
            for position information.
        add_healpix_29 (bool): should we add a _healpix_29 column to
            the resulting parquet file?
        delete_input_files (bool): should we delete the intermediate files
            used as input for this method.
        use_schema_file (str): use the parquet schema from the indicated
            parquet file.
        write_table_kwargs (dict): additional keyword arguments to use when
            writing files to parquet (e.g. compression schemes)
        row_group_kwargs (dict): additional keyword arguments to use in
            creation of rowgroups when writing files to parquet.
        npix_suffix (str): suffix for Npix files. Defaults to ".parquet".
        npix_parquet_name (str): name of the pixel parquet file to be used
            when npix_suffix=/. By default, it will be named after the pixel
            with a .parquet extension (e.g. 'Npix=10.parquet').

    Raises:
        ValueError: if the number of rows written doesn't equal provided
            `destination_pixel_size`
    """
    try:
        healpix_pixel = HealpixPixel(destination_pixel_order, destination_pixel_number)
        destination_file = paths.new_pixel_catalog_file(
            output_path, healpix_pixel, npix_suffix=npix_suffix, npix_parquet_name=npix_parquet_name
        )

        # If the destination file exists, and has the right number of rows, return early.
        # If there's anything wrong with the destination file, attempt to write it again.
        try:
            _check_destination_file(
                destination_file,
                destination_pixel_size,
                healpix_pixel,
                delete_input_files,
                cache_shard_path,
                resume_path,
                reducing_key,
            )

            return
        except:  # pylint: disable=bare-except
            pass

        schema = None
        if use_schema_file:
            schema = file_io.read_parquet_metadata(use_schema_file).schema.to_arrow_schema()

        healpix_pixel = HealpixPixel(destination_pixel_order, destination_pixel_number)
        pixel_dir = get_pixel_cache_directory(cache_shard_path, healpix_pixel)

        merged_table = pq.read_table(pixel_dir.path, filesystem=pixel_dir.fs, schema=schema)

        rows_written = len(merged_table)

        if rows_written != destination_pixel_size:
            raise ValueError(
                "Unexpected number of objects at pixel "
                f"({healpix_pixel})."
                f" Expected {destination_pixel_size}, wrote {rows_written}"
            )

        if add_healpix_29:
            merged_table = merged_table.add_column(
                0,
                SPATIAL_INDEX_COLUMN,
                [
                    pixel_math.compute_spatial_index(
                        merged_table[ra_column].to_numpy().astype(np.float64),
                        merged_table[dec_column].to_numpy().astype(np.float64),
                    )
                ],
            )

        if not write_table_kwargs:
            write_table_kwargs = {}
        if not row_group_kwargs:
            row_group_kwargs = {}

        sorting_columns = sort_columns.split(",") if sort_columns is not None else []
        if add_healpix_29 or use_healpix_29:
            # Sort by healpix_29 first and then by the other sorting columns to resolve unambiguity
            sorting_columns.insert(0, SPATIAL_INDEX_COLUMN)
        if sorting_columns:
            ordering = [(col_name, "ascending") for col_name in sorting_columns]
            merged_table = merged_table.sort_by(ordering)
            # For metadata purposes this needs to be of type pq.SortingColumn
            write_table_kwargs["sorting_columns"] = pq.SortingColumn.from_ordering(
                merged_table.schema, ordering
            )

        # Obtain the row groups for the target file
        rowgroup_tables = _split_to_row_groups(merged_table, row_group_kwargs, destination_pixel_order)

        with pq.ParquetWriter(
            destination_file.path,
            merged_table.schema,
            filesystem=destination_file.fs,
            **write_table_kwargs,
        ) as writer:
            for table in rowgroup_tables:
                writer.write_table(table)
        del merged_table, rowgroup_tables

        _check_destination_file(
            destination_file,
            destination_pixel_size,
            healpix_pixel,
            delete_input_files,
            cache_shard_path,
            resume_path,
            reducing_key,
        )
    except Exception as exception:  # pylint: disable=broad-exception-caught
        print_task_failure(
            f"Failed REDUCING stage for shard: {destination_pixel_order} {destination_pixel_number}",
            exception,
        )
        raise exception


def _check_destination_file(
    destination_file,
    destination_pixel_size,
    healpix_pixel,
    delete_input_files,
    cache_shard_path,
    resume_path,
    reducing_key,
):
    if not destination_file.exists():
        raise FileNotFoundError(f"Reduced file not found where expected ({destination_file})")

    rows_written = file_io.read_parquet_metadata(destination_file).num_rows

    if rows_written != destination_pixel_size:
        raise ValueError(
            "Unexpected number of objects in RESUMED pixel data "
            f"({healpix_pixel})."
            f" Expected {destination_pixel_size}, found {rows_written}"
        )
    if delete_input_files:
        pixel_dir = get_pixel_cache_directory(cache_shard_path, healpix_pixel)
        file_io.remove_directory(pixel_dir, ignore_errors=True)

    ResumePlan.reducing_key_done(tmp_path=resume_path, reducing_key=reducing_key)
    return True


def _split_to_row_groups(table, row_group_kwargs, pixel_order):
    """Split the pixel table into its row group chunks according to the specified splitting strategy."""
    if "num_rows" in row_group_kwargs:
        chunk_size = row_group_kwargs["num_rows"]
        return [table.slice(i, chunk_size) for i in range(0, len(table), chunk_size)]
    if "subtile_order_delta" in row_group_kwargs:
        split_tables = []
        parent_pixels = table[SPATIAL_INDEX_COLUMN].to_numpy()
        target_order = row_group_kwargs["subtile_order_delta"] + pixel_order
        child_pixs = spatial_index_to_healpix(parent_pixels, target_order=target_order)
        for child_pix in np.unique(child_pixs):
            indices = np.where(child_pixs == child_pix)[0]
            row_group = table.take(pa.array(indices))
            split_tables.append(row_group)
        return split_tables
    return [table]
