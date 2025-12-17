"""Data class to hold common runtime arguments for dataset creation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import pandas as pd
from hats.catalog import TableProperties
from hats.io import file_io
from hats.io.validation import is_valid_catalog
from hats.pixel_math import spatial_index
from upath import UPath

# pylint: disable=too-many-instance-attributes


@dataclass
class RuntimeArguments:
    """Data class for holding runtime arguments"""

    ## Output
    output_path: str | Path | UPath | None = None
    """base path where new catalog should be output"""
    output_artifact_name: str = ""
    """short, convenient name for the catalog"""
    addl_hats_properties: dict | None = None
    """Any additional keyword arguments you would like to provide when writing
    the `hats.properties` file for the final HATS table. e.g. 
    {"hats_cols_default":"id, mjd", "hats_cols_survey_id":"unique_id", 
    "creator_did": "ivo://CDS/P/2MASS/J"}"""
    npix_suffix: str = ".parquet"
    """Suffix for pixel data. When specified as "/" each pixel will have a directory in its name."""
    npix_parquet_name: str | None = None
    """Name of the pixel parquet file to be used when npix_suffix=/. By default, it will be named
    after the pixel with a .parquet extension (e.g. 'Npix=10.parquet')"""
    write_table_kwargs: dict | None = None
    """additional keyword arguments to use when writing files to parquet (e.g. compression schemes)."""
    row_group_kwargs: dict | None = None
    """additional keyword arguments to use in creation of rowgroups when writing files to parquet."""
    should_write_skymap: bool = True
    """main catalogs should contain skymap fits files"""
    skymap_alt_orders: list[int] | None = None
    """Additional alternative healpix orders to write a HEALPix skymap."""
    create_thumbnail: bool = False
    """Create /dataset/data_thumbnail.parquet from one row of each data partition."""
    create_metadata: bool = True
    """Create /dataset/_metadata parquet from all data partitions."""

    ## Execution
    tmp_dir: str | Path | UPath | None = None
    """path for storing intermediate files"""
    resume: bool = True
    """If True, we try to read any existing intermediate files and continue to run
    the pipeline where we left off. If False, we start the import from scratch,
    overwriting any content of the output directory."""
    progress_bar: bool = True
    """if true, a progress bar will be displayed for user
    feedback of map reduce progress"""
    simple_progress_bar: bool = False
    """if displaying a progress bar, use a text-only simple progress
    bar instead of widget. this can be useful in some environments when running
    in a notebook where ipywidgets cannot be used (see `progress_bar` argument)"""
    tqdm_kwargs: dict | None = None
    """Additional arguments to pass to the tqdm progress bar. 
    See details https://tqdm.github.io/docs/tqdm/"""
    dask_tmp: str | Path | UPath | None = None
    """directory for dask worker space. this should be local to
    the execution of the pipeline, for speed of reads and writes"""
    dask_n_workers: int = 1
    """number of workers for the dask client"""
    dask_threads_per_worker: int = 1
    """number of threads per dask worker"""
    resume_tmp: str | Path | UPath | None = None
    """directory for intermediate resume files, when needed. see RTD for more info."""
    delete_intermediate_parquet_files: bool = True
    """should we delete the smaller intermediate parquet files generated in the
    splitting stage, once the relevant reducing stage is complete?"""
    delete_resume_log_files: bool = True
    """should we delete task-level done files once each stage is complete?
    if False, we will keep all done marker files at the end of the pipeline."""

    completion_email_address: str = ""
    """if provided, send an email to the indicated email address once the 
    import pipeline has completed."""

    catalog_path: UPath | None = None
    """constructed output path for the catalog that will be something like
    <output_path>/<output_artifact_name>"""
    tmp_path: UPath | None = None
    """constructed temp path - defaults to tmp_dir, then dask_tmp, but will create
    a new temp directory under catalog_path if no other options are provided"""
    tmp_base_path: UPath | None = None
    """temporary base directory: either `tmp_dir` or `dask_dir`, if those were provided by the user"""

    def __post_init__(self):
        self._check_arguments()

    def _check_arguments(self):
        if not self.output_path:
            raise ValueError("output_path is required")
        if not self.output_artifact_name:
            raise ValueError("output_artifact_name is required")
        if re.search(r"[^A-Za-z0-9\._\-\\]", self.output_artifact_name):
            raise ValueError("output_artifact_name contains invalid characters")

        if self.dask_n_workers <= 0:
            raise ValueError("dask_n_workers should be greather than 0")
        if self.dask_threads_per_worker <= 0:
            raise ValueError("dask_threads_per_worker should be greater than 0")

        self.catalog_path = file_io.get_upath(self.output_path) / self.output_artifact_name
        if is_valid_catalog(self.catalog_path):
            raise ValueError(f"Output path {self.catalog_path} already contains a valid catalog")
        if not self.resume:
            file_io.remove_directory(self.catalog_path, ignore_errors=True)
        file_io.make_directory(self.catalog_path, exist_ok=True)

        if self.tmp_dir and str(self.tmp_dir) != str(self.output_path):
            self.tmp_path = file_io.append_paths_to_pointer(
                self.tmp_dir, self.output_artifact_name, "intermediate"
            )
            self.tmp_base_path = file_io.append_paths_to_pointer(self.tmp_dir, self.output_artifact_name)
        elif self.dask_tmp and str(self.dask_tmp) != str(self.output_path):
            self.tmp_path = file_io.append_paths_to_pointer(
                self.dask_tmp, self.output_artifact_name, "intermediate"
            )
            self.tmp_base_path = file_io.append_paths_to_pointer(self.dask_tmp, self.output_artifact_name)
        else:
            self.tmp_path = file_io.append_paths_to_pointer(self.catalog_path, "intermediate")
        if not self.resume:
            file_io.remove_directory(self.tmp_path, ignore_errors=True)
        file_io.make_directory(self.tmp_path, exist_ok=True)
        if self.resume_tmp:
            self.resume_tmp = file_io.append_paths_to_pointer(self.resume_tmp, self.output_artifact_name)
        else:
            self.resume_tmp = self.tmp_path

        if self.write_table_kwargs is None:
            self.write_table_kwargs = {}
        if "compression" not in self.write_table_kwargs:
            self.write_table_kwargs = self.write_table_kwargs | {
                "compression": "ZSTD",
                "compression_level": 15,
            }

    def extra_property_dict(self):
        """Generate additional HATS properties for this import run as a dictionary."""
        properties = {}
        if self.addl_hats_properties:
            properties = self.addl_hats_properties
        return TableProperties.new_provenance_dict(
            self.catalog_path, builder=f"hats-import v{version('hats-import')}", **properties
        )

    def resume_kwargs_dict(self):
        """Convenience method to convert fields for resume functionality."""
        return {
            "resume": self.resume,
            "progress_bar": self.progress_bar,
            "simple_progress_bar": self.simple_progress_bar,
            "tmp_path": self.resume_tmp,
            "tmp_base_path": self.tmp_base_path,
            "delete_resume_log_files": self.delete_resume_log_files,
            "delete_intermediate_parquet_files": self.delete_intermediate_parquet_files,
            "tqdm_kwargs": self.tqdm_kwargs,
        }


def find_input_paths(input_path="", file_matcher="", input_file_list=None):
    """Helper method to find input paths, given either a prefix and format, or an
    explicit list of paths.

    Args:
        input_path (str): prefix to search for
        file_matcher (str): matcher to use when searching for files
        input_file_list (List[str]): list of input paths
    Returns:
        matching files, if input_path is provided, otherwise, input_file_list
    Raises:
        FileNotFoundError: if no files are found at the input_path and the provided list is empty.
    """
    input_paths = []
    if input_path:
        if input_file_list:
            raise ValueError("exactly one of input_path or input_file_list is required")
        input_paths = file_io.find_files_matching_path(input_path, file_matcher)
    elif input_file_list is not None:
        if pd.api.types.is_list_like(input_file_list):
            # It's common for users to accidentally pass in an empty list. Give them a friendly error.
            if len(input_file_list) == 0:
                raise ValueError("input_file_list is empty")
        else:
            input_file_list = [input_file_list]
        input_paths = input_file_list
    else:
        raise ValueError("exactly one of input_path or input_file_list is required")
    if len(input_paths) == 0:
        raise FileNotFoundError("No input files found")
    return input_paths


def check_healpix_order_range(
    order, field_name, lower_bound=0, upper_bound=spatial_index.SPATIAL_INDEX_ORDER
):
    """Helper method to check if the ``order`` is within the range determined by the
    ``lower_bound`` and ``upper_bound``, inclusive.

    Args:
        order (int): healpix order to check
        field_name (str): field name to use in the error message
        lower_bound (int): lower bound of range
        upper_bound (int): upper bound of range
    Raise:
        ValueError: if the order is outside the specified range, or bounds
            are unreasonable.
    """
    if lower_bound < 0:
        raise ValueError("healpix orders must be positive")
    if upper_bound > spatial_index.SPATIAL_INDEX_ORDER:
        raise ValueError(f"healpix order should be <= {spatial_index.SPATIAL_INDEX_ORDER}")
    if not lower_bound <= order <= upper_bound:
        raise ValueError(f"{field_name} should be between {lower_bound} and {upper_bound}")
