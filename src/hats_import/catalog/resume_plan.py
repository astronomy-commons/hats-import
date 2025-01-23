"""Utility to hold the file-level pipeline execution plan."""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass, field

import hats.pixel_math.healpix_shim as hp
import numpy as np
from hats import pixel_math
from hats.io import file_io
from hats.pixel_math.healpix_pixel import HealpixPixel
from hats.pixel_math.sparse_histogram import HistogramAggregator, SparseHistogram
from numpy import frombuffer
from upath import UPath

from hats_import.pipeline_resume_plan import PipelineResumePlan


@dataclass
class ResumePlan(PipelineResumePlan):
    """Container class for holding the state of each file in the pipeline plan."""

    input_paths: list[UPath] = field(default_factory=list)
    """resolved list of all files that will be used in the importer"""
    map_files: list[tuple[str, str]] = field(default_factory=list)
    """list of files (and job keys) that have yet to be mapped"""
    split_keys: list[tuple[str, str]] = field(default_factory=list)
    """set of files (and job keys) that have yet to be split"""
    destination_pixel_map: dict[HealpixPixel, int] | None = None
    """Destination pixels and their expected final count"""
    should_run_mapping: bool = True
    should_run_splitting: bool = True
    should_run_reducing: bool = True
    should_run_finishing: bool = True

    MAPPING_STAGE = "mapping"
    SPLITTING_STAGE = "splitting"
    REDUCING_STAGE = "reducing"

    HISTOGRAM_BINARY_FILE = "mapping_histogram.npz"
    HISTOGRAMS_DIR = "histograms"
    ALIGNMENT_FILE = "alignment.pickle"

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        resume: bool = True,
        progress_bar: bool = True,
        simple_progress_bar: bool = False,
        input_paths=None,
        tmp_path=None,
        tmp_base_path: UPath | None = None,
        delete_resume_log_files: bool = True,
        delete_intermediate_parquet_files: bool = True,
        run_stages: list[str] | None = None,
        import_args=None,
    ):
        if import_args:
            super().__init__(**import_args.resume_kwargs_dict())
            if import_args.debug_stats_only:
                run_stages = ["mapping", "finishing"]
            self.input_paths = import_args.input_paths
        else:
            super().__init__(
                resume=resume,
                progress_bar=progress_bar,
                simple_progress_bar=simple_progress_bar,
                tmp_path=tmp_path,
                tmp_base_path=tmp_base_path,
                delete_resume_log_files=delete_resume_log_files,
                delete_intermediate_parquet_files=delete_intermediate_parquet_files,
            )
            self.input_paths = input_paths
        self.gather_plan(run_stages)

    def gather_plan(self, run_stages: list[str] | None = None):
        """Initialize the plan."""
        with self.print_progress(total=4, stage_name="Planning") as step_progress:
            ## Make sure it's safe to use existing resume state.
            super().safe_to_resume()
            step_progress.update(1)

            ## Validate existing resume state.
            ## - if a later stage is complete, the earlier stages should be complete too.
            mapping_done = self.done_file_exists(self.MAPPING_STAGE)
            splitting_done = self.done_file_exists(self.SPLITTING_STAGE)
            reducing_done = self.done_file_exists(self.REDUCING_STAGE)

            if reducing_done and (not mapping_done or not splitting_done):
                raise ValueError("mapping and splitting must be complete before reducing")
            if splitting_done and not mapping_done:
                raise ValueError("mapping must be complete before splitting")
            step_progress.update(1)

            ## Figure out which stages we should run, based on requested `run_stages`
            self.should_run_mapping = not mapping_done
            self.should_run_splitting = not splitting_done
            self.should_run_reducing = not reducing_done
            self.should_run_finishing = True

            if run_stages:
                self.should_run_mapping &= self.MAPPING_STAGE in run_stages
                self.should_run_splitting &= self.SPLITTING_STAGE in run_stages
                self.should_run_reducing &= self.REDUCING_STAGE in run_stages
                self.should_run_finishing = "finishing" in run_stages

            ## Validate that we're operating on the same file set as the previous instance.
            self.input_paths = self.check_original_input_paths(self.input_paths)
            step_progress.update(1)

            ## Gather keys for execution.
            if self.should_run_mapping:
                self.map_files = self.get_remaining_map_keys()
                file_io.make_directory(
                    file_io.append_paths_to_pointer(self.tmp_path, self.HISTOGRAMS_DIR),
                    exist_ok=True,
                )
            if self.should_run_splitting:
                if not (mapping_done or self.should_run_mapping):
                    raise ValueError("mapping must be complete before splitting")

                self.split_keys = self.get_remaining_split_keys()
                file_io.make_directory(
                    file_io.append_paths_to_pointer(self.tmp_path, self.SPLITTING_STAGE),
                    exist_ok=True,
                )
            if self.should_run_reducing:
                ## We don't pre-gather the plan for the reducing keys.
                ## It requires the full destination pixel map.
                if not (splitting_done or self.should_run_splitting):
                    raise ValueError("splitting must be complete before reducing")

                file_io.make_directory(
                    file_io.append_paths_to_pointer(self.tmp_path, self.REDUCING_STAGE),
                    exist_ok=True,
                )
            step_progress.update(1)

    def get_remaining_map_keys(self):
        """Gather remaining keys, dropping successful mapping tasks from histogram names.

        Returns:
            list of mapping keys *not* found in files like /resume/path/mapping_key.npz
        """
        prefix = file_io.get_upath(self.tmp_path) / self.HISTOGRAMS_DIR
        map_file_pattern = re.compile(r"map_(\d+).npz")
        done_indexes = [int(map_file_pattern.match(path.name).group(1)) for path in prefix.glob("*.npz")]
        remaining_indexes = list(set(range(0, len(self.input_paths))) - (set(done_indexes)))
        return [(f"map_{key}", self.input_paths[key]) for key in remaining_indexes]

    def read_histogram(self, healpix_order):
        """Return histogram with healpix_order'd shape

        - Try to find a combined histogram
        - Otherwise, combine histograms from partials
        - Otherwise, return an empty histogram
        """
        file_name = file_io.append_paths_to_pointer(self.tmp_path, self.HISTOGRAM_BINARY_FILE)
        if not file_io.does_file_or_directory_exist(file_name):
            # Read the histogram from partial histograms and combine.
            remaining_map_files = self.get_remaining_map_keys()
            if len(remaining_map_files) > 0:
                raise RuntimeError(f"{len(remaining_map_files)} map stages did not complete successfully.")
            histogram_files = file_io.find_files_matching_path(self.tmp_path, self.HISTOGRAMS_DIR, "*.npz")
            aggregate_histogram = HistogramAggregator(healpix_order)
            for partial_file_name in histogram_files:
                partial = SparseHistogram.from_file(partial_file_name)
                aggregate_histogram.add(partial)

            file_name = file_io.append_paths_to_pointer(self.tmp_path, self.HISTOGRAM_BINARY_FILE)
            with open(file_name, "wb+") as file_handle:
                file_handle.write(aggregate_histogram.full_histogram)
            if self.delete_resume_log_files:
                file_io.remove_directory(
                    file_io.append_paths_to_pointer(self.tmp_path, self.HISTOGRAMS_DIR),
                    ignore_errors=True,
                )

        with open(file_name, "rb") as file_handle:
            full_histogram = frombuffer(file_handle.read(), dtype=np.int64)

        if len(full_histogram) != hp.order2npix(healpix_order):
            raise ValueError(
                "The histogram from the previous execution is incompatible with "
                + "the highest healpix order. To start the importing pipeline "
                + "from scratch with the current order set `resume` to False."
            )
        return full_histogram

    @classmethod
    def partial_histogram_file(cls, tmp_path, mapping_key: str):
        """File name for writing a histogram file to a special intermediate directory.

        As a side effect, this method may create the special intermediate directory.

        Args:
            tmp_path (str): where to write intermediate resume files.
            mapping_key (str): unique string for each mapping task (e.g. "map_57")
        """
        file_io.make_directory(
            file_io.append_paths_to_pointer(tmp_path, cls.HISTOGRAMS_DIR),
            exist_ok=True,
        )
        return file_io.append_paths_to_pointer(tmp_path, cls.HISTOGRAMS_DIR, f"{mapping_key}.npz")

    def get_remaining_split_keys(self):
        """Gather remaining keys, dropping successful split tasks from done file names.

        Returns:
            list of splitting keys *not* found in files like /resume/path/split_key.done
        """
        prefix = file_io.get_upath(self.tmp_path) / self.SPLITTING_STAGE
        split_file_pattern = re.compile(r"split_(\d+)_done")
        done_indexes = [int(split_file_pattern.match(path.name).group(1)) for path in prefix.glob("*_done")]
        remaining_indexes = list(set(range(0, len(self.input_paths))) - set(done_indexes))
        return [(f"split_{key}", self.input_paths[key]) for key in remaining_indexes]

    @classmethod
    def splitting_key_done(cls, tmp_path, splitting_key: str):
        """Mark a single splitting task as done

        Args:
            tmp_path (str): where to write intermediate resume files.
            splitting_key (str): unique string for each splitting task (e.g. "split_57")
        """
        cls.touch_key_done_file(tmp_path, cls.SPLITTING_STAGE, splitting_key)

    @classmethod
    def reducing_key_done(cls, tmp_path, reducing_key: str):
        """Mark a single reducing task as done

        Args:
            tmp_path (str): where to write intermediate resume files.
            reducing_key (str): unique string for each reducing task (e.g. "3_57")
        """
        cls.touch_key_done_file(tmp_path, cls.REDUCING_STAGE, reducing_key)

    def wait_for_mapping(self, futures):
        """Wait for mapping futures to complete."""
        self.wait_for_futures(futures, self.MAPPING_STAGE)
        remaining_map_items = self.get_remaining_map_keys()
        if len(remaining_map_items) > 0:
            raise RuntimeError("some map stages did not complete successfully.")
        self.touch_stage_done_file(self.MAPPING_STAGE)

    def get_alignment_file(
        self,
        raw_histogram,
        constant_healpix_order,
        highest_healpix_order,
        lowest_healpix_order,
        pixel_threshold,
        drop_empty_siblings,
        expected_total_rows,
    ) -> UPath:
        """Get a pointer to the existing alignment file for the pipeline, or
        generate a new alignment using provided arguments.

        Args:
            raw_histogram (:obj:`np.array`): one-dimensional numpy array of long integers where the
                value at each index corresponds to the number of objects found at the healpix pixel.
            constant_healpix_order (int): if positive, use this as the order for
                all non-empty partitions. else, use remaining arguments.
            highest_healpix_order (int):  the highest healpix order (e.g. 5-10)
            lowest_healpix_order (int): the lowest healpix order (e.g. 1-5). specifying a lowest order
                constrains the partitioning to prevent spatially large pixels.
            threshold (int): the maximum number of objects allowed in a single pixel
            drop_empty_siblings (bool):  if 3 of 4 pixels are empty, keep only the non-empty pixel
            expected_total_rows (int): number of expected rows found in the dataset.

        Returns:
            path to cached alignment file.
        """
        file_name = file_io.append_paths_to_pointer(self.tmp_path, self.ALIGNMENT_FILE)
        if not file_io.does_file_or_directory_exist(file_name):
            if constant_healpix_order >= 0:
                alignment = np.full(len(raw_histogram), None)
                for pixel_num, pixel_sum in enumerate(raw_histogram):
                    alignment[pixel_num] = (
                        constant_healpix_order,
                        pixel_num,
                        pixel_sum,
                    )
            else:
                alignment = pixel_math.generate_alignment(
                    raw_histogram,
                    highest_order=highest_healpix_order,
                    lowest_order=lowest_healpix_order,
                    threshold=pixel_threshold,
                    drop_empty_siblings=drop_empty_siblings,
                )
            with open(file_name, "wb") as pickle_file:
                pickle.dump(alignment, pickle_file)

        if self.destination_pixel_map is None:
            with open(file_name, "rb") as pickle_file:
                alignment = pickle.load(pickle_file)
            non_none_elements = alignment[alignment != np.array(None)]
            pixel_list = np.unique(non_none_elements)
            self.destination_pixel_map = {
                HealpixPixel(order, pix): count for (order, pix, count) in pixel_list if int(count) > 0
            }
        total_rows = sum(self.destination_pixel_map.values())
        if total_rows != expected_total_rows:
            raise ValueError(
                f"Number of rows ({total_rows}) does not match expectation ({expected_total_rows})"
            )

        return file_name

    def wait_for_splitting(self, futures):
        """Wait for splitting futures to complete."""
        self.wait_for_futures(futures, self.SPLITTING_STAGE)
        remaining_split_items = self.get_remaining_split_keys()
        if len(remaining_split_items) > 0:
            raise RuntimeError(f"{len(remaining_split_items)} split stages did not complete successfully.")
        self.touch_stage_done_file(self.SPLITTING_STAGE)

    def get_reduce_items(self):
        """Fetch a triple for each partition to reduce.

        Triple contains:

        - destination pixel (healpix pixel with both order and pixel)
        - number of rows expected for this pixel
        - reduce key (string of destination order+pixel)
        """
        if self.destination_pixel_map is None:
            raise RuntimeError("destination pixel map not provided for progress tracking.")

        reduced_pixels = self.read_done_pixels(self.REDUCING_STAGE)

        remaining_pixels = list(set(self.destination_pixel_map.keys()) - set(reduced_pixels))
        return [
            (hp_pixel, self.destination_pixel_map[hp_pixel], f"{hp_pixel.order}_{hp_pixel.pixel}")
            for hp_pixel in remaining_pixels
        ]

    def get_destination_pixels(self):
        """Create HealpixPixel list of all destination pixels."""
        if self.destination_pixel_map is None:
            raise RuntimeError("destination pixel map not known.")
        return list(self.destination_pixel_map.keys())

    def wait_for_reducing(self, futures):
        """Wait for reducing futures to complete."""
        self.wait_for_futures(futures, self.REDUCING_STAGE, fail_fast=True)
        remaining_reduce_items = self.get_reduce_items()
        if len(remaining_reduce_items) > 0:
            raise RuntimeError(f"{len(remaining_reduce_items)} reduce stages did not complete successfully.")
        self.touch_stage_done_file(self.REDUCING_STAGE)
