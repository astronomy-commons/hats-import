"""Utility to hold the pipeline execution plan."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import pandas as pd
from hipscat import pixel_math
from hipscat.io import file_io, paths
from hipscat.pixel_math.healpix_pixel import HealpixPixel
from tqdm.auto import tqdm

from hipscat_import.margin_cache.margin_cache_arguments import MarginCacheArguments
from hipscat_import.pipeline_resume_plan import PipelineResumePlan


@dataclass
class MarginCachePlan(PipelineResumePlan):
    """Container class for holding the state of each file in the pipeline plan."""

    margin_pair_file: str | None = None
    partition_pixels: List[HealpixPixel] = field(default_factory=list)
    combined_pixels: List[HealpixPixel] = field(default_factory=list)

    MAPPING_STAGE = "mapping"
    REDUCING_STAGE = "reducing"
    MARGIN_PAIR_FILE = "margin_pair.csv"

    def __init__(self, args: MarginCacheArguments, combined_pixels, partition_pixels):
        if not args.tmp_path:  # pragma: no cover (not reachable, but required for mypy)
            raise ValueError("tmp_path is required")
        super().__init__(
            resume=args.resume,
            progress_bar=args.progress_bar,
            tmp_path=args.tmp_path,
        )
        self.combined_pixels = combined_pixels
        self.partition_pixels = partition_pixels
        self._gather_plan(args)

    def _gather_plan(self, args):
        """Initialize the plan."""
        with tqdm(
            total=3, desc=self.get_formatted_stage_name("Planning"), disable=not self.progress_bar
        ) as step_progress:
            ## Make sure it's safe to use existing resume state.
            super().safe_to_resume()
            mapping_done = self.is_mapping_done()
            reducing_done = self.is_reducing_done()
            if reducing_done and (not mapping_done):
                raise ValueError("mapping must be complete before reducing")
            step_progress.update(1)

            self.margin_pair_file = file_io.append_paths_to_pointer(self.tmp_path, self.MARGIN_PAIR_FILE)
            if not file_io.does_file_or_directory_exist(self.margin_pair_file):
                margin_pairs = _find_partition_margin_pixel_pairs(self.combined_pixels, args.margin_order)
                margin_pairs.to_csv(self.margin_pair_file)
            step_progress.update(1)

            file_io.make_directory(
                file_io.append_paths_to_pointer(self.tmp_path, self.MAPPING_STAGE),
                exist_ok=True,
            )
            file_io.make_directory(
                file_io.append_paths_to_pointer(self.tmp_path, self.REDUCING_STAGE),
                exist_ok=True,
            )
            _create_margin_directory(self.combined_pixels, args.catalog_path, args.output_storage_options)

            step_progress.update(1)

    @classmethod
    def mapping_key_done(cls, tmp_path, mapping_key: str):
        """Mark a single mapping task as done

        Args:
            tmp_path (str): where to write intermediate resume files.
            mapping_key (str): unique string for each mapping task (e.g. "map_1_24")
        """
        cls.touch_key_done_file(tmp_path, cls.MAPPING_STAGE, mapping_key)

    def wait_for_mapping(self, futures):
        """Wait for mapping stage futures to complete."""
        self.wait_for_futures(futures, self.MAPPING_STAGE)
        remaining_pixels_to_map = self.get_remaining_map_keys()
        if len(remaining_pixels_to_map) > 0:
            raise RuntimeError(
                f"{len(remaining_pixels_to_map)} mapping stages did not complete successfully."
            )
        self.touch_stage_done_file(self.MAPPING_STAGE)

    def is_mapping_done(self) -> bool:
        """Are there sources left to count?"""
        return self.done_file_exists(self.MAPPING_STAGE)

    def get_remaining_map_keys(self):
        """Fetch a tuple for each pixel/partition left to map."""
        map_keys = set(self.read_done_keys(self.MAPPING_STAGE))
        return [
            (f"{hp_pixel.order}_{hp_pixel.pixel}", hp_pixel)
            for hp_pixel in self.partition_pixels
            if f"{hp_pixel.order}_{hp_pixel.pixel}" not in map_keys
        ]

    @classmethod
    def reducing_key_done(cls, tmp_path, reducing_key: str):
        """Mark a single reducing task as done

        Args:
            tmp_path (str): where to write intermediate resume files.
            reducing_key (str): unique string for each reducing task (e.g. "3_57")
        """
        cls.touch_key_done_file(tmp_path, cls.REDUCING_STAGE, reducing_key)

    def get_remaining_reduce_keys(self):
        """Fetch a tuple for each object catalog pixel to reduce."""
        reduced_keys = set(self.read_done_keys(self.REDUCING_STAGE))
        reduce_items = [
            (f"{hp_pixel.order}_{hp_pixel.pixel}", hp_pixel)
            for hp_pixel in self.combined_pixels
            if f"{hp_pixel.order}_{hp_pixel.pixel}" not in reduced_keys
        ]
        return reduce_items

    def is_reducing_done(self) -> bool:
        """Are there partitions left to reduce?"""
        return self.done_file_exists(self.REDUCING_STAGE)

    def wait_for_reducing(self, futures):
        """Wait for reducing stage futures to complete."""
        self.wait_for_futures(futures, self.REDUCING_STAGE)
        remaining_sources_to_reduce = self.get_remaining_reduce_keys()
        if len(remaining_sources_to_reduce) > 0:
            raise RuntimeError(
                f"{len(remaining_sources_to_reduce)} reducing stages did not complete successfully."
            )
        self.touch_stage_done_file(self.REDUCING_STAGE)


def _find_partition_margin_pixel_pairs(stats, margin_order):
    """Creates a DataFrame filled with many-to-many connections between
    the catalog partition pixels and the margin pixels at `margin_order`.
    """
    norders = []
    part_pix = []
    margin_pix = []

    for healpixel in stats:
        order = healpixel.order
        pix = healpixel.pixel

        d_order = margin_order - order

        margins = pixel_math.get_margin(order, pix, d_order)

        for m_p in margins:
            norders.append(order)
            part_pix.append(pix)
            margin_pix.append(m_p)

    margin_pairs_df = pd.DataFrame(
        zip(norders, part_pix, margin_pix),
        columns=["partition_order", "partition_pixel", "margin_pixel"],
    )
    return margin_pairs_df


def _create_margin_directory(stats, output_path, storage_options):
    """Creates directories for all the catalog partitions."""
    for healpixel in stats:
        order = healpixel.order
        pix = healpixel.pixel

        destination_dir = paths.pixel_directory(output_path, order, pix)
        file_io.make_directory(destination_dir, exist_ok=True, storage_options=storage_options)
