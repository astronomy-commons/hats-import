"""Utility to hold the pipeline execution plan."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from hats import pixel_math
from hats.io import file_io
from hats.pixel_math.healpix_pixel import HealpixPixel

from hats_import.margin_cache.margin_cache_arguments import MarginCacheArguments
from hats_import.pipeline_resume_plan import PipelineResumePlan


@dataclass
class MarginCachePlan(PipelineResumePlan):
    """Container class for holding the state of each file in the pipeline plan."""

    margin_pair_file: str | None = None
    partition_pixels: list[HealpixPixel] = field(default_factory=list)
    combined_pixels: list[HealpixPixel] = field(default_factory=list)

    MAPPING_STAGE = "mapping"
    REDUCING_STAGE = "reducing"
    MARGIN_PAIR_FILE = "margin_pair.csv"
    MAPPING_TOTAL_FILE = "mapping_total"

    def __init__(self, args: MarginCacheArguments):
        if not args.tmp_path:  # pragma: no cover (not reachable, but required for mypy)
            raise ValueError("tmp_path is required")
        super().__init__(**args.resume_kwargs_dict())
        self._gather_plan(args)

    def _gather_plan(self, args):
        """Initialize the plan."""
        with self.print_progress(total=3, stage_name="Planning") as step_progress:
            ## Make sure it's safe to use existing resume state.
            super().safe_to_resume()
            mapping_done = self.is_mapping_done()
            reducing_done = self.is_reducing_done()
            if reducing_done and (not mapping_done):
                raise ValueError("mapping must be complete before reducing")
            step_progress.update(1)

            self.partition_pixels = args.catalog.partition_info.get_healpix_pixels()
            negative_pixels = args.catalog.generate_negative_tree_pixels()
            self.combined_pixels = self.partition_pixels + negative_pixels
            self.margin_pair_file = file_io.append_paths_to_pointer(self.tmp_path, self.MARGIN_PAIR_FILE)
            if not file_io.does_file_or_directory_exist(self.margin_pair_file):
                margin_pairs = _find_partition_margin_pixel_pairs(self.combined_pixels, args.margin_order)
                margin_pairs.to_csv(self.margin_pair_file, index=False)
            step_progress.update(1)

            file_io.make_directory(
                file_io.append_paths_to_pointer(self.tmp_path, self.MAPPING_STAGE),
                exist_ok=True,
            )
            file_io.make_directory(
                file_io.append_paths_to_pointer(self.tmp_path, self.REDUCING_STAGE),
                exist_ok=True,
            )

            step_progress.update(1)

    @classmethod
    def mapping_key_done(cls, tmp_path, mapping_key: str, num_rows: int):
        """Mark a single mapping task as done

        Args:
            tmp_path (str): where to write intermediate resume files.
            mapping_key (str): unique string for each mapping task (e.g. "map_1_24")
        """
        cls.write_marker_file(tmp_path, cls.MAPPING_STAGE, mapping_key, str(num_rows))

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

    def get_mapping_total(self) -> int:
        """Find the total number of rows, based on the intermediate markers."""
        if not self.is_mapping_done():
            raise ValueError("mapping stage is not done yet.")

        total_marker_file = file_io.append_paths_to_pointer(self.tmp_path, self.MAPPING_TOTAL_FILE)

        if file_io.does_file_or_directory_exist(total_marker_file):
            marker_value = file_io.load_text_file(total_marker_file)
            return _marker_value_to_int(marker_value)

        markers = self.read_markers(self.MAPPING_STAGE)
        total_marker_value = sum(_marker_value_to_int(value) for value in markers.values())
        file_io.write_string_to_file(total_marker_file, str(total_marker_value))
        return total_marker_value

    def get_remaining_map_keys(self):
        """Fetch a tuple for each pixel/partition left to map."""
        mapped_pixels = set(self.read_done_pixels(self.MAPPING_STAGE))
        remaining_pixels = list(set(self.partition_pixels) - set(mapped_pixels))
        return [(f"{hp_pixel.order}_{hp_pixel.pixel}", hp_pixel) for hp_pixel in remaining_pixels]

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
        reduced_pixels = self.read_done_pixels(self.REDUCING_STAGE)
        remaining_pixels = list(set(self.combined_pixels) - set(reduced_pixels))
        return [(f"{hp_pixel.order}_{hp_pixel.pixel}", hp_pixel) for hp_pixel in remaining_pixels]

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


def _find_partition_margin_pixel_pairs(combined_pixels, margin_order):
    """Creates a DataFrame filled with many-to-many connections between
    the catalog partition pixels and the negative margin pixels at `margin_order`.

    Args:
        combined_pixels (List[HealpixPixel]): union of catalog partition pixels
            and the negative tree pixels for the catalog
        margin_order (int): the order of healpixels that will be used to constrain
            the margin data before doing more precise boundary checking.
    """
    norders = []
    part_pix = []
    margin_pix = []

    for healpixel in combined_pixels:
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
    ).sort_values("margin_pixel")
    return margin_pairs_df


def _marker_value_to_int(marker_value: list[str]) -> int:
    """Convenience method to parse the contents of a marker file."""
    if len(marker_value) != 1:
        raise ValueError("Marker file should contain only one integer value.")
    return int(marker_value[0])
