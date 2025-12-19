"""Utility to hold the pipeline execution plan."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import hats.pixel_math.healpix_shim as hp
import numpy as np
import pandas as pd
from hats import pixel_math, read_hats
from hats.io import file_io
from hats.pixel_math.healpix_pixel import HealpixPixel
from hats.pixel_tree import PixelAlignment, align_trees

from hats_import.nest_light_curves.arguments import NestLightCurveArguments
from hats_import.pipeline_resume_plan import PipelineResumePlan


@dataclass
class NestLightCurvePlan(PipelineResumePlan):
    """Container class for holding the state of each file in the pipeline plan."""

    count_keys: list[tuple[HealpixPixel, list[HealpixPixel], str]] = field(default_factory=list)
    """set of pixels (and job keys) that have yet to be counted"""
    object_map: dict[HealpixPixel, list[HealpixPixel]] | None = None
    """Map of object pixels to source pixels, with counting key."""

    COUNTING_STAGE = "counting"
    SOURCE_MAP_FILE = "source_object_map.npz"

    def __init__(self, args: NestLightCurveArguments):
        if not args.tmp_path:  # pragma: no cover (not reachable, but required for mypy)
            raise ValueError("tmp_path is required")
        super().__init__(**args.resume_kwargs_dict())
        self.gather_plan(args)

    def gather_plan(self, args):
        """Initialize the plan."""
        with self.print_progress(total=2, stage_name="Planning") as step_progress:
            ## Make sure it's safe to use existing resume state.
            super().safe_to_resume()
            step_progress.update(1)

            self.count_keys = []
            object_catalog = read_hats(args.object_catalog_dir)
            source_map_file = file_io.append_paths_to_pointer(self.tmp_path, self.SOURCE_MAP_FILE)
            if file_io.does_file_or_directory_exist(source_map_file):
                self.object_map = np.load(source_map_file, allow_pickle=True)["arr_0"].item()
            else:
                source_catalog = read_hats(args.source_catalog_dir)
                self.object_map = object_neighbor_map(object_catalog, source_catalog)
                np.savez_compressed(source_map_file, self.object_map)
            self.count_keys = self.get_sources_to_count()
            step_progress.update(1)

    def wait_for_counting(self, futures):
        """Wait for counting stage futures to complete."""
        self.wait_for_futures(futures, self.COUNTING_STAGE)
        remaining_sources_to_count = self.get_sources_to_count()
        if len(remaining_sources_to_count) > 0:
            raise RuntimeError(
                f"{len(remaining_sources_to_count)} counting stages did not complete successfully."
            )
        self.touch_stage_done_file(self.COUNTING_STAGE)

    def is_counting_done(self) -> bool:
        """Are there sources left to count?"""
        return self.done_file_exists(self.COUNTING_STAGE)

    def get_sources_to_count(self):
        """Fetch a triple for each source pixel to join and count.

        Triple contains:
            - source pixel
            - object pixels (healpix pixel with both order and pixel, for aligning and
              neighboring object pixels)
            - source key (string of source order+pixel)
        """
        if self.object_map is None:
            raise ValueError("object_map not provided for progress tracking.")
        count_file_pattern = re.compile(r"(\d+)_(\d+).csv")
        counted_pixel_tuples = [
            count_file_pattern.match(path.name).group(1, 2) for path in self.tmp_path.glob("*.csv")
        ]
        counted_pixels = [HealpixPixel(int(match[0]), int(match[1])) for match in counted_pixel_tuples]

        remaining_pixels = list(set(self.object_map.keys()) - set(counted_pixels))
        return [
            (hp_pixel, self.object_map[hp_pixel], f"{hp_pixel.order}_{hp_pixel.pixel}")
            for hp_pixel in remaining_pixels
        ]

    def combine_partial_results(self, output_path) -> int:
        """Combine many partial CSVs into single partition join info.

        Args:
            input_path(str): intermediate directory with partial result CSVs. likely, the
                directory used in the previous `count_joins` call as `cache_path`
            output_path(str): directory to write the combined results CSVs.

        Returns:
            integer that is the sum of all matched num_rows.
        """
        if self.object_map is None:
            raise ValueError("object_map not provided for progress tracking.")
        count_file_pattern = re.compile(r"(\d+)_(\d+).csv")
        partial_files = list(self.tmp_path.glob("*.csv"))
        counted_pixel_tuples = [
            matches.group(1, 2)
            for path in partial_files
            if (matches := count_file_pattern.match(path.name)) is not None
        ]
        counted_pixels = [HealpixPixel(int(match[0]), int(match[1])) for match in counted_pixel_tuples]
        remaining_pixels = list(set(self.object_map.keys()) - set(counted_pixels))
        if len(remaining_pixels) > 0:
            raise ValueError("All partitions must be counted before combining results.")

        partials = []

        for partial_file in partial_files:
            partials.append(file_io.load_csv_to_pandas(partial_file))

        dataframe = pd.concat(partials)

        file_io.write_dataframe_to_csv(
            dataframe=dataframe, file_pointer=output_path / "partition_info.csv", index=False
        )

        return dataframe["num_rows"].sum()


def object_neighbor_map(object_catalog, source_catalog):
    """Build up a map of object pixels to the neighboring source pixels."""

    ## Direct aligment from object to source
    ###############################################
    grouped_alignment = align_trees(
        object_catalog.pixel_tree, source_catalog.pixel_tree, "outer"
    ).pixel_mapping.groupby(
        [
            PixelAlignment.PRIMARY_ORDER_COLUMN_NAME,
            PixelAlignment.PRIMARY_PIXEL_COLUMN_NAME,
        ],
        group_keys=False,
    )

    ## Lots of cute comprehension is happening here.
    ## create tuple of (object order/pixel) and [array of tuples of (source order/pixel)]
    object_to_source = [
        (
            HealpixPixel(int(object_idx[0]), int(object_idx[1])),
            [
                HealpixPixel(int(source_elem[0]), int(source_elem[1]))
                for source_elem in source_group.dropna().to_numpy().T[2:4].T
            ],
        )
        for object_idx, source_group in grouped_alignment
    ]
    ## Treat the array of tuples as a dictionary.
    object_to_source = dict(object_to_source)

    ## Object neighbors for source
    ###############################################
    max_order = max(
        object_catalog.partition_info.get_highest_order(),
        source_catalog.partition_info.get_highest_order(),
    )

    source_order_map = np.full(hp.order2npix(max_order), -1)

    for pixel in source_catalog.partition_info.get_healpix_pixels():
        explosion_factor = 4 ** (max_order - pixel.order)
        exploded_pixels = [
            *range(
                pixel.pixel * explosion_factor,
                (pixel.pixel + 1) * explosion_factor,
            )
        ]
        source_order_map[exploded_pixels] = pixel.order

    for object_pixel, sources in object_to_source.items():
        # get all neighboring pixels
        neighbors = pixel_math.get_margin(object_pixel.order, object_pixel.pixel, 0)

        ## get rid of -1s and normalize to max order
        explosion_factor = 4 ** (max_order - object_pixel.order)
        ## explode out the object pixels to the same order as source map
        ## NB: This may find non-bordering source neighbors, but that's ok!
        neighbors = [
            [
                *range(
                    pixel * explosion_factor,
                    (pixel + 1) * explosion_factor,
                )
            ]
            for pixel in neighbors
            if pixel != -1
        ]
        ## Flatten out the exploded list of lists
        neighbors = [item for sublist in neighbors for item in sublist]

        neighbors_orders = source_order_map[neighbors]
        desploded = [
            HealpixPixel(order, hoo_pixel >> 2 * (max_order - order))
            for order, hoo_pixel in list(zip(neighbors_orders, neighbors))
            if order != -1
        ]
        neighbors = set(desploded) - set(sources)
        sources.extend(list(neighbors))

    return object_to_source
