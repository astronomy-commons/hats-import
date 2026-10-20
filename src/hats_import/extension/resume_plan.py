"""Utility to hold the pipeline execution plan for an extension split"""

from __future__ import annotations

from dataclasses import dataclass

from hats.io import file_io
from hats.pixel_math.healpix_pixel import HealpixPixel

import hats_import.file_io as import_io
from hats_import.extension.arguments import ExtensionArguments
from hats_import.pipeline_resume_plan import PipelineResumePlan


@dataclass
class ExtensionSplitPlan(PipelineResumePlan):
    """Container class for holding the state of an extension split.

    One task splits one partition of one input table into both sides, so a task is done when
    the partition has been written to the core and to the extension. A resumed run skips the
    partitions that are already done, and writes the rest."""

    SPLITTING_STAGE = "splitting"

    def __init__(self, args: ExtensionArguments):
        if not args.tmp_path:  # pragma: no cover (not reachable, but required for mypy)
            raise ValueError("tmp_path is required")
        super().__init__(pipeline_name="extension", **args.resume_kwargs_dict())
        with self.print_progress(total=1, stage_name="Planning") as step_progress:
            self.safe_to_resume()
            file_io.make_directory(
                import_io.append_paths_to_pointer(self.tmp_path, self.SPLITTING_STAGE), exist_ok=True
            )
            step_progress.update(1)

    @staticmethod
    def splitting_key(input_catalog, pixel: HealpixPixel) -> str:
        """Key of the task that splits one partition of one input table."""
        return f"{input_catalog.catalog_info.catalog_name}_{pixel.order}_{pixel.pixel}"

    @classmethod
    def splitting_key_done(cls, tmp_path, splitting_key: str):
        """Mark a single splitting task as done.

        Args:
            tmp_path (str): where to write intermediate resume files.
            splitting_key (str): unique string for each splitting task (e.g. "small_sky_1_44")
        """
        cls.touch_key_done_file(tmp_path, cls.SPLITTING_STAGE, splitting_key)

    def remaining_pixels(self, input_catalog) -> list[HealpixPixel]:
        """The partitions of an input table that are not split yet."""
        done_keys = set(self.read_markers(self.SPLITTING_STAGE))
        return [
            pixel
            for pixel in input_catalog.get_healpix_pixels()
            if self.splitting_key(input_catalog, pixel) not in done_keys
        ]

    def wait_for_splitting(self, futures, input_catalog):
        """Wait for the splitting tasks of one input table to complete."""
        stage_name = f"{self.SPLITTING_STAGE} {input_catalog.catalog_info.catalog_name}"
        self.wait_for_futures(futures, stage_name, fail_fast=True)
        remaining = self.remaining_pixels(input_catalog)
        if len(remaining) > 0:
            raise RuntimeError(f"{len(remaining)} splitting stages did not complete successfully.")

    def is_splitting_done(self) -> bool:
        """Is every partition of every input table split?"""
        return self.done_file_exists(self.SPLITTING_STAGE)

    def splitting_done(self):
        """Mark the splitting stage as done, once every input table is split."""
        self.touch_stage_done_file(self.SPLITTING_STAGE)
