"""Utility to hold a pipeline's execution and resume plan."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from dask.distributed import as_completed, get_worker
from dask.distributed import print as dask_print
from hipscat.io import FilePointer, file_io
from hipscat.pixel_math.healpix_pixel import HealpixPixel

from tqdm.auto import tqdm as auto_tqdm
from tqdm.std import tqdm as std_tqdm


@dataclass
class PipelineResumePlan:
    """Container class for holding the state of pipeline plan."""

    tmp_path: FilePointer
    """path for any intermediate files"""
    resume: bool = True
    """if there are existing intermediate resume files, should we
    read those and continue to run pipeline where we left off"""
    progress_bar: bool = True
    """if true, a tqdm progress bar will be displayed for user
    feedback of planning progress"""
    simple_progress_bar: bool = False
    """if displaying a progress bar, use a text-only simple progress
    bar instead of widget. this can be useful in some environments when running
    in a notebook where ipywidgets cannot be used (see `progress_bar` argument)"""
    delete_resume_log_files: bool = True
    """should we delete task-level done files once each stage is complete?
    if False, we will keep all sub-histograms from the mapping stage, and all
    done marker files at the end of the pipeline."""

    ORIGINAL_INPUT_PATHS = "input_paths.txt"

    def safe_to_resume(self):
        """Check that we are ok to resume an in-progress pipeline, if one exists.

        Raises:
            ValueError: if the tmp_path already exists and contains some files.
        """
        if file_io.directory_has_contents(self.tmp_path):
            if not self.resume:
                self.clean_resume_files()
            else:
                print(f"tmp_path ({self.tmp_path}) contains intermediate files; resuming prior progress.")
        file_io.make_directory(self.tmp_path, exist_ok=True)

    def done_file_exists(self, stage_name):
        """Is there a file at a given path?
        For a done file, the existence of the file is the only signal needed to indicate
        a pipeline segment is complete.

        Args:
            stage_name(str): name of the stage (e.g. mapping, reducing)
        Returns:
            boolean, True if the done file exists at tmp_path. False otherwise.
        """
        return file_io.does_file_or_directory_exist(
            file_io.append_paths_to_pointer(self.tmp_path, f"{stage_name}_done")
        )

    def touch_stage_done_file(self, stage_name):
        """Touch (create) a done file for a whole pipeline stage.
        For a done file, the existence of the file is the only signal needed to indicate
        a pipeline segment is complete.

        Args:
            stage_name(str): name of the stage (e.g. mapping, reducing)
        """
        Path(file_io.append_paths_to_pointer(self.tmp_path, f"{stage_name}_done")).touch()

    @classmethod
    def touch_key_done_file(cls, tmp_path, stage_name, key):
        """Touch (create) a done file for a single key, within a pipeline stage.

        Args:
            stage_name(str): name of the stage (e.g. mapping, reducing)
            stage_name(str): name of the stage (e.g. mapping, reducing)
            key (str): unique string for each task (e.g. "map_57")
        """
        Path(file_io.append_paths_to_pointer(tmp_path, stage_name, f"{key}_done")).touch()

    def read_done_keys(self, stage_name):
        """Inspect the stage's directory of done files, fetching the keys from done file names.

        Args:
            stage_name(str): name of the stage (e.g. mapping, reducing)
        Return:
            List[str] - all keys found in done directory
        """
        prefix = file_io.append_paths_to_pointer(self.tmp_path, stage_name)
        return self.get_keys_from_file_names(prefix, "_done")

    @staticmethod
    def get_keys_from_file_names(directory, extension):
        """Gather keys for successful tasks from result file names.

        Args:
            directory: where to look for result files. this is NOT a recursive lookup
            extension (str): file suffix to look for and to remove from all file names.
                if you expect a file like "map_01.csv", extension should be ".csv"

        Returns:
            list of keys taken from files like /resume/path/{key}{extension}
        """
        result_files = file_io.find_files_matching_path(directory, f"*{extension}")
        keys = []
        for file_path in result_files:
            result_file_name = file_io.get_basename_from_filepointer(file_path)
            match = re.match(r"(.*)" + extension, str(result_file_name))
            keys.append(match.group(1))
        return keys

    def clean_resume_files(self):
        """Remove all intermediate files created in execution."""
        if self.delete_resume_log_files:
            file_io.remove_directory(self.tmp_path, ignore_errors=True)

    def wait_for_futures(self, futures, stage_name, fail_fast=False):
        """Wait for collected futures to complete.

        As each future completes, check the returned status.

        Args:
            futures(List[future]): collected futures
            stage_name(str): name of the stage (e.g. mapping, reducing)
            fail_fast (bool): if True, we will re-raise the first exception
                encountered and NOT continue. this may lead to unexpected
                behavior for in-progress tasks.
        Raises:
            RuntimeError: if any future returns an error status.
        """
        some_error = False
        for future in self.print_progress(as_completed(futures), stage_name=stage_name, total=len(futures)):
            if future.status == "error":
                some_error = True
                if fail_fast:
                    raise future.exception()

        if some_error:
            raise RuntimeError(f"Some {stage_name} stages failed. See logs for details.")

    def print_progress(self, iterable=None, total=None, stage_name=None):
        """Create a progress bar that will provide user with task feedback.

        This is a thin wrapper around the static ``print_progress`` method that uses
        member variables for the caller's convenience.

        Args:
            iterable (iterable): Optional. provides iterations to progress updates.
            total (int): Optional. Expected iterations.
            stage_name (str): name of the stage (e.g. mapping, reducing). this will
                be further formatted with ``get_formatted_stage_name``, so the caller
                doesn't need to worry about that.
        """
        return print_progress(
            iterable=iterable,
            total=total,
            stage_name=stage_name,
            use_progress_bar=self.progress_bar,
            simple_progress_bar=self.simple_progress_bar,
        )

    def check_original_input_paths(self, input_paths):
        """Validate that we're operating on the same file set as the original pipeline,
        or save the inputs as the originals if not found.

        Args:
            input_paths (list[str]): input paths that will be processed by a pipeline.

        Raises:
            ValueError: if the retrieved file set differs from `input_paths`.
        """
        input_paths = set(input_paths)
        input_paths = [str(p) for p in input_paths]
        input_paths.sort()

        original_input_paths = []

        log_file_path = file_io.append_paths_to_pointer(self.tmp_path, self.ORIGINAL_INPUT_PATHS)
        try:
            with open(log_file_path, "r", encoding="utf-8") as file_handle:
                contents = file_handle.readlines()
            contents = [path.strip() for path in contents]
            original_input_paths = list(set(contents))
            original_input_paths.sort()
        except FileNotFoundError:
            pass

        if len(original_input_paths) == 0:
            with open(log_file_path, "w", encoding="utf-8") as file_handle:
                for path in input_paths:
                    file_handle.write(f"{path}\n")
        else:
            if original_input_paths != input_paths:
                raise ValueError("Different file set from resumed pipeline execution.")

        return input_paths


def get_pixel_cache_directory(cache_path, pixel: HealpixPixel):
    """Create a path for intermediate pixel data.

    You can use this over the paths.get_pixel_directory method, as it will include the pixel
    number in the path. Further, it will just *look* different from a real hipscat
    path, so it's clearer that it's a temporary directory::

        {cache_path}/order_{order}/dir_{dir}/pixel_{pixel}/

    Args:
        cache_path (str): root path to cache
        pixel (HealpixPixel): pixel partition data
    """
    return file_io.append_paths_to_pointer(
        cache_path, f"order_{pixel.order}", f"dir_{pixel.dir}", f"pixel_{pixel.pixel}"
    )


def print_task_failure(custom_message, exception):
    """Use dask's distributed print feature to print the exception message to the task's logs
    and to the controller job's logs.

    Optionally print the worker address if a worker is found.

    Args:
        custom_message (str): some custom message for the task that might help with debugging
        exception (Exception): error raised in execution of the task.
    """
    dask_print(custom_message)
    try:
        dask_print("  worker address:", get_worker().address)
    except Exception:  # pylint: disable=broad-exception-caught
        pass
    dask_print(exception)


def get_formatted_stage_name(stage_name) -> str:
    """Create a stage name of consistent minimum length. Ensures that the tqdm
    progress bars can line up nicely when multiple stages must run.

    Args:
        stage_name (str): name of the stage (e.g. mapping, reducing)
    """
    if stage_name is None or len(stage_name) == 0:
        stage_name = "progress"

    return f"{stage_name.capitalize(): <10}"


def print_progress(
    iterable=None, total=None, stage_name=None, use_progress_bar=True, simple_progress_bar=False
):
    """Create a progress bar that will provide user with task feedback.

    Args:
        iterable (iterable): Optional. provides iterations to progress updates.
        total (int): Optional. Expected iterations.
        stage_name (str): name of the stage (e.g. mapping, reducing). this will
            be further formatted with ``get_formatted_stage_name``, so the caller
            doesn't need to worry about that.
        use_progress_bar (bool): should we display any progress. typically False
            when no stdout is expected.
        simple_progress_bar (bool): if displaying a progress bar, use a text-only
            simple progress bar instead of widget. this can be useful when running
            in a particular notebook where ipywidgets cannot be used
            (only used when ``use_progress_bar`` is True)
    """
    if simple_progress_bar:
        return std_tqdm(
            iterable,
            desc=get_formatted_stage_name(stage_name),
            total=total,
            disable=not use_progress_bar,
        )

    return auto_tqdm(
        iterable,
        desc=get_formatted_stage_name(stage_name),
        total=total,
        disable=not use_progress_bar,
    )
