"""Utility to hold all arguments required throughout verification pipeline"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import hats.io.paths
from hats.io import file_io
from upath import UPath


@dataclass(kw_only=True)
class VerificationArguments:
    """Container for verification arguments."""

    input_catalog_path: UPath = field()
    """Path to an existing catalog that will be inspected. This must be a directory
    containing (at least) the hats ancillary files and a 'dataset/' directory
    containing the parquet dataset. Can be supplied as a string or path object."""
    output_path: UPath = field()
    """Directory where the verification report should be written.
     Can be supplied as a string or path object."""
    output_filename: str = field(default="verifier_results.csv")
    """Filename for the verification report."""
    truth_total_rows: int | None = field(default=None)
    """Total number of rows expected in this catalog."""
    truth_schema: UPath | None = field(default=None)
    """Path to a parquet file or dataset containing the expected schema. If None (default),
    the catalog's _common_metadata file will be used. This schema will be used to verify
    all non-hats columns and (optionally) the file-level metadata. Can be supplied as a
    string or path object."""
    verbose: bool = True
    """Should we output progress and results to standard out?"""
    write_mode: Literal["a", "w", "x"] = "a"
    """Mode to be used when writing the output file. Options have the typical meanings:
        - 'w': truncate the file first
        - 'x': exclusive creation, failing if the file already exists
        - 'a': append to the end of file if it exists"""
    check_metadata: bool = False
    """Whether to check the metadata as well as the schema."""

    @property
    def input_dataset_path(self) -> UPath:
        """Path to the directory under `input_catalog_path` that contains the parquet dataset."""
        return file_io.append_paths_to_pointer(self.input_catalog_path, hats.io.paths.DATASET_DIR)

    @property
    def output_file_path(self) -> UPath:
        """Path to the output file (`output_path` / `output_filename`)."""
        return file_io.append_paths_to_pointer(self.output_path, self.output_filename)

    def __post_init__(self) -> None:
        self.input_catalog_path = file_io.get_upath(self.input_catalog_path)
        self.output_path = file_io.get_upath(self.output_path)

        if self.truth_schema is not None:
            self.truth_schema = file_io.append_paths_to_pointer(self.truth_schema)
            if not self.truth_schema.exists():
                raise FileNotFoundError("truth_schema must be an existing file or directory")
