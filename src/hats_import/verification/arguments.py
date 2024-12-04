"""Utility to hold all arguments required throughout verification pipeline"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from upath import UPath


@dataclass(kw_only=True)
class VerificationArguments:
    """Container for verification arguments."""

    input_catalog_path: str | Path | UPath = field()
    """Path to an existing catalog that will be inspected. This must be a directory
    containing (at least) the hats ancillary files and a 'dataset/' directory
    containing the parquet dataset."""
    output_path: str | Path | UPath = field()
    """Directory where the verification report should be written."""
    output_filename: str = field(default="verifier_results.csv")
    """Filename for the verification report."""
    truth_total_rows: int | None = field(default=None)
    """Total number of rows expected in this catalog."""
    truth_schema: str | Path | UPath | None = field(default=None)
    """Path to a parquet file or dataset containing the expected schema. If None (default),
    the catalog's _common_metadata file will be used. This schema will be used to verify
    the catalog's column names and data types for all non-hats columns. It will be
    ignored when verifying all hats-specific columns and all metadata."""

    @property
    def input_dataset_path(self) -> UPath:
        """Directory containing the parquet dataset associated with `input_catalog_path`."""
        return self.input_catalog_path / "dataset"

    def __post_init__(self) -> None:
        self.input_catalog_path = UPath(self.input_catalog_path)
        if not self.input_catalog_path.is_dir():
            raise ValueError("input_catalog_path must be an existing directory")

        self.output_path = UPath(self.output_path)

        if self.truth_schema is not None:
            self.truth_schema = UPath(self.truth_schema)
            if not self.truth_schema.exists():
                raise ValueError("truth_schema must be an existing file or directory")
