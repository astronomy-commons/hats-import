"""Run pass/fail checks and generate verification report of existing hats table."""

import datetime
import re
from dataclasses import dataclass, field

import hats
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pds
from hats.pixel_math.spatial_index import SPATIAL_INDEX_COLUMN

import hats_import
from hats_import.verification.arguments import VerificationArguments


def now() -> str:
    """Get the current time as a string."""
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y/%m/%d %H:%M:%S %Z")


@dataclass(kw_only=True, frozen=True)
class Result:
    """Verification test result for a single test."""

    datetime: str = field(default_factory=now)
    """The date and time when the test was run."""
    passed: bool = field()
    """Whether the test passed."""
    test: str = field()
    """Test name."""
    target: str = field()
    """The file(s) targeted by the test."""
    description: str = field()
    """Test description."""
    bad_files: list[str] = field(default_factory=list)
    """List of additional files that caused the test to fail (empty if none or not applicable)."""


@dataclass(kw_only=True)
class BaseVerifier:
    """Run verification tests. To create an instance of this class, use `Verifier.from_args`."""

    args: VerificationArguments = field()
    """Arguments to use during verification."""
    results: list[Result] = field(default_factory=list)
    """List of results, one for each test that has been done."""

    @property
    def results_df(self) -> pd.DataFrame:
        """Test results as a dataframe."""
        return pd.DataFrame(self.results)

    @property
    def all_tests_passed(self):
        """Simple pass/fail if all of the test results have passed."""
        return np.all([res.passed for res in self.results])

    @staticmethod
    def _load_nrows(dataset: pds.Dataset) -> pd.DataFrame:
        """Load the number of rows in each file in the dataset.

        Parameters
        ----------
        dataset : pyarrow.dataset.Dataset
            The dataset from which to load the number of rows.

        Returns
        -------
        pd.DataFrame: A DataFrame with the number of rows per file, indexed by file path.
        """
        num_rows = [frag.metadata.num_rows for frag in dataset.get_fragments()]
        frag_names = BaseVerifier._relative_paths([frag.path for frag in dataset.get_fragments()])
        nrows_df = pd.DataFrame({"num_rows": num_rows, "frag_path": frag_names})
        nrows_df = nrows_df.set_index("frag_path").sort_index()
        return nrows_df

    @staticmethod
    def _construct_truth_schema(
        *, input_truth_schema: pa.Schema | None, common_metadata_schema: pa.Schema
    ) -> pa.Schema:
        """Copy of `input_truth_schema` with HATS fields added from `common_metadata_schema`.

        If `input_truth_schema` is not provided, this is just `common_metadata_schema`.

        Parameters
        ----------
        input_truth_schema : pyarrow.Schema or None
            The input truth schema, if provided.
        common_metadata_schema : pyarrow.Schema
            The common metadata schema.

        Returns
        -------
        pyarrow.Schema
            The constructed truth schema.
        """
        if input_truth_schema is None:
            return common_metadata_schema

        hats_cols = ["Norder", "Dir", "Npix"]
        hats_idx_fields = []
        if SPATIAL_INDEX_COLUMN in common_metadata_schema.names:
            hats_cols.append(SPATIAL_INDEX_COLUMN)
            hats_idx_fields.append(common_metadata_schema.field(SPATIAL_INDEX_COLUMN))
        input_truth_fields = [fld for fld in input_truth_schema if fld.name not in hats_cols]

        constructed_fields = hats_idx_fields + input_truth_fields
        constructed_schema = pa.schema(constructed_fields).with_metadata(input_truth_schema.metadata)
        return constructed_schema

    @staticmethod
    def _relative_paths(absolute_paths):
        """Find the relative path for dataset parquet files,
        assuming a pattern like <base_path>/Norder=d/Dir=d/Npix=d"""
        relative_path_pattern = re.compile(r".*(Norder.*)")
        relative_paths = [str(relative_path_pattern.match(file).group(1)) or file for file in absolute_paths]
        return relative_paths

    def write_results(self) -> None:
        """Write the verification results to file at `args.output_path` / `args.output_filename`."""
        self.args.output_file_path.parent.mkdir(exist_ok=True, parents=True)
        # Write provenance info
        with open(self.args.output_file_path, self.args.write_mode, encoding="utf8") as fout:
            fout.writelines(
                [
                    "# HATS verification results for\n",
                    f"# {self.args.input_catalog_path}\n",
                    f"# Package versions: hats v{hats.__version__}; hats-import v{hats_import.__version__}\n",
                    f"# User-supplied truth schema: {self.args.truth_schema}\n",
                    f"# User-supplied truth total rows: {self.args.truth_total_rows}\n",
                ]
            )
        # Write results
        self.results_df.to_csv(self.args.output_file_path, mode="a", header=True, index=False)
        self.print_if_verbose(f"\nVerifier results written to {self.args.output_file_path}")

    def print_if_verbose(self, message):
        """If the args.verbose=True flag is enabled, print to standard out. Otherwise, no operation."""
        if self.args.verbose:
            print(message)
