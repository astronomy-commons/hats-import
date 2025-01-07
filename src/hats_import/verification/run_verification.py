"""Run pass/fail checks and generate verification report of existing hats table."""

import datetime
import re
from dataclasses import dataclass, field
from typing import Literal

import hats.io.paths
import hats.io.validation
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pds
from hats.pixel_math.spatial_index import SPATIAL_INDEX_COLUMN

from hats_import.verification.arguments import VerificationArguments


def run(
    args: VerificationArguments, check_metadata: bool = False, write_mode: Literal["a", "w", "x"] = "a"
) -> "Verifier":
    """Create a `Verifier` using `args`, run all tests, and write a verification report.

    Parameters
    ----------
    args : VerificationArguments
        Arguments to construct the Verifier.
    check_metadata : bool, optional
        Whether to check the metadata along with the schema. Default is False.
    write_mode : Literal["a", "w", "x"], optional
        Mode to be used when writing output files. Default is "a" (append).

    Returns
    -------
    Verifier
        The `Verifier` instance used to perform the tests. The `results_df` property contains
        the same information as written to the output report.

    Raises
    ------
    TypeError
        If `args` is not provided or is not an instance of `VerificationArguments`.
    """
    if not args:
        raise TypeError("args is required and should be type VerificationArguments")
    if not isinstance(args, VerificationArguments):
        raise TypeError("args must be type VerificationArguments")

    verifier = Verifier.from_args(args)
    verifier.run(write_mode=write_mode, check_metadata=check_metadata)

    return verifier


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
class Verifier:
    """Run verification tests. To create an instance of this class, use `Verifier.from_args`."""

    args: VerificationArguments = field()
    """Arguments to use during verification."""
    metadata_ds: pds.Dataset = field()
    """Pyarrow dataset, loaded from the _metadata file."""
    files_ds: pds.Dataset = field()
    """Pyarrow dataset, loaded from the parquet data files."""
    common_metadata_schema: pa.Schema = field()
    """Pyarrow schema, loaded from the _common_metadata file."""
    constructed_truth_schema: pa.Schema = field()
    """Pyarrow schema treated as truth during verification. This is constructed
    from `common_metadata_schema` and `args.truth_schema`. `common_metadata_schema`
    is used for hats-specific columns. If provided, `args.truth_schema` is used
    for all other columns plus metadata, otherwise `common_metadata_schema` is used.
    """
    results: list[Result] = field(default_factory=list)
    """List of results, one for each test that has been done."""

    @classmethod
    def from_args(cls, args: VerificationArguments) -> "Verifier":
        """Create a `Verifier` with initialized datasets and schemas based on `args`.

        Parameters
        ----------
        args : VerificationArguments
            Arguments for the Verifier.

        Returns
        -------
        Verifier
        """
        args.output_path.mkdir(exist_ok=True, parents=True)

        print("Loading dataset and schema.")
        parquet_fs = args.input_catalog_path.fs
        files_ds = pds.dataset(args.input_dataset_path.path, filesystem=parquet_fs)
        metadata_ds = pds.parquet_dataset(
            hats.io.paths.get_parquet_metadata_pointer(args.input_catalog_path), filesystem=parquet_fs
        )

        input_truth_schema = None
        if args.truth_schema is not None:
            input_truth_schema = pds.parquet_dataset(args.truth_schema, filesystem=parquet_fs).schema
        common_metadata_schema = pds.parquet_dataset(
            hats.io.paths.get_common_metadata_pointer(args.input_catalog_path), filesystem=parquet_fs
        ).schema
        constructed_truth_schema = cls._construct_truth_schema(
            input_truth_schema=input_truth_schema, common_metadata_schema=common_metadata_schema
        )

        return cls(
            args=args,
            metadata_ds=metadata_ds,
            files_ds=files_ds,
            common_metadata_schema=common_metadata_schema,
            constructed_truth_schema=constructed_truth_schema,
        )

    @property
    def results_df(self) -> pd.DataFrame:
        """Test results as a dataframe."""
        return pd.DataFrame(self.results)

    def run(self, write_mode: Literal["a", "w", "x"] = "a", check_metadata: bool = False) -> None:
        """Run all tests and write a verification report. See `results_df` property or
        written report for results.

        Parameters
        ----------
        write_mode : Literal["a", "w", "x"], optional
            Mode to be used when writing output files.
        check_metadata : bool, optional
            Whether to check the metadata as well as the schema.
        """
        self.test_is_valid_catalog()
        self.test_file_sets()
        self.test_num_rows()
        self.test_schemas(check_metadata=check_metadata)

        self.write_results(write_mode=write_mode)

    def test_is_valid_catalog(self) -> bool:
        """Test if the provided catalog is a valid HATS catalog. Add one `Result` to `results`.

        Returns
        -------
            bool: True if the test passed, else False.
        """
        version = f"hats version {hats.__version__}"
        test, description = "valid hats", f"Test hats.io.validation.is_valid_catalog ({version})."
        target = self.args.input_catalog_path
        print(f"\nStarting: {description}")

        passed = hats.io.validation.is_valid_catalog(target, strict=True)
        self.results.append(Result(test=test, description=description, passed=passed, target=target.name))
        print(f"Result: {'PASSED' if passed else 'FAILED'}")
        return passed

    def test_file_sets(self) -> bool:
        """Test that files in _metadata match the parquet files on disk. Add one `Result` to `results`.

        This is a simple test that can be especially useful to run after copying or moving
        the catalog to a different local or cloud-based destination.

        Returns
        -------
            bool: True if the file sets match, else False.
        """
        # info for the report
        description = "Test that files in _metadata match the data files on disk."
        print(f"\nStarting: {description}")

        files_ds_files = self._relative_paths(self.files_ds.files)
        metadata_ds_files = self._relative_paths(self.metadata_ds.files)
        failed_files = list(set(files_ds_files).symmetric_difference(metadata_ds_files))
        passed = len(failed_files) == 0
        self.results.append(
            Result(
                passed=passed,
                test="file sets",
                target="_metadata vs data files",
                bad_files=failed_files,
                description=description,
            )
        )

        print(f"Result: {'PASSED' if passed else 'FAILED'}")
        return passed

    def test_num_rows(self) -> bool:
        """Test the number of rows in the dataset. Add `Result`s to `results`.

        File footers are compared with _metadata and the user-supplied truth (if provided).

        Returns
        -------
        bool: True if all checks pass, else False.
        """
        test = "num rows"
        description = "Test that number of rows are equal."
        print(f"\nStarting: {description}")

        # get the number of rows in each file, indexed by file path. we treat this as truth.
        files_df = self._load_nrows(self.files_ds)
        files_df_total = f"file footers ({files_df.num_rows.sum():,})"

        # check _metadata
        target = "file footers vs _metadata"
        print(f"\t{target}")
        metadata_df = self._load_nrows(self.metadata_ds)
        row_diff = files_df - metadata_df
        failed_frags = row_diff.loc[row_diff.num_rows != 0].index.to_list()
        passed_md = len(failed_frags) == 0
        _description = f" {files_df_total} vs _metadata ({metadata_df.num_rows.sum():,})."
        self.results.append(
            Result(
                passed=passed_md,
                test=test,
                bad_files=failed_frags,
                target=target,
                description=description + _description,
            )
        )

        # check user-supplied total, if provided
        if self.args.truth_total_rows is not None:
            target = "file footers vs truth"
            print(f"\t{target}")
            passed_th = self.args.truth_total_rows == files_df.num_rows.sum()
            _description = f" {files_df_total} vs user-provided truth ({self.args.truth_total_rows:,})."
            self.results.append(
                Result(passed=passed_th, test=test, target=target, description=description + _description)
            )
        else:
            passed_th = True  # this test did not fail. this is only needed for the return value.

        all_passed = all([passed_md, passed_th])
        print(f"Result: {'PASSED' if all_passed else 'FAILED'}")
        return all_passed

    def _load_nrows(self, dataset: pds.Dataset) -> pd.DataFrame:
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
        frag_names = self._relative_paths([frag.path for frag in dataset.get_fragments()])
        nrows_df = pd.DataFrame({"num_rows": num_rows, "frag_path": frag_names})
        nrows_df = nrows_df.set_index("frag_path").sort_index()
        return nrows_df

    def test_schemas(self, check_metadata: bool = False) -> bool:
        """Test the equality of schemas. Add `Result`s to `results`.

        This performs three tests:
        1. `common_metadata_schema` vs `constructed_truth_schema`.
        2. `metadata_ds.schema` vs `constructed_truth_schema`.
        3. File footers vs `constructed_truth_schema`.

        Parameters
        ----------
        check_metadata : bool, optional
            Whether to check the metadata as well as the schema.

        Returns
        -------
        bool: True if all tests pass, else False.
        """
        # info for the report
        _include_md = "including metadata" if check_metadata else "excluding metadata"
        test_info = {"test": "schema", "description": f"Test that schemas are equal, {_include_md}."}
        print(f"\nStarting: {test_info['description']}")

        passed_cm = self._test_schema__common_metadata(test_info, check_metadata=check_metadata)
        passed_md = self._test_schema__metadata(test_info, check_metadata=check_metadata)
        passed_ff = self._test_schema_file_footers(test_info, check_metadata=check_metadata)

        all_passed = all([passed_cm, passed_md, passed_ff])
        print(f"Result: {'PASSED' if all_passed else 'FAILED'}")
        return all_passed

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
        hats_partition_fields = [common_metadata_schema.field(fld) for fld in hats_cols]
        hats_idx_fields = []
        if SPATIAL_INDEX_COLUMN in common_metadata_schema.names:
            hats_cols.append(SPATIAL_INDEX_COLUMN)
            hats_idx_fields.append(common_metadata_schema.field(SPATIAL_INDEX_COLUMN))
        input_truth_fields = [fld for fld in input_truth_schema if fld.name not in hats_cols]

        constructed_fields = hats_idx_fields + input_truth_fields + hats_partition_fields
        constructed_schema = pa.schema(constructed_fields).with_metadata(input_truth_schema.metadata)
        return constructed_schema

    def _test_schema__common_metadata(self, test_info: dict, check_metadata: bool = False) -> bool:
        """Test `common_metadata_schema` against `constructed_truth_schema`.

        Parameters
        ----------
        test_info : dict
            Information about this test for the reported results.
        check_metadata : bool, optional
            Whether to check the metadata as well as the schema.

        Returns
        -------
        bool: True if all tests pass, else False.
        """
        targets = "_common_metadata vs truth"
        print(f"\t{targets}")
        passed = self.common_metadata_schema.equals(
            self.constructed_truth_schema, check_metadata=check_metadata
        )
        self.results.append(
            Result(
                passed=passed, target=targets, test=test_info["test"], description=test_info["description"]
            )
        )
        return passed

    def _test_schema__metadata(self, test_info: dict, check_metadata: bool = False) -> bool:
        """Test _metadata schema against the truth schema.

        Parameters
        ----------
        test_info : dict
            Information about this test for the reported results.
        check_metadata : bool, optional
            Whether to check the metadata as well as the schema.

        Returns
        -------
        bool: True if both schema and metadata match the truth source, else False.
        """
        targets = "_metadata vs truth"
        print(f"\t{targets}")
        passed = self.metadata_ds.schema.equals(self.constructed_truth_schema, check_metadata=check_metadata)
        self.results.append(
            Result(
                passed=passed, target=targets, test=test_info["test"], description=test_info["description"]
            )
        )
        return passed

    def _test_schema_file_footers(self, test_info: dict, check_metadata: bool = False) -> bool:
        """Test the file footers schema and metadata against the truth schema.

        Parameters
        ----------
        test_info : dict
            Information about this test for the reported results.
        check_metadata : bool, optional
            Whether to check the metadata as well as the schema.

        Returns
        -------
        bool: True if all schema and metadata tests pass, else False.
        """
        targets = "file footers vs truth"
        print(f"\t{targets}")

        bad_files = []
        for frag in self.files_ds.get_fragments():
            if not frag.physical_schema.equals(self.constructed_truth_schema, check_metadata=check_metadata):
                bad_files.append(frag.path)
        bad_files = self._relative_paths(bad_files)

        passed = len(bad_files) == 0
        self.results.append(
            Result(
                passed=passed,
                target=targets,
                bad_files=bad_files,
                test=test_info["test"],
                description=test_info["description"],
            )
        )
        return passed

    def _relative_paths(self, absolute_paths):
        """Find the relative path for dataset parquet files,
        assuming a pattern like <base_path>/Norder=d/Dir=d/Npix=d"""
        relative_path_pattern = re.compile(r".*(Norder.*)")
        relative_paths = [str(relative_path_pattern.match(file).group(1)) or file for file in absolute_paths]
        return relative_paths

    def write_results(self, *, write_mode: Literal["a", "w", "x"] = "a") -> None:
        """Write the verification results to file at `args.output_path` / `args.output_filename`.

        Parameters
        ----------
        write_mode : Literal["a", "w", "x"], optional
            Mode to be used when writing output file. Passed to pandas.DataFrame.to_csv as `mode`.
        """
        self.args.output_file_path.parent.mkdir(exist_ok=True, parents=True)
        header = not (write_mode == "a" and self.args.output_file_path.exists())
        self.results_df.to_csv(self.args.output_file_path, mode=write_mode, header=header, index=False)
        print(f"\nVerifier results written to {self.args.output_file_path}")
