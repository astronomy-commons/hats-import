"""Run pass/fail checks and generate verification report of existing hats table."""

from __future__ import annotations

from dataclasses import dataclass, field
from urllib.parse import unquote

import hats
import hats.io.paths
import hats.io.validation
import pyarrow as pa
import pyarrow.dataset as pds
from hats import read_hats

from hats_import.verification.arguments import VerificationArguments
from hats_import.verification.common import BaseVerifier, Result


@dataclass(kw_only=True)
class Verifier(BaseVerifier):
    """Run verification tests. To create an instance of this class, use `Verifier.from_args`."""

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

    @classmethod
    def from_args(cls, args: VerificationArguments) -> Verifier:
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

        if args.verbose:
            print("Loading dataset and schema.")
        parquet_fs = args.input_catalog_path.fs

        ## Fetch all sub-URLs that could contain hats leaf files.
        all_files = []
        for child in args.input_dataset_path.rglob("Norder*/**/*"):
            if not child.is_dir():
                all_files.append(unquote(child.path))

        files_ds = pds.dataset(all_files, filesystem=parquet_fs)
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

    def run(self) -> None:
        """Run all tests and write a verification report. See `results_df` property or
        written report for results.
        """
        self.test_is_valid_catalog()
        self.test_file_sets()
        self.test_num_rows()
        self.test_schemas()

    def test_is_valid_catalog(self) -> bool:
        """Test if the provided catalog is a valid HATS catalog. Add one `Result` to `results`.

        Returns
        -------
            bool: True if the test passed, else False.
        """
        test, description = "valid hats", "Test hats.io.validation.is_valid_catalog."
        target = self.args.input_catalog_path
        self.print_if_verbose(f"\nStarting: {description}")

        passed = hats.io.validation.is_valid_catalog(target, strict=True, verbose=self.args.verbose)
        self.results.append(Result(test=test, description=description, passed=passed, target=target.name))
        self.print_if_verbose(f"Result: {'PASSED' if passed else 'FAILED'}")
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
        self.print_if_verbose(f"\nStarting: {description}")

        files_ds_files = BaseVerifier._relative_paths(self.files_ds.files)
        metadata_ds_files = BaseVerifier._relative_paths(self.metadata_ds.files)
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

        self.print_if_verbose(f"Result: {'PASSED' if passed else 'FAILED'}")
        return passed

    def test_num_rows(self) -> bool:
        """Test the number of rows in the dataset. Add `Result`s to `results`.

        Row counts in parquet file footers are compared with the '_metadata' file,
        HATS 'properties' file, and (if provided) the user-supplied truth.

        Returns
        -------
        bool: True if all checks pass, else False.
        """
        test = "num rows"
        description = "Test that number of rows are equal."
        self.print_if_verbose(f"\nStarting: {description}")

        catalog = read_hats(self.args.input_catalog_path)

        catalog_prop_len = catalog.catalog_info.total_rows

        # get the number of rows in each file, indexed by file path. we treat this as truth.
        files_df = self._load_nrows(self.files_ds)
        files_df_sum = files_df.num_rows.sum()
        files_df_total = f"file footers ({files_df_sum:,})"

        target = "file footers vs catalog properties"
        self.print_if_verbose(f"\t{target}")
        passed_cat = catalog_prop_len == files_df_sum
        _description = f" {files_df_total} vs catalog properties ({catalog_prop_len:,})."
        self.results.append(
            Result(passed=passed_cat, test=test, target=target, description=description + _description)
        )

        # check _metadata
        target = "file footers vs _metadata"
        self.print_if_verbose(f"\t{target}")
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
            self.print_if_verbose(f"\t{target}")
            passed_th = self.args.truth_total_rows == files_df_sum
            _description = f" {files_df_total} vs user-provided truth ({self.args.truth_total_rows:,})."
            self.results.append(
                Result(passed=passed_th, test=test, target=target, description=description + _description)
            )
        else:
            passed_th = True  # this test did not fail. this is only needed for the return value.

        all_passed = all([passed_md, passed_th, passed_cat])
        self.print_if_verbose(f"Result: {'PASSED' if all_passed else 'FAILED'}")
        return all_passed

    def test_schemas(self) -> bool:
        """Test the equality of schemas. Add `Result`s to `results`.

        This performs three tests:
        1. `common_metadata_schema` vs `constructed_truth_schema`.
        2. `metadata_ds.schema` vs `constructed_truth_schema`.
        3. File footers vs `constructed_truth_schema`.

        Returns
        -------
        bool: True if all tests pass, else False.
        """
        # info for the report
        _include_md = "including metadata" if self.args.check_metadata else "excluding metadata"
        test_info = {"test": "schema", "description": f"Test that schemas are equal, {_include_md}."}
        self.print_if_verbose(f"\nStarting: {test_info['description']}")

        passed_cm = self._test_schema__common_metadata(test_info)
        passed_md = self._test_schema__metadata(test_info)
        passed_ff = self._test_schema_file_footers(test_info)

        all_passed = all([passed_cm, passed_md, passed_ff])
        self.print_if_verbose(f"Result: {'PASSED' if all_passed else 'FAILED'}")
        return all_passed

    def _test_schema__common_metadata(self, test_info: dict) -> bool:
        """Test `common_metadata_schema` against `constructed_truth_schema`.

        Parameters
        ----------
        test_info : dict
            Information about this test for the reported results.

        Returns
        -------
        bool: True if all tests pass, else False.
        """
        targets = "_common_metadata vs truth"
        self.print_if_verbose(f"\t{targets}")
        passed = self.common_metadata_schema.equals(
            self.constructed_truth_schema, check_metadata=self.args.check_metadata
        )
        self.results.append(
            Result(
                passed=passed, target=targets, test=test_info["test"], description=test_info["description"]
            )
        )
        return passed

    def _test_schema__metadata(self, test_info: dict) -> bool:
        """Test _metadata schema against the truth schema.

        Parameters
        ----------
        test_info : dict
            Information about this test for the reported results.

        Returns
        -------
        bool: True if both schema and metadata match the truth source, else False.
        """
        targets = "_metadata vs truth"
        self.print_if_verbose(f"\t{targets}")
        passed = self.metadata_ds.schema.equals(
            self.constructed_truth_schema, check_metadata=self.args.check_metadata
        )
        self.results.append(
            Result(
                passed=passed, target=targets, test=test_info["test"], description=test_info["description"]
            )
        )
        return passed

    def _test_schema_file_footers(self, test_info: dict) -> bool:
        """Test the file footers schema and metadata against the truth schema.

        Parameters
        ----------
        test_info : dict
            Information about this test for the reported results.

        Returns
        -------
        bool: True if all schema and metadata tests pass, else False.
        """
        targets = "file footers vs truth"
        self.print_if_verbose(f"\t{targets}")

        bad_files = []
        for frag in self.files_ds.get_fragments():
            if not frag.physical_schema.equals(
                self.constructed_truth_schema, check_metadata=self.args.check_metadata
            ):
                bad_files.append(frag.path)
        bad_files = BaseVerifier._relative_paths(bad_files)

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
