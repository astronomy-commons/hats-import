import pandas as pd
import pytest

import hats_import.verification.run_verification as runner
from hats_import.verification.arguments import VerificationArguments


def test_bad_args():
    """Runner should fail with empty or mis-typed arguments"""
    with pytest.raises(TypeError, match="VerificationArguments"):
        runner.run(None)

    args = {"output_artifact_name": "bad_arg_type"}
    with pytest.raises(TypeError, match="VerificationArguments"):
        runner.run(args)


def test_runner(small_sky_object_catalog, wrong_files_and_rows_dir, tmp_path):
    """Runner should execute all tests and write a report to file."""
    result_cols = ["datetime", "passed", "test", "target"]

    args = VerificationArguments(input_catalog_path=small_sky_object_catalog, output_path=tmp_path)
    verifier = runner.run(args, write_mode="w")
    all_passed = verifier.results_df.passed.all()
    assert all_passed, "valid catalog failed"
    # # [FIXME] pandas metadata is unexpectedly missing hats columns
    # if not all_passed:
    #     _test = verifier.results_df.test == "schema consistency"
    #     _target = verifier.results_df.target == "constructed_truth_schema"
    #     assert verifier.results_df.loc[~(_test & _target)].passed.all()
    written_results = pd.read_csv(args.output_path / args.output_filename)
    assert written_results[result_cols].equals(verifier.results_df[result_cols]), "report failed"

    args = VerificationArguments(input_catalog_path=wrong_files_and_rows_dir, output_path=tmp_path)
    verifier = runner.run(args, write_mode="w")
    assert not verifier.results_df.passed.all(), "invalid catalog passed"
    written_results = pd.read_csv(args.output_path / args.output_filename)
    assert written_results[result_cols].equals(verifier.results_df[result_cols]), "report failed"


def test_test_file_sets(small_sky_object_catalog, wrong_files_and_rows_dir, tmp_path):
    """File set tests should fail if files listed in _metadata don't match the actual data files."""
    args = VerificationArguments(input_catalog_path=small_sky_object_catalog, output_path=tmp_path)
    verifier = runner.Verifier.from_args(args)
    passed = verifier.test_file_sets()
    assert passed, "valid catalog failed"

    args = VerificationArguments(input_catalog_path=wrong_files_and_rows_dir, output_path=tmp_path)
    verifier = runner.Verifier.from_args(args)
    passed = verifier.test_file_sets()
    assert not passed, "invalid catalog passed"
    bad_files = {"Norder=0/Dir=0/Npix=11.extra_file.parquet", "Norder=0/Dir=0/Npix=11.missing_file.parquet"}
    assert bad_files == set(verifier.results_df.bad_files.squeeze()), "bad_files failed"


def test_test_is_valid_catalog(small_sky_object_catalog, wrong_files_and_rows_dir, tmp_path):
    """`hats.is_valid_catalog` should pass for valid catalogs, fail for catalogs without ancillary files."""
    args = VerificationArguments(input_catalog_path=small_sky_object_catalog, output_path=tmp_path)
    verifier = runner.Verifier.from_args(args)
    passed = verifier.test_is_valid_catalog()
    assert passed, "valid catalog failed"

    args = VerificationArguments(input_catalog_path=wrong_files_and_rows_dir, output_path=tmp_path)
    verifier = runner.Verifier.from_args(args)
    passed = verifier.test_is_valid_catalog()
    assert not passed, "invalid catalog passed"


def test_test_num_rows(small_sky_object_catalog, wrong_files_and_rows_dir, tmp_path):
    """Row count tests should pass if all row counts match, else fail."""
    args = VerificationArguments(
        input_catalog_path=small_sky_object_catalog, output_path=tmp_path, truth_total_rows=131
    )
    verifier = runner.Verifier.from_args(args)
    verifier.test_num_rows()
    all_passed = verifier.results_df.passed.all()
    assert all_passed, "valid catalog failed"

    args = VerificationArguments(
        input_catalog_path=wrong_files_and_rows_dir, output_path=tmp_path, truth_total_rows=131
    )
    verifier = runner.Verifier.from_args(args)
    verifier.test_num_rows()
    results = verifier.results_df
    all_failed = not results.passed.any()
    assert all_failed, "invalid catalog passed"

    targets = {"file footers vs _metadata", "file footers vs truth"}
    assert targets == set(results.target), "wrong targets"

    bad_files = {
        "Norder=0/Dir=0/Npix=11.extra_file.parquet",
        "Norder=0/Dir=0/Npix=11.extra_rows.parquet",
        "Norder=0/Dir=0/Npix=11.missing_file.parquet",
    }
    _result = results.loc[results.target == "file footers vs _metadata"].squeeze()
    assert bad_files == set(_result.bad_files), "wrong bad_files"


@pytest.mark.parametrize("check_metadata", [(False,), (True,)])
def test_test_schemas(small_sky_object_catalog, bad_schemas_dir, tmp_path, check_metadata):
    """Schema tests should pass if all column names, dtypes, and (optionally) metadata match, else fail."""
    args = VerificationArguments(
        input_catalog_path=small_sky_object_catalog,
        output_path=tmp_path,
        truth_schema=small_sky_object_catalog / "dataset/_common_metadata",
    )
    verifier = runner.Verifier.from_args(args)
    verifier.test_schemas(check_metadata=check_metadata)
    all_passed = verifier.results_df.passed.all()
    assert all_passed, "valid catalog failed"
    # # [FIXME] pandas metadata is unexpectedly missing hats columns
    # if not all_passed:
    #     _test = verifier.results_df.test == "schema consistency"
    #     _target = verifier.results_df.target == "constructed_truth_schema"
    #     assert verifier.results_df.loc[~(_test & _target)].passed.all()

    args = VerificationArguments(
        input_catalog_path=bad_schemas_dir,
        output_path=tmp_path,
        truth_schema=bad_schemas_dir / "dataset/_common_metadata.import_truth",
    )
    verifier = runner.Verifier.from_args(args)
    verifier.test_schemas(check_metadata=check_metadata)
    results = verifier.results_df
    all_failed = not any(results.passed)
    assert all_failed, "invalid catalog passed"

    targets_failed = {"constructed_truth_schema", "_common_metadata vs truth", "file footers vs truth"}
    if not check_metadata:
        targets_passed = {"_metadata vs truth"}
    else:
        targets_passed = set()
        targets_failed = targets_failed.union({"_metadata vs truth"})
    assert targets_passed.union(targets_failed) == set(results.target), "wrong targets"
    assert all(results.loc[results.target.isin(targets_passed)].passed), "valid targets failed"
    assert not any(results.loc[results.target.isin(targets_failed)].passed), "invalid targets passed"

    target = "file footers vs truth"
    result = results.loc[results.target == target].squeeze()
    expected_bad_files = {
        "Norder=0/Dir=0/Npix=11.extra_column.parquet",
        "Norder=0/Dir=0/Npix=11.missing_column.parquet",
        "Norder=0/Dir=0/Npix=11.wrong_dtypes.parquet",
    }
    if check_metadata:
        expected_bad_files = expected_bad_files.union({"Norder=0/Dir=0/Npix=11.no_metadata.parquet"})
    assert expected_bad_files == set(result.bad_files), "wrong bad_files"
