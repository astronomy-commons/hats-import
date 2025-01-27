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

    args = VerificationArguments(
        input_catalog_path=small_sky_object_catalog, output_path=tmp_path, verbose=False
    )
    verifier = runner.run(args, write_mode="w")
    assert verifier.all_tests_passed, "good catalog failed"
    written_results = pd.read_csv(args.output_path / args.output_filename, comment="#")
    assert written_results[result_cols].equals(verifier.results_df[result_cols]), "report failed"

    args = VerificationArguments(
        input_catalog_path=wrong_files_and_rows_dir, output_path=tmp_path, verbose=False
    )
    verifier = runner.run(args, write_mode="w")
    assert not verifier.all_tests_passed, "bad catalog passed"
    written_results = pd.read_csv(args.output_path / args.output_filename, comment="#")
    assert written_results[result_cols].equals(verifier.results_df[result_cols]), "report failed"


def test_test_file_sets(small_sky_object_catalog, wrong_files_and_rows_dir, tmp_path):
    """File set tests should fail if files listed in _metadata don't match the actual data files."""
    args = VerificationArguments(
        input_catalog_path=small_sky_object_catalog, output_path=tmp_path, verbose=False
    )
    verifier = runner.Verifier.from_args(args)
    passed = verifier.test_file_sets()
    assert passed, "good catalog failed"

    args = VerificationArguments(
        input_catalog_path=wrong_files_and_rows_dir, output_path=tmp_path, verbose=False
    )
    verifier = runner.Verifier.from_args(args)
    passed = verifier.test_file_sets()
    assert not passed, "bad catalog passed"
    expected_bad_file_names = {"Npix=11.extra_file.parquet", "Npix=11.missing_file.parquet"}
    actual_bad_file_names = {
        file_name.split("/")[-1] for file_name in verifier.results_df.bad_files.squeeze()
    }
    assert expected_bad_file_names == actual_bad_file_names, "bad_files failed"


def test_test_is_valid_catalog(small_sky_object_catalog, wrong_files_and_rows_dir, tmp_path):
    """`hats.is_valid_catalog` should pass for good catalogs, fail for catalogs without ancillary files."""
    args = VerificationArguments(
        input_catalog_path=small_sky_object_catalog, output_path=tmp_path, verbose=False
    )
    verifier = runner.Verifier.from_args(args)
    passed = verifier.test_is_valid_catalog()
    assert passed, "good catalog failed"

    args = VerificationArguments(
        input_catalog_path=wrong_files_and_rows_dir, output_path=tmp_path, verbose=False
    )
    verifier = runner.Verifier.from_args(args)
    passed = verifier.test_is_valid_catalog()
    assert not passed, "bad catalog passed"


def test_test_num_rows(small_sky_object_catalog, wrong_files_and_rows_dir, tmp_path):
    """Row count tests should pass if all row counts match, else fail."""
    args = VerificationArguments(
        input_catalog_path=small_sky_object_catalog, output_path=tmp_path, truth_total_rows=131, verbose=False
    )
    verifier = runner.Verifier.from_args(args)
    verifier.test_num_rows()
    assert verifier.all_tests_passed, "good catalog failed"

    args = VerificationArguments(
        input_catalog_path=wrong_files_and_rows_dir, output_path=tmp_path, truth_total_rows=131, verbose=False
    )
    verifier = runner.Verifier.from_args(args)
    verifier.test_num_rows()
    results = verifier.results_df
    all_failed = not results.passed.any()
    assert all_failed, "bad catalog passed"

    targets = {"file footers vs catalog properties", "file footers vs _metadata", "file footers vs truth"}
    assert targets == set(results.target), "wrong targets"

    expected_bad_file_names = {
        "Npix=11.extra_file.parquet",
        "Npix=11.extra_rows.parquet",
        "Npix=11.missing_file.parquet",
    }
    _result = results.loc[results.target == "file footers vs _metadata"].squeeze()
    actual_bad_file_names = {file_name.split("/")[-1] for file_name in _result.bad_files}
    assert expected_bad_file_names == actual_bad_file_names, "wrong bad_files"


@pytest.mark.parametrize("check_metadata", [(False,), (True,)])
def test_test_schemas(small_sky_object_catalog, bad_schemas_dir, tmp_path, check_metadata):
    """Schema tests should pass if all column names, dtypes, and (optionally) metadata match, else fail."""
    # Show that a good catalog passes
    args = VerificationArguments(
        input_catalog_path=small_sky_object_catalog,
        output_path=tmp_path,
        truth_schema=small_sky_object_catalog / "dataset/_common_metadata",
        verbose=False,
    )
    verifier = runner.Verifier.from_args(args)
    verifier.test_schemas(check_metadata=check_metadata)
    assert verifier.all_tests_passed, "good catalog failed"

    # Show that bad schemas fail.
    args = VerificationArguments(
        input_catalog_path=bad_schemas_dir,
        output_path=tmp_path,
        truth_schema=bad_schemas_dir / "dataset/_common_metadata.import_truth",
        verbose=False,
    )
    verifier = runner.Verifier.from_args(args)
    verifier.test_schemas(check_metadata=check_metadata)
    results = verifier.results_df

    # Expecting _common_metadata and some file footers to always fail
    # and _metadata to fail if check_metadata is true.
    expect_failed = ["_common_metadata vs truth", "file footers vs truth"]
    if check_metadata:
        expect_passed = []
        expect_failed = expect_failed + ["_metadata vs truth"]
    else:
        expect_passed = ["_metadata vs truth"]
    assert set(expect_passed + expect_failed) == set(results.target), "wrong targets"
    assert all(results.loc[results.target.isin(expect_passed)].passed), "good targets failed"
    assert not any(results.loc[results.target.isin(expect_failed)].passed), "bad targets passed"

    # Expecting data files with wrong columns or dtypes to always fail
    # and files with wrong metadata to fail if check_metadata is true.
    result = results.loc[results.target == "file footers vs truth"].squeeze()
    expected_bad_files = [
        "Npix=11.extra_column.parquet",
        "Npix=11.missing_column.parquet",
        "Npix=11.wrong_dtypes.parquet",
    ]
    if check_metadata:
        expected_bad_files = expected_bad_files + ["Npix=11.wrong_metadata.parquet"]
    actual_bad_file_names = {file_name.split("/")[-1] for file_name in result.bad_files}
    assert set(expected_bad_files) == set(actual_bad_file_names), "wrong bad_files"
