"""Tests of argument validation"""

import pytest

from hats_import.verification.arguments import VerificationArguments


def test_none():
    """No arguments provided. Should error for required args."""
    with pytest.raises(TypeError):
        VerificationArguments()


def test_empty_required(tmp_path):
    """*Most* required arguments are provided."""
    ## Input path is missing
    with pytest.raises(TypeError, match="input_catalog_path"):
        VerificationArguments(output_path=tmp_path)


def test_invalid_paths(tmp_path, small_sky_object_catalog):
    """Required arguments are provided, but paths aren't found."""
    ## Prove that it works with required args
    VerificationArguments(input_catalog_path=small_sky_object_catalog, output_path=tmp_path)

    ## Input path is not an existing directory
    with pytest.raises(ValueError, match="input_catalog_path must be an existing directory"):
        VerificationArguments(input_catalog_path="path", output_path=f"{tmp_path}/path")


@pytest.mark.timeout(5)
def test_good_paths(tmp_path, small_sky_object_catalog):
    """Required arguments are provided, and paths are found.

    NB: This is currently the last test in alpha-order, and may require additional
    time to teardown fixtures."""
    tmp_path_str = str(tmp_path)
    args = VerificationArguments(input_catalog_path=small_sky_object_catalog, output_path=tmp_path)
    assert args.input_catalog_path == small_sky_object_catalog
    assert str(args.output_path) == tmp_path_str
