import hats_import


def test_version():
    """Check to see that we can get the package version"""
    assert hats_import.__version__ is not None
