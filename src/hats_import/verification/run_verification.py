"""Run pass/fail checks and generate verification report of existing hats table."""

from time import perf_counter

from hats import read_hats
from hats.catalog import AssociationCatalog, Catalog, CatalogCollection, IndexCatalog, MarginCatalog

from hats_import.verification.arguments import VerificationArguments
from hats_import.verification.catalog_verifier import Verifier


# pylint: disable=too-many-lines
def run(args: VerificationArguments) -> Verifier:
    """Create a `Verifier` using `args`, run all tests, and write a verification report.

    Parameters
    ----------
    args : VerificationArguments
        Arguments to construct the Verifier.

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

    start = perf_counter()
    hats_catalog = read_hats(args.input_catalog_path)
    if isinstance(hats_catalog, CatalogCollection):
        raise NotImplementedError("Cannot verify catalog collection")
    elif isinstance(hats_catalog, Catalog):
        verifier = Verifier.from_args(args)
        verifier.run()
        verifier.write_results()
    elif isinstance(hats_catalog, MarginCatalog):
        raise NotImplementedError("Cannot verify margin catalog")
    elif isinstance(hats_catalog, IndexCatalog):
        raise NotImplementedError("Cannot verify index catalog")
    elif isinstance(hats_catalog, AssociationCatalog):
        raise NotImplementedError("Cannot verify association catalog")

    if args.verbose:
        print(f"Elapsed time (seconds): {perf_counter()-start:.2f}")

    return verifier
