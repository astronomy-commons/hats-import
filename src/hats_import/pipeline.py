"""Flow control and pipeline entry points."""

import smtplib
from email.message import EmailMessage

from dask.distributed import Client

import hats_import.catalog.run_import as catalog_runner
import hats_import.collection.run_import as collection_runner
import hats_import.hipscat_conversion.run_conversion as conversion_runner
import hats_import.index.run_index as index_runner
import hats_import.margin_cache.margin_cache as margin_runner
import hats_import.verification.run_verification as verification_runner
from hats_import.catalog.arguments import ImportArguments
from hats_import.collection.arguments import CollectionArguments
from hats_import.hipscat_conversion.arguments import ConversionArguments
from hats_import.index.arguments import IndexArguments
from hats_import.margin_cache.margin_cache_arguments import MarginCacheArguments
from hats_import.runtime_arguments import RuntimeArguments
from hats_import.verification.arguments import VerificationArguments

# pragma: no cover


def pipeline(args: RuntimeArguments):
    """Pipeline that creates its own client from the provided runtime arguments"""
    if isinstance(args, VerificationArguments):
        verification_runner.run(args)
        return
    with Client(
        local_directory=args.dask_tmp,
        n_workers=args.dask_n_workers,
        threads_per_worker=args.dask_threads_per_worker,
    ) as client:
        pipeline_with_client(args, client)


def pipeline_with_client(args: RuntimeArguments, client: Client):
    """Pipeline that is run using an existing client.

    This can be useful in tests, or when a dask client requires some more complex
    configuration.
    """
    if isinstance(args, VerificationArguments):
        verification_runner.run(args)
        return
    try:
        if not args:
            raise ValueError("args is required and should be subclass of RuntimeArguments")

        if isinstance(args, ImportArguments):
            catalog_runner.run(args, client)
        elif isinstance(args, IndexArguments):
            index_runner.run(args, client)
        elif isinstance(args, MarginCacheArguments):
            margin_runner.generate_margin_cache(args, client)
        elif isinstance(args, CollectionArguments):
            collection_runner.run(args, client)
        elif isinstance(args, VerificationArguments):
            verification_runner.run(args)
        elif isinstance(args, ConversionArguments):
            conversion_runner.run(args, client)
        else:
            raise ValueError("unknown args type")

        _send_success_email(args)
    except Exception as exception:  # pylint: disable=broad-exception-caught
        if args.completion_email_address:
            _send_failure_email(args, exception)
        raise exception


def _send_failure_email(args: RuntimeArguments, exception: Exception):
    message = EmailMessage()
    message["Subject"] = "hats-import failure."
    message["To"] = args.completion_email_address
    message.set_content(
        f"output_artifact_name: {args.output_artifact_name}"
        "\n\nSee logs for more details"
        f"\n\nFailed with message:\n\n{exception}"
    )

    _send_email(message)


def _send_success_email(args):
    if not args.completion_email_address:
        return
    message = EmailMessage()
    message["Subject"] = "hats-import success."
    message["To"] = args.completion_email_address
    message.set_content(f"output_artifact_name: {args.output_artifact_name}")

    _send_email(message)


def _send_email(message: EmailMessage):
    message["From"] = "updates@lsdb.io"

    # Send the message via our own SMTP server.
    with smtplib.SMTP("localhost") as server:
        server.send_message(message)
