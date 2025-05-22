import pytest

import hats_import


def test_runner(small_sky_object_catalog, tmp_path):
    """Runner should execute all tests and write a report to file."""
    args = hats_import.VerificationArguments(
        input_catalog_path=small_sky_object_catalog, output_path=tmp_path, verbose=False, write_mode="w"
    )
    hats_import.pipeline(args)
    hats_import.pipeline_with_client(args, None)


@pytest.mark.dask(timeout=10)
def test_reimport_runner(dask_client, small_sky_object_catalog, tmp_path, mocker):
    mock_smtp = mocker.patch("smtplib.SMTP")

    args = hats_import.ImportArguments.reimport_from_hats(
        small_sky_object_catalog,
        tmp_path,
        output_artifact_name="small_sky_higher_order",
        highest_healpix_order=1,
        completion_email_address="nonsense",
    )

    hats_import.pipeline_with_client(args, dask_client)
    mock_smtp.assert_called_once_with("localhost")
    mock_smtp.return_value.__enter__.return_value.send_message.assert_called_once()
    sent_message = mock_smtp.return_value.__enter__.return_value.send_message.call_args.args[0]
    assert sent_message["Subject"] == "hats-import success."
    assert sent_message["To"] == "nonsense"
    assert sent_message["From"] == "updates@lsdb.io"
    assert sent_message.get_content().startswith("output_artifact_name: small_sky_higher_order")


@pytest.mark.dask(timeout=10)
def test_email_error(small_sky_object_catalog, tmp_path, mocker):
    mock_smtp = mocker.patch("smtplib.SMTP")

    args = hats_import.ImportArguments.reimport_from_hats(
        small_sky_object_catalog,
        tmp_path,
        output_artifact_name="small_sky_higher_order",
        highest_healpix_order=1,
        completion_email_address="nonsense",
    )
    with pytest.raises(AttributeError):
        hats_import.pipeline_with_client(args, None)

    mock_smtp.assert_called_once_with("localhost")
    mock_smtp.return_value.__enter__.return_value.send_message.assert_called_once()
    sent_message = mock_smtp.return_value.__enter__.return_value.send_message.call_args.args[0]
    assert sent_message["Subject"] == "hats-import failure."
    assert sent_message["To"] == "nonsense"
    assert sent_message["From"] == "updates@lsdb.io"
    assert sent_message.get_content().startswith("output_artifact_name: small_sky_higher_order")
