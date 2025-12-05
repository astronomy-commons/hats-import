import pytest

from hats_import.nest_light_curves.arguments import NestLightCurveArguments


def test_none():
    """No arguments provided. Should error for required args."""
    with pytest.raises(ValueError):
        NestLightCurveArguments()


def test_empty_required(tmp_path, small_sky_object_catalog, small_sky_source_catalog):
    """All non-runtime arguments are required."""

    ## List of required args:
    ##  - match expression that should be found when missing
    ##  - default value
    required_args = [
        ["object_catalog_dir", small_sky_object_catalog],
        ["object_id_column", "id"],
        ["source_catalog_dir", small_sky_source_catalog],
        ["source_object_id_column", "object_id"],
        ["output_artifact_name", "small_sky_association"],
        ["output_path", tmp_path],
    ]

    ## For each required argument, check that a ValueError is raised that matches the
    ## expected name of the missing param.
    for index, args in enumerate(required_args):
        test_args = [
            list_args[1] if list_index != index else None
            for list_index, list_args in enumerate(required_args)
        ]

        with pytest.raises(ValueError, match=args[0]):
            NestLightCurveArguments(
                object_catalog_dir=test_args[0],
                object_id_column=test_args[1],
                source_catalog_dir=test_args[2],
                source_object_id_column=test_args[3],
                output_artifact_name=test_args[4],
                output_path=test_args[5],
                ## always set these False
                progress_bar=False,
            )


def test_catalog_paths(tmp_path, small_sky_object_catalog, small_sky_source_catalog):
    """*Most* required arguments are provided."""
    ## Object catalog path is bad.
    with pytest.raises(ValueError, match="object_catalog_dir"):
        NestLightCurveArguments(
            object_catalog_dir="/foo",
            object_id_column="id",
            source_catalog_dir=small_sky_source_catalog,
            source_object_id_column="object_id",
            output_artifact_name="small_sky_association",
            output_path=tmp_path,
            progress_bar=False,
        )

    ## Source catalog path is bad.
    with pytest.raises(ValueError, match="source_catalog_dir"):
        NestLightCurveArguments(
            object_catalog_dir=small_sky_object_catalog,
            object_id_column="id",
            source_catalog_dir="/foo",
            source_object_id_column="object_id",
            output_artifact_name="small_sky_association",
            output_path=tmp_path,
            progress_bar=False,
        )
