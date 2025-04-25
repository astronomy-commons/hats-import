import pytest

from hats_import.collection.arguments import CollectionArguments


def test_none():
    """No arguments provided. Should error for required args."""
    with pytest.raises(ValueError):
        CollectionArguments()


def test_missing_required(tmp_path):
    """No arguments provided. Should error for required args."""
    CollectionArguments(
        output_artifact_name="good_name",
        output_path=tmp_path,
    )

    with pytest.raises(ValueError, match="invalid character"):
        CollectionArguments(
            output_artifact_name="bad_a$$_name",
            output_path=tmp_path,
        )
    with pytest.raises(ValueError):
        CollectionArguments()

    ## Output path is missing
    with pytest.raises(ValueError, match="output_path"):
        CollectionArguments(
            output_artifact_name="catalog",
            output_path="",
        )

    ## Output catalog name is missing
    with pytest.raises(ValueError, match="output_artifact_name"):
        CollectionArguments(
            output_artifact_name="",
            output_path=tmp_path,
        )


def test_subcatalog_table_properties(tmp_path, blank_data_dir):
    args = (
        CollectionArguments(
            output_artifact_name="good_name",
            output_path=tmp_path,
            progress_bar=False,
            addl_hats_properties={"hats_cols_default": "id, mjd", "obs_regime": "Optical"},
        )
        .catalog(
            input_path=blank_data_dir,
            file_reader="csv",
        )
        .add_margin(margin_threshold=5.0)
        .add_margin(margin_threshold=15.0)
    )

    catalog_info = args.catalog_args.to_table_properties(
        total_rows=10, highest_order=4, moc_sky_fraction=22 / 7
    )

    assert catalog_info.catalog_name == "good_name"
    assert catalog_info.total_rows == 10
    assert catalog_info.default_columns == ["id", "mjd"]
    assert catalog_info.__pydantic_extra__["obs_regime"] == "Optical"


def test_subcatalog_wrong_order(tmp_path, blank_data_dir):
    with pytest.raises(ValueError, match="add catalog arguments"):
        CollectionArguments(
            output_artifact_name="good_name",
            output_path=tmp_path,
            progress_bar=False,
            addl_hats_properties={"hats_cols_default": "id, mjd", "obs_regime": "Optical"},
        ).add_margin(margin_threshold=5.0).add_margin(margin_threshold=15.0)

    with pytest.raises(ValueError, match="add catalog arguments"):
        CollectionArguments(
            output_artifact_name="good_name",
            output_path=tmp_path,
            progress_bar=False,
            addl_hats_properties={"hats_cols_default": "id, mjd", "obs_regime": "Optical"},
        ).add_index(indexing_column="id")

    with pytest.raises(ValueError, match="exactly once"):
        CollectionArguments(
            output_artifact_name="good_name",
            output_path=tmp_path,
            progress_bar=False,
            addl_hats_properties={"hats_cols_default": "id, mjd", "obs_regime": "Optical"},
        ).catalog(
            input_path=blank_data_dir,
            file_reader="csv",
        ).catalog(
            input_path=blank_data_dir,
            file_reader="csv",
        )

    args = CollectionArguments(
        output_artifact_name="good_name",
        output_path=tmp_path,
        progress_bar=False,
        addl_hats_properties={"hats_cols_default": "id, mjd", "obs_regime": "Optical"},
    )
    with pytest.raises(ValueError, match="add catalog arguments"):
        args.get_margin_args()
    with pytest.raises(ValueError, match="add catalog arguments"):
        args.get_index_args()


def test_subcatalog_existing_catalog(tmp_path, small_sky_object_catalog):
    args = (
        CollectionArguments(
            output_artifact_name="small_sky_collection",
            output_path=tmp_path,
            progress_bar=False,
            tmp_dir=tmp_path,
        )
        .catalog(
            catalog_path=small_sky_object_catalog,
        )
        .add_margin(margin_threshold=5.0)
        .add_margin(margin_threshold=15.0)
        .add_index(indexing_column="id")
    )

    margin_args = args.get_margin_args()
    assert len(margin_args) == 2

    # test idempotency
    margin_args = args.get_margin_args()
    assert len(margin_args) == 2

    index_args = args.get_index_args()
    assert len(index_args) == 1

    # test idempotency
    index_args = args.get_index_args()
    assert len(index_args) == 1


def test_index_bad_values(tmp_path, small_sky_object_catalog):
    args = (
        CollectionArguments(
            output_artifact_name="small_sky_collection",
            output_path=tmp_path,
            progress_bar=False,
            tmp_dir=tmp_path,
        )
        .catalog(
            catalog_path=small_sky_object_catalog,
        )
        .add_margin(margin_threshold=1_000_000)
        .add_index(indexing_column="id", extra_columns=["not", "there"])
    )

    with pytest.raises(ValueError, match="higher order"):
        args.get_margin_args()

    with pytest.raises(ValueError, match="not in input catalog"):
        args.get_index_args()
