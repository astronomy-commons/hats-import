import pytest

from hats_import.collection.arguments import CollectionArguments, _pretty_print_angle


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
        args.get_catalog_args()
    with pytest.raises(ValueError, match="add catalog arguments"):
        args.get_margin_args()
    with pytest.raises(ValueError, match="add catalog arguments"):
        args.get_index_args()
    with pytest.raises(ValueError, match="add catalog arguments"):
        args.to_collection_properties()


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
    with pytest.raises(ValueError, match="indexing_column is required"):
        (
            CollectionArguments(
                output_artifact_name="small_sky_collection",
                output_path=tmp_path,
                progress_bar=False,
                tmp_dir=tmp_path,
            )
            .catalog(
                catalog_path=small_sky_object_catalog,
            )
            .add_index()
        )

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
        .add_index(indexing_column="id", extra_columns=["not", "there"])
    )

    with pytest.raises(ValueError, match="not in input catalog"):
        args.get_index_args()


def test_margin_bad_values(tmp_path, small_sky_object_catalog):
    with pytest.raises(ValueError, match="threshold required"):
        (
            CollectionArguments(
                output_artifact_name="small_sky_collection",
                output_path=tmp_path,
                progress_bar=False,
                tmp_dir=tmp_path,
            )
            .catalog(
                catalog_path=small_sky_object_catalog,
            )
            .add_margin()
        )

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
    )

    with pytest.raises(ValueError, match="higher order"):
        args.get_margin_args()


def test_collection_properties_basic(tmp_path, blank_data_dir):
    args = CollectionArguments(
        output_artifact_name="good_name",
        output_path=tmp_path,
        progress_bar=False,
        addl_hats_properties={"obs_regime": "Optical"},
    ).catalog(
        input_path=blank_data_dir,
        file_reader="csv",
        addl_hats_properties={"hats_cols_default": "id, mjd"},
    )

    collection_info = args.to_collection_properties()
    assert collection_info.name == "good_name"
    assert collection_info.hats_primary_table_url == "good_name"
    assert collection_info.all_margins is None
    assert collection_info.all_indexes is None
    assert collection_info.__pydantic_extra__["obs_regime"] == "Optical"


def test_collection_properties_supplemental(tmp_path, small_sky_object_catalog):
    args = (
        CollectionArguments(
            output_artifact_name="small_sky_collection",
            new_catalog_name="small_sky",
            output_path=tmp_path,
            progress_bar=False,
            addl_hats_properties={"obs_regime": "Optical"},
        )
        .catalog(
            catalog_path=small_sky_object_catalog,
        )
        .add_margin(margin_threshold=5.0, is_default=True)
        .add_margin(margin_threshold=35.0)
        .add_index(indexing_column="id")
    )

    collection_info = args.to_collection_properties()
    assert collection_info.name == "small_sky_collection"
    assert len(collection_info.all_margins) == 2
    assert collection_info.default_margin == "small_sky_object_catalog_5arcs"
    assert len(collection_info.all_indexes) == 1
    assert collection_info.__pydantic_extra__["obs_regime"] == "Optical"

    margin_args = args.get_margin_args()[0]
    assert margin_args.output_artifact_name == "small_sky_object_catalog_5arcs"

    margin_args = args.get_margin_args()[1]
    assert margin_args.output_artifact_name == "small_sky_object_catalog_35arcs"

    index_args = args.get_index_args()[0]
    assert index_args.output_artifact_name == "small_sky_object_catalog_id"


def test_collection_properties_existing_supplemental(tmp_path, small_sky_object_catalog):
    """If we have already created a catalog or supplemental table in the
    collection, confirm that the collection properties are written with
    the union of supplemental tables."""
    args = (
        CollectionArguments(
            output_artifact_name="small_sky_collection",
            new_catalog_name="small_sky",
            output_path=tmp_path,
            progress_bar=False,
            addl_hats_properties={"obs_regime": "Optical"},
        ).catalog(
            catalog_path=small_sky_object_catalog,
        )
        # .add_margin(margin_threshold=5.0, is_default=True)
        # .add_margin(margin_threshold=35.0)
        # .add_index(indexing_column="id")
    )

    collection_info = args.to_collection_properties()
    collection_info.to_properties_file(args.catalog_path)

    args = (
        CollectionArguments(
            output_artifact_name="small_sky_collection",
            new_catalog_name="small_sky",
            output_path=tmp_path,
            progress_bar=False,
        )
        .catalog(
            catalog_path=small_sky_object_catalog,
        )
        .add_margin(margin_threshold=5.0, is_default=True)
        # .add_margin(margin_threshold=35.0)
        .add_index(indexing_column="id")
    )

    collection_info = args.to_collection_properties()
    collection_info.to_properties_file(args.catalog_path)

    args = (
        CollectionArguments(
            output_artifact_name="small_sky_collection",
            new_catalog_name="small_sky",
            output_path=tmp_path,
            progress_bar=False,
        ).catalog(
            catalog_path=small_sky_object_catalog,
        )
        # .add_margin(margin_threshold=5.0, is_default=True)
        .add_margin(margin_threshold=35.0)
        # .add_index(indexing_column="id")
    )

    collection_info = args.to_collection_properties()
    assert collection_info.name == "small_sky_collection"
    assert len(collection_info.all_margins) == 2
    assert collection_info.default_margin == "small_sky_object_catalog_5arcs"
    assert len(collection_info.all_indexes) == 1
    assert collection_info.__pydantic_extra__["obs_regime"] == "Optical"


def test_collection_default_margin(tmp_path, small_sky_object_catalog):
    """Test behavior of the default margin setting."""
    ## Easy case - name derived from primary catalog
    args = (
        CollectionArguments(
            output_artifact_name="small_sky_collection",
            new_catalog_name="small_sky",
            output_path=tmp_path,
            progress_bar=False,
        )
        .catalog(
            catalog_path=small_sky_object_catalog,
        )
        .add_margin(margin_threshold=5.0, is_default=True)
        .add_margin(margin_threshold=35.0)
    )

    collection_info = args.to_collection_properties()
    assert collection_info.name == "small_sky_collection"
    assert collection_info.all_margins == [
        "small_sky_object_catalog_5arcs",
        "small_sky_object_catalog_35arcs",
    ]
    assert collection_info.default_margin == "small_sky_object_catalog_5arcs"

    ## Margin name passed explicitly
    args = (
        CollectionArguments(
            output_artifact_name="small_sky_collection",
            new_catalog_name="small_sky",
            output_path=tmp_path,
            progress_bar=False,
        )
        .catalog(
            catalog_path=small_sky_object_catalog,
        )
        .add_margin(output_artifact_name="my_5arcs_margin", margin_threshold=5.0, is_default=True)
        .add_margin(margin_threshold=35.0)
    )

    collection_info = args.to_collection_properties()
    assert collection_info.name == "small_sky_collection"
    assert collection_info.all_margins == ["my_5arcs_margin", "small_sky_object_catalog_35arcs"]
    assert collection_info.default_margin == "my_5arcs_margin"

    with pytest.raises(ValueError, match="Only one margin catalog may be the default"):
        ## Only one default margin allowed.
        (
            CollectionArguments(
                output_artifact_name="small_sky_collection",
                new_catalog_name="small_sky",
                output_path=tmp_path,
                progress_bar=False,
            )
            .catalog(
                catalog_path=small_sky_object_catalog,
            )
            .add_margin(margin_threshold=5.0, is_default=True)
            .add_margin(margin_threshold=35.0, is_default=True)
        )


def test_pretty_print_angle():
    assert _pretty_print_angle(1) == "1arcs"
    assert _pretty_print_angle(100) == "1arcmin"
    assert _pretty_print_angle(10_000) == "2deg"
    assert _pretty_print_angle(0.1) == "100msec"
