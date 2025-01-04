import random
import shutil
from pathlib import Path

import attrs
import pyarrow
import pyarrow.dataset
import pyarrow.parquet

DATA_DIR = Path(__file__).parent.parent.parent.parent / "tests/data"
INPUT_CATALOG_DIR = DATA_DIR / "small_sky_object_catalog"


def run(input_catalog_dir: Path = INPUT_CATALOG_DIR, data_dir: Path = DATA_DIR) -> None:
    """Generate malformed catalogs to be used as test data for verification.
    This only needs to be run once unless/until it is desirable to regenerate the dataset.
    """
    Generate().run(input_catalog_dir=input_catalog_dir, data_dir=data_dir)


@attrs.define
class CatalogBase:
    """Container to hold the input catalog (loaded as pyarrow objects) and output paths."""

    dataset: pyarrow.dataset.Dataset = attrs.field()
    frag: pyarrow.dataset.FileFragment = attrs.field()
    tbl: pyarrow.Table = attrs.field()
    schema: pyarrow.Schema = attrs.field()
    input_catalog_dir: Path = attrs.field()
    data_dir: Path = attrs.field()
    cat_dir_name: str = attrs.field(factory=str)

    @classmethod
    def from_dirs(cls, input_catalog_dir: Path, data_dir: Path) -> "CatalogBase":
        input_ds = pyarrow.dataset.parquet_dataset(input_catalog_dir / "dataset/_metadata")
        # assert input_ds.metadata is not None, "Unit tests expect "
        input_frag = next(input_ds.get_fragments())
        input_tbl = input_frag.to_table()
        return cls(
            dataset=input_ds,
            frag=input_frag,
            tbl=input_tbl,
            schema=input_tbl.schema,
            input_catalog_dir=input_catalog_dir,
            data_dir=data_dir,
        )

    @property
    def fmeta(self) -> Path:
        return self.data_dir / self.cat_dir_name / "dataset/_metadata"

    @property
    def fcmeta(self) -> Path:
        return self.data_dir / self.cat_dir_name / "dataset/_common_metadata"

    @property
    def fdata(self) -> Path:
        frag_key = Path(self.frag.path).relative_to(self.input_catalog_dir)
        return self.data_dir / self.cat_dir_name / frag_key


@attrs.define
class Generate:
    """Generate malformed catalogs for verification testing."""

    def run(self, input_catalog_dir: Path = INPUT_CATALOG_DIR, data_dir: Path = DATA_DIR) -> None:
        """Generate malformed catalogs to be used as test data for verification.
        This only needs to be run once unless/until it is desirable to regenerate the dataset.
        """
        print(f"Generating malformed catalogs from input catalog {input_catalog_dir}.")

        catbase = CatalogBase.from_dirs(input_catalog_dir=input_catalog_dir, data_dir=data_dir)
        Generate().bad_schemas(catbase=catbase)
        Generate().wrong_files_and_rows(catbase=catbase)

    def bad_schemas(self, catbase: CatalogBase) -> None:
        """Case: Files are altered in a way that affects the schema after _metadata gets written."""
        catbase.cat_dir_name = "bad_schemas"
        self._start_new_catalog(catbase)

        # Write new files with the correct schema
        fextra_col = catbase.fdata.with_suffix(".extra_column.parquet")
        fmissing_col = catbase.fdata.with_suffix(".missing_column.parquet")
        fwrong_metadata = catbase.fdata.with_suffix(".wrong_metadata.parquet")
        fwrong_types = catbase.fdata.with_suffix(".wrong_dtypes.parquet")
        for _fout in [fmissing_col, fextra_col, fwrong_types]:
            pyarrow.parquet.write_table(catbase.tbl, _fout)

        # Write a _metadata that is correct except for file-level metadata
        extra_metadata = {b"extra key": b"extra value"}
        self._collect_and_write_metadata(catbase, schema=catbase.schema.with_metadata(extra_metadata))

        # Overwrite the new data files using incorrect schemas.
        # drop a column
        pyarrow.parquet.write_table(catbase.tbl.drop_columns("dec_error"), fmissing_col)
        # add an extra column
        extra_col = pyarrow.array(random.sample(range(1000), len(catbase.tbl)))
        extra_col_tbl = catbase.tbl.add_column(5, pyarrow.field("extra", pyarrow.int64()), extra_col)
        pyarrow.parquet.write_table(extra_col_tbl, fextra_col)
        # add or drop file-level metadata
        wrong_metadata = {"bad key": "bad value"} if catbase.tbl.schema.metadata is None else None
        pyarrow.parquet.write_table(catbase.tbl.replace_schema_metadata(wrong_metadata), fwrong_metadata)
        # change some types
        wrong_dtypes = [
            fld if not fld.name.startswith("ra") else fld.with_type(pyarrow.float16())
            for fld in catbase.schema
        ]
        wrong_dtypes_schema = pyarrow.schema(wrong_dtypes).with_metadata(catbase.schema.metadata)
        pyarrow.parquet.write_table(catbase.tbl.cast(wrong_dtypes_schema), fwrong_types)

        # Write a _common_metadata with the wrong dtypes.
        pyarrow.parquet.write_metadata(schema=wrong_dtypes_schema, where=catbase.fcmeta)

        # Write a _common_metadata without hats columns.
        # This mimics a schema that could have been passed as 'use_schema_file' upon import.
        fcustom_md = catbase.fcmeta.with_suffix(".import_truth")
        hats_cols = ["_healpix_29", "Norder", "Dir", "Npix"]
        import_fields = [fld for fld in catbase.schema if fld.name not in hats_cols]
        import_schema = pyarrow.schema(import_fields)
        pyarrow.parquet.write_metadata(schema=import_schema, where=fcustom_md)

        print(f"Malformed catalog written to {catbase.fmeta.parent.parent}")

    def wrong_files_and_rows(self, catbase: CatalogBase) -> None:
        """Case: Dataset is altered in a way that affects the number of rows and/or files
        after _metadata gets written."""
        catbase.cat_dir_name = "wrong_files_and_rows"
        self._start_new_catalog(catbase)

        fmissing_file = catbase.fdata.with_suffix(".missing_file.parquet")
        fextra_file = catbase.fdata.with_suffix(".extra_file.parquet")
        fextra_rows = catbase.fdata.with_suffix(".extra_rows.parquet")

        # Write the "correct" dataset, including metadata.
        pyarrow.parquet.write_table(catbase.tbl, fmissing_file)
        pyarrow.parquet.write_table(catbase.tbl, fextra_rows)
        self._collect_and_write_metadata(catbase)

        # Mangle the dataset.
        fmissing_file.unlink()
        pyarrow.parquet.write_table(catbase.tbl, fextra_file)
        pyarrow.parquet.write_table(self._tbl_with_extra_rows(catbase), fextra_rows)

        print(f"Malformed catalog written to {catbase.fmeta.parent.parent}")

    def _tbl_with_extra_rows(self, catbase: CatalogBase) -> pyarrow.Table:
        """Generate a table with extra rows."""
        # generate new rows
        rng = range(len(catbase.tbl))
        nrows, new_rows = 2, {}
        for col in catbase.tbl.column_names:
            if col not in ("_healpix_29", "id"):
                # just take a random sample
                new_rows[col] = catbase.tbl.column(col).take(random.sample(rng, nrows))
            else:
                # increment the max value to avoid duplicates
                max_id = catbase.tbl.column(col).sort()[-1].as_py()
                new_rows[col] = [i + max_id for i in range(1, nrows + 1)]

        # add the rows to the table
        new_tbl = pyarrow.concat_tables(
            [catbase.tbl, pyarrow.Table.from_pydict(new_rows, schema=catbase.schema)]
        )
        return new_tbl

    @staticmethod
    def _start_new_catalog(catbase: CatalogBase, with_ancillaries: bool = False) -> None:
        # Start a new catalog by creating the directory and copying in input files.
        dataset_dir = catbase.fmeta.parent
        if dataset_dir.is_dir():
            print(f"Output directory exists. Remove it and try again.\n{dataset_dir}")
            return

        catbase.fdata.parent.mkdir(parents=True)
        shutil.copy(catbase.frag.path, catbase.fdata)

        if with_ancillaries:
            for fin in catbase.input_catalog_dir.iterdir():
                if fin.is_file():
                    shutil.copy(fin, dataset_dir.parent / fin.name)
        for fin in (catbase.input_catalog_dir / "dataset").iterdir():
            if fin.is_file():
                shutil.copy(fin, dataset_dir / fin.name)

    @staticmethod
    def _collect_and_write_metadata(catbase: CatalogBase, schema: pyarrow.Schema | None = None) -> None:
        base_dir = catbase.fmeta.parent
        schema = schema or catbase.schema
        dataset = pyarrow.dataset.dataset(base_dir)
        metadata_collector = []
        for frag in dataset.get_fragments():
            frag.ensure_complete_metadata()
            frag.metadata.set_file_path(str(Path(frag.path).relative_to(base_dir)))
            metadata_collector.append(frag.metadata)
        pyarrow.parquet.write_metadata(
            schema=schema, where=catbase.fmeta, metadata_collector=metadata_collector
        )
