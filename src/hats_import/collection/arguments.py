from __future__ import annotations

from dataclasses import dataclass, field

import hats.pixel_math.healpix_shim as hp
from hats import read_hats
from hats.catalog.dataset.collection_properties import CollectionProperties
from hats.io import file_io
from hats.io.validation import is_valid_catalog
from upath import UPath

from hats_import.catalog import ImportArguments
from hats_import.index.arguments import IndexArguments
from hats_import.margin_cache.margin_cache_arguments import MarginCacheArguments
from hats_import.runtime_arguments import RuntimeArguments


@dataclass
class CollectionArguments(RuntimeArguments):
    """Container class for building arguments for importing catalog as a collection."""

    new_catalog_path: UPath | None = None
    """Constructed path for the new catalog, relative to the collection."""
    new_catalog_name: str | None = None
    """Name for the new catalog that will be created. Taken from the ``output_artifact_name`` of
    catalog kwargs. Used to construct names of the supplemental catalogs, if their own
    ``output_artifact_name`` isn't specified."""
    catalog_args: ImportArguments | None = None
    """Constructed arguments for catalog import."""
    margin_kwargs: list[dict] = field(default_factory=list)
    """List of all argument dictionaries passed to this builder, for creating margins."""
    default_margin_name: str | None = None
    index_kwargs: list[dict] = field(default_factory=list)
    """List of all argument dictionaries passed to this builder, for creating indexes."""
    margin_args: list[dict] = field(default_factory=list)
    """Constructed arguments for creating margins."""
    margin_paths: list[str] = field(default_factory=list)
    """Paths to margins that may be created by these arguments"""
    index_args: list[dict] = field(default_factory=list)
    """Constructed arguments for creating indexes."""
    index_paths: dict[str, str] = field(default_factory=dict)
    """Paths to indexes that may be created by these arguments"""

    def catalog(self, **kwargs):
        """Set the primary catalog for the collection.

        NB: This should only be called EXACTLY ONCE per catalog collection.
        If building a new catalog from scratch, these should be all the arguments necessary
        for import. If using an existing catalog, you should provide the ``output_artifact_name``
        so the primary catalog subdirectory can be located and validated.
        """
        if self.new_catalog_path is not None:
            raise ValueError("Must call catalog method exactly once.")
        useful_kwargs = self._get_subarg_dict()
        useful_kwargs.update(kwargs)
        if "output_artifact_name" not in kwargs:
            useful_kwargs["output_artifact_name"] = self.output_artifact_name

        ## Test for an existing catalog at the indicated path.
        if "catalog_path" in kwargs:
            new_catalog_path = file_io.get_upath(kwargs["catalog_path"])
        else:
            new_catalog_path = (
                file_io.get_upath(self.output_path)
                / self.output_artifact_name
                / useful_kwargs["output_artifact_name"]
            )
        if is_valid_catalog(new_catalog_path):
            ## There is already a valid catalog (either from resume or pre-existing).
            ## Leave it alone and write the remainder of the collection contents.
            catalog = read_hats(new_catalog_path)
            self.new_catalog_path = catalog.catalog_path
            self.new_catalog_name = catalog.catalog_name
            return self

        self.catalog_args = ImportArguments(**useful_kwargs)
        self.new_catalog_path = self.catalog_args.catalog_path
        self.new_catalog_name = self.catalog_args.output_artifact_name

        return self

    def get_catalog_args(self):
        """Retrieve the catalog arguments, if a catalog must be created or resumed."""
        if self.new_catalog_path is None:
            raise ValueError("Must add catalog arguments before fetching catalog arguments")

        return self.catalog_args

    def add_margin(self, is_default=False, **kwargs):
        """Add arguments for a margin catalog.

        NB: This can be called 0, 1, or many times for a single collection.
        This method will only stash the provided arguments for later use,
        as the arguments cannot be validated until the catalog exists on disk.

        Args:
            is_default(bool): If True, this margin will be set as the default on the
                catalog collection's properties.
            kwargs (dict): arguments passed to ``MarginCacheArguments`` constructor.
        """
        if self.new_catalog_path is None:
            raise ValueError("Must add catalog arguments before adding margin arguments")

        if "margin_threshold" not in kwargs and "margin_order" not in kwargs:
            raise ValueError("Margin threshold required, either in arcseconds or HEALPix order.")

        margin_threshold = kwargs["margin_threshold"] or hp.order2mindist(kwargs["margin_order"]) * 60

        useful_kwargs = self._get_subarg_dict()
        useful_kwargs.update(kwargs)

        if "output_artifact_name" not in kwargs:
            margin_suffix = _pretty_print_angle(margin_threshold)
            useful_kwargs["output_artifact_name"] = f"{self.new_catalog_name}_{margin_suffix}"
        if is_default:
            default_name = useful_kwargs["output_artifact_name"]
            if self.default_margin_name:
                raise ValueError(
                    "Only one margin catalog may be the default "
                    f"({self.default_margin_name} already set, trying to add {default_name})"
                )
            self.default_margin_name = default_name

        if "input_catalog_path" not in kwargs:
            useful_kwargs["input_catalog_path"] = self.new_catalog_path

        if "catalog_path" in kwargs:
            new_catalog_path = file_io.get_upath(kwargs["catalog_path"])
        else:
            new_catalog_path = (
                file_io.get_upath(self.output_path)
                / self.output_artifact_name
                / useful_kwargs["output_artifact_name"]
            )
        if not is_valid_catalog(new_catalog_path):
            self.margin_kwargs.append(useful_kwargs)

        self.margin_paths.append(_maybe_relative(new_catalog_path, self.catalog_path))

        return self

    def get_margin_args(self):
        """Construct and return the margin argument objects, validating the inputs."""
        if self.new_catalog_path is None:
            raise ValueError("Must add catalog arguments before fetching margin arguments")
        if not self.margin_args:
            self.margin_args = []
            for margin_kwargs in self.margin_kwargs:
                self.margin_args.append(MarginCacheArguments(**margin_kwargs))
        return self.margin_args

    def add_index(self, **kwargs):
        """Add arguments for an index catalog.

        NB: This can be called 0, 1, or many times for a single collection.
        This method will only stash the provided arguments for later use,
        as the arguments cannot be validated until the catalog exists on disk."""
        if self.new_catalog_path is None:
            raise ValueError("Must add catalog arguments before adding index arguments")

        if "indexing_column" not in kwargs:
            raise ValueError("indexing_column is required")
        indexing_column = kwargs["indexing_column"]

        useful_kwargs = self._get_subarg_dict()
        useful_kwargs.update(kwargs)

        if "output_artifact_name" not in kwargs:
            useful_kwargs["output_artifact_name"] = f"{self.new_catalog_name}_{indexing_column}"
        if "input_catalog_path" not in kwargs:
            useful_kwargs["input_catalog_path"] = self.new_catalog_path

        if "catalog_path" in kwargs:
            new_catalog_path = file_io.get_upath(kwargs["catalog_path"])
        else:
            new_catalog_path = (
                file_io.get_upath(self.output_path)
                / self.output_artifact_name
                / useful_kwargs["output_artifact_name"]
            )
        if not is_valid_catalog(new_catalog_path):
            self.index_kwargs.append(useful_kwargs)

        self.index_paths[indexing_column] = _maybe_relative(new_catalog_path, self.catalog_path)

        return self

    def get_index_args(self):
        """Construct and return the index argument objects, validating the inputs."""
        if self.new_catalog_path is None:
            raise ValueError("Must add catalog arguments before fetching index arguments")
        if not self.index_args:
            self.index_args = []
            for index_kwargs in self.index_kwargs:
                self.index_args.append(IndexArguments(**index_kwargs))
        return self.index_args

    def _get_subarg_dict(self):
        """Get a subset of this object's arguments as a dictionary.

        Not all of the original arguments are useful, and ALL of them
        can potentially be overriden by the user's ``kwargs`` on the method.

        Note that we will not copy over arguments related to dask runtime, as
        the client will be created once at the beginning of collection creation
        and used throughout.
        """
        useful_kwargs = {
            "output_path": self.catalog_path,
            "addl_hats_properties": self.addl_hats_properties,
            "tmp_dir": self.tmp_path,
            "resume": self.resume,
            "progress_bar": self.progress_bar,
            "simple_progress_bar": self.simple_progress_bar,
            "delete_intermediate_parquet_files": self.delete_intermediate_parquet_files,
            "delete_resume_log_files": self.delete_resume_log_files,
        }
        return useful_kwargs

    def to_collection_properties(self):
        """Collection-specific dataset info."""
        if self.new_catalog_path is None:
            raise ValueError("Must add catalog arguments before collection properties")
        info = {"name": self.output_artifact_name}

        info["hats_primary_table_url"] = _maybe_relative(self.new_catalog_path, self.catalog_path)

        if self.margin_paths:
            info["all_margins"] = self.margin_paths
        if self.default_margin_name:
            info["default_margin"] = self.default_margin_name
        if self.index_paths:
            info["all_indexes"] = self.index_paths

        info = info | self.extra_property_dict()
        new_properties = CollectionProperties(**info)
        existing_properties = None
        try:
            ## If there is already a collection at this location, try to update the values.
            existing_properties = CollectionProperties.read_from_dir(self.catalog_path)
        except FileNotFoundError:
            pass

        if existing_properties:
            if new_properties.all_indexes or existing_properties.all_indexes:
                info["all_indexes"] = existing_properties.all_indexes or {} | new_properties.all_indexes or {}

            if new_properties.all_margins or existing_properties.all_margins:
                info["all_margins"] = list(
                    set((new_properties.all_margins or []) + (existing_properties.all_margins or []))
                )

            new_properties = existing_properties.model_copy(update=info)
            CollectionProperties.model_validate(new_properties)
        return new_properties


def _pretty_print_angle(arc_seconds):
    if arc_seconds >= 3600:
        return f"{int(arc_seconds / 3600)}deg"
    if arc_seconds >= 60:
        return f"{int(arc_seconds/60)}arcmin"
    if arc_seconds >= 1:
        return f"{int(arc_seconds)}arcs"
    return f"{int(arc_seconds * 1000)}msec"


def _maybe_relative(artifact_path, collection_path):
    if artifact_path.is_relative_to(collection_path):
        return str(artifact_path.relative_to(collection_path))
    return str(artifact_path)
