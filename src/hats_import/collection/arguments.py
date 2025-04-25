"""Utility to hold all arguments required throughout partitioning"""

from __future__ import annotations

from dataclasses import dataclass, field

from hats.io import file_io
from hats.io.validation import is_valid_catalog
from upath import UPath

from hats_import.catalog import ImportArguments
from hats_import.index.arguments import IndexArguments
from hats_import.margin_cache.margin_cache_arguments import MarginCacheArguments
from hats_import.runtime_arguments import RuntimeArguments


@dataclass
class CollectionArguments(RuntimeArguments):
    """Container class for holding partitioning arguments"""

    new_catalog_path: UPath | None = None
    catalog_args: ImportArguments | None = None
    margin_kwargs: list[dict] = field(default_factory=list)
    index_kwargs: list[dict] = field(default_factory=list)
    margin_args: list[dict] = field(default_factory=list)
    index_args: list[dict] = field(default_factory=list)

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
            self.new_catalog_path = new_catalog_path
            return self

        self.catalog_args = ImportArguments(**useful_kwargs)
        self.new_catalog_path = self.catalog_args.catalog_path

        return self

    def add_margin(self, **kwargs):
        """Add arguments for a margin catalog.

        NB: This can be called 0, 1, or many times for a single collection.
        This method will only stash the provided arguments for later use,
        as the arguments cannot be validated until the catalog exists on disk."""
        if self.new_catalog_path is None:
            raise ValueError("Must add catalog arguments before adding margin arguments")
        useful_kwargs = self._get_subarg_dict()
        useful_kwargs.update(kwargs)

        if "output_artifact_name" not in kwargs:
            useful_kwargs["output_artifact_name"] = f"{self.output_artifact_name}_margin"
        if "input_catalog_path" not in kwargs:
            useful_kwargs["input_catalog_path"] = self.new_catalog_path

        self.margin_kwargs.append(useful_kwargs)

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
        useful_kwargs = self._get_subarg_dict()
        useful_kwargs.update(kwargs)

        if "output_artifact_name" not in kwargs:
            useful_kwargs["output_artifact_name"] = f"{self.output_artifact_name}_index"
        if "input_catalog_path" not in kwargs:
            useful_kwargs["input_catalog_path"] = self.new_catalog_path

        self.index_kwargs.append(useful_kwargs)

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
            "tmp_dir": self.tmp_base_path,
            "resume": self.resume,
            "progress_bar": self.progress_bar,
            "simple_progress_bar": self.simple_progress_bar,
            "resume_tmp": self.resume_tmp,
            "delete_intermediate_parquet_files": self.delete_intermediate_parquet_files,
            "delete_resume_log_files": self.delete_resume_log_files,
        }
        return useful_kwargs
