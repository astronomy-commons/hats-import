import abc

from hats.io import file_io
from upath import UPath


class InputReader(abc.ABC):
    """Base class for chunking file readers."""

    @abc.abstractmethod
    def read(self, input_file, read_columns=None):
        """Read the input file, or chunk of the input file.

        Args:
            input_file(str): path to the input file.
            read_columns(List[str]): subset of columns to read.
                if None, all columns are read
        Yields:
            DataFrame containing chunk of file info.
        """

    def regular_file_exists(self, input_file, **_kwargs):
        """Check that the `input_file` points to a single regular file

        Raises:
            FileNotFoundError: if nothing exists at path, or directory found.
        """
        input_file = file_io.get_upath(input_file)
        if not file_io.does_file_or_directory_exist(input_file):
            raise FileNotFoundError(f"File not found at path: {input_file}")
        if not file_io.is_regular_file(input_file):
            raise FileNotFoundError(f"Directory found at path - requires regular file: {input_file}")
        return input_file

    def read_index_file(self, input_file, upath_kwargs=None, **kwargs):
        """Read an "indexed" file.

        This should contain a list of paths to files to be read and batched.

        In order to create a valid connection to the string paths, provide any
        additional universal pathlib (i.e. fsspec) arguments to the `upath_kwargs` kwarg.
        In this way, the "index" file may contain a list of paths on a remote service,
        and the `upath_kwargs` will be used to create a connection to that remote service.

        Raises:
            FileNotFoundError: if nothing exists at path, or directory found.
        """
        input_file = self.regular_file_exists(input_file, **kwargs)
        file_names = file_io.load_text_file(input_file)
        file_names = [f.strip() for f in file_names]
        if upath_kwargs is None:
            upath_kwargs = {}

        file_paths = [UPath(f, **upath_kwargs) for f in file_names if f]

        return file_paths
