from astropy.io import ascii as ascii_reader

from hats_import.catalog.file_readers.input_reader import InputReader


class AstropyEcsvReader(InputReader):
    """Reads astropy ascii .ecsv files.

    Note that this is NOT a chunked reader. Use caution when reading
    large ECSV files with this reader.

    Attributes:
        kwargs: keyword arguments passed to astropy ascii reader.
            See https://docs.astropy.org/en/stable/api/astropy.io.ascii.read.html#astropy.io.ascii.read
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def read(self, input_file, read_columns=None):
        self.regular_file_exists(input_file, **self.kwargs)
        if read_columns:
            self.kwargs["include_names"] = read_columns

        astropy_table = ascii_reader.read(input_file, format="ecsv", **self.kwargs)
        yield astropy_table.to_pandas()
