from __future__ import annotations

from pathlib import Path

from hats.io.file_io import get_upath
from upath import UPath


def append_paths_to_pointer(pointer: str | Path | UPath, *paths: str) -> UPath:
    """Append directories and/or a file name to a specified file pointer.

    Parameters
    ----------
    pointer : str | Path | UPath
        `FilePointer` object to add path to
    *paths: str
        any number of directory names optionally followed by a file name to append to the
        pointer

    Returns
    -------
    UPath
        New file pointer to path given by joining given pointer and path names
    """
    pointer = get_upath(pointer)
    return pointer.joinpath(*paths)


def find_files_matching_path(pointer: str | Path | UPath, *paths: str) -> list[UPath]:
    """Find files or directories matching the provided path parts.

    Parameters
    ----------
    pointer : str | Path | UPath
        base File Pointer in which to find contents
    *paths: str
        any number of directory names optionally followed by a file name.
        directory or file names may be replaced with `*` as a matcher.

    Returns
    -------
    list[UPath]
        New file pointers to files found matching the path
    """
    pointer = get_upath(pointer)

    if len(paths) == 0:
        return [pointer]

    matcher = pointer.fs.sep.join(paths)
    contents = []
    for child in pointer.rglob(matcher):
        contents.append(child)

    if len(contents) == 0:
        return []

    contents.sort()
    return contents


def directory_has_contents(pointer: str | Path | UPath) -> bool:
    """Checks if a directory already has some contents (any files or subdirectories)

    Parameters
    ----------
    pointer : str | Path | UPath
        File Pointer to check for existing contents

    Returns
    -------
    bool
        True if there are any files or subdirectories below this directory.
    """
    pointer = get_upath(pointer)

    return next(pointer.rglob("*"), None) is not None
