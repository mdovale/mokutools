# BSD 3-Clause License
#
# Copyright (c) 2025, Miguel Dovale
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
Backward-compatible wrapper for mokutools.filetools.

This module maintains the old API while delegating to the new moku_io modules.
For new code, prefer using mokutools.moku_io directly.
"""

import sys
import warnings
from typing import List, Optional, Union

# Import core functions
from mokutools.moku_io.core import (
    list_files,
    download,
    upload,
    delete,
    parse_csv_file,
    is_mat_file,
    is_li_file,
    read_lines,
    get_columns_with_nans,
)

# Import CLI helpers
from mokutools.moku_io.cli import (
    print_menu,
    pick_two_files,
    pick_file,
)

# Import notebook widgets
from mokutools.moku_io.notebook import (
    select_file_widget,
)

# Legacy constants (kept for backward compatibility)
SERVER_URL = "http://10.128.100.198/api/ssd"
DATA_DIR = "./data"


# Backward-compatible function wrappers
def get_file_list(ip: str, filter: Optional[List[str]] = None):
    """
    Fetch the list of files available from the Moku server.
    
    .. deprecated:: 1.0.0
        Use :func:`mokutools.moku_io.core.list_files` instead.
    
    Parameters
    ----------
    ip : str
        IP address of the device (e.g., '10.128.100.198').
    filter : list of str, optional
        List of substrings to filter filenames by. Only filenames containing
        all the substrings (case-insensitive) will be returned.

    Returns
    -------
    list of str
        List of filenames available from the server, filtered if `filter` is given.
    """
    warnings.warn(
        "get_file_list is deprecated. Use mokutools.moku_io.core.list_files instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return list_files(ip, filters=filter)


def download_files(
    ip: str,
    file_names: Optional[Union[str, List[str]]] = None,
    date: Optional[str] = None,
    convert: bool = True,
    archive: bool = True,
    output_path: Optional[str] = None,
    remove_from_server: bool = False,
) -> None:
    """
    Download `.li` files from a Moku device and optionally convert, compress, and delete them.
    
    .. deprecated:: 1.0.0
        Use :func:`mokutools.moku_io.core.download` for programmatic use,
        or :func:`mokutools.moku_io.cli.download_cli` for CLI use.

    Parameters
    ----------
    ip : str
        IP address of the device (e.g., '10.128.100.198').
    file_names : str or list of str, optional
        Partial filename or list of partial strings to match files.
        If provided, the `date` argument is ignored.
    date : str, optional
        A date string in 'YYYYMMDD' format to filter filenames.
    convert : bool, default True
        If True, convert the `.li` file to `.csv` using `mokucli`.
    archive : bool, default True
        If True, zip the `.csv` file. Applies only if `convert=True`.
    output_path : str, optional
        Directory where output files will be saved. Defaults to current directory.
    remove_from_server : bool, default False
        If True, delete the `.li` file from the device after processing.

    Returns
    -------
    None
        This function processes files as described but returns no value.
    """
    warnings.warn(
        "download_files is deprecated. Use mokutools.moku_io.core.download "
        "or mokutools.moku_io.cli.download_cli instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    try:
        patterns = file_names if file_names else None
        processed = download(
            ip=ip,
            patterns=patterns,
            date=date,
            convert=convert,
            archive=archive,
            output_path=output_path,
            remove_from_server=remove_from_server,
        )
        
        if not processed:
            print("⚠️ No matching files found.")
        else:
            for filename in processed:
                print(f"✅ Finished processing: {filename}")
    except ValueError as e:
        # Re-raise validation errors (like missing file_names/date)
        # but handle mokucli not found specially
        if "mokucli not found" in str(e):
            print("❌ `mokucli` not found. Please install it from:")
            print("   https://liquidinstruments.com/software/utilities/")
        else:
            # Re-raise validation errors to maintain backward compatibility
            raise
    except Exception as e:
        print(f"❌ Error processing files: {e}")


def delete_files(
    ip: str,
    file_names: Optional[Union[str, List[str]]] = None,
    delete_all: bool = False,
) -> None:
    """
    Delete files from a Moku device, optionally by partial match, full list, or all files.
    
    .. deprecated:: 1.0.0
        Use :func:`mokutools.moku_io.notebook.delete_files_widget` for notebook use,
        or :func:`mokutools.moku_io.cli.delete_cli` for CLI use.

    Parameters
    ----------
    ip : str
        IP address of the device (e.g., '10.128.100.198').
    file_names : str or list of str, optional
        Partial filename string or list of substrings to match files. Ignored if `delete_all` is True.
    delete_all : bool, default False
        If True, delete all files on the device. Overrides `file_names`.

    Returns
    -------
    None
    """
    warnings.warn(
        "delete_files is deprecated. Use mokutools.moku_io.notebook.delete_files_widget "
        "or mokutools.moku_io.cli.delete_cli instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Use notebook widget version for backward compatibility
    from mokutools.moku_io.notebook import delete_files_widget
    delete_files_widget(ip, file_names=file_names, delete_all=delete_all)


def upload_files(ip: str, files: Union[str, List[str]]) -> None:
    """
    Upload one or more files to the Moku device's SSD.
    
    .. deprecated:: 1.0.0
        Use :func:`mokutools.moku_io.core.upload` for programmatic use,
        or :func:`mokutools.moku_io.cli.upload_cli` for CLI use.

    Parameters
    ----------
    ip : str
        IP address of the device (e.g., '10.128.100.198').
    files : str or list of str
        Path to a local file or list of local file paths to upload.

    Returns
    -------
    None
    """
    warnings.warn(
        "upload_files is deprecated. Use mokutools.moku_io.core.upload "
        "or mokutools.moku_io.cli.upload_cli instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    try:
        results = upload(ip, files)
        for filename, success in results.items():
            if success:
                print(f"✅ Uploaded: {filename}")
            else:
                print(f"❌ Failed to upload: {filename}")
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
    except Exception as e:
        print(f"❌ Error uploading files: {e}")


# Re-export core functions (no deprecation warnings for these)
__all__ = [
    # Core functions (new API)
    "list_files",
    "download",
    "upload",
    "delete",
    "parse_csv_file",
    "is_mat_file",
    "is_li_file",
    "read_lines",
    "get_columns_with_nans",
    # Legacy functions (deprecated)
    "get_file_list",
    "download_files",
    "delete_files",
    "upload_files",
    # CLI helpers
    "print_menu",
    "pick_two_files",
    "pick_file",
    # Notebook widgets
    "select_file_widget",
    # Constants
    "SERVER_URL",
    "DATA_DIR",
]
