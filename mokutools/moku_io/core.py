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
Core functions for Moku I/O operations.

This module provides pure functions that return data and raise exceptions.
No interactive I/O (print, input, widgets, sys.exit) is used.
"""

import csv
import os
import re
import shutil
import zipfile
import requests
import subprocess
from io import TextIOWrapper
import scipy.io
import tarfile
import gzip
import tempfile
from py7zr import SevenZipFile
import numpy as np
import logging
from typing import List, Optional, Union, Tuple, Dict


def list_files(ip: str, filters: Optional[List[str]] = None) -> List[str]:
    """
    Fetch the list of files available from the Moku server at the specified IP address,
    optionally filtering the filenames by a list of substrings (case-insensitive, AND logic).

    Parameters
    ----------
    ip : str
        IP address of the device (e.g., '10.128.100.198').
    filters : list of str, optional
        List of substrings to filter filenames by. Only filenames containing
        all the substrings (case-insensitive) will be returned.

    Returns
    -------
    list of str
        List of filenames available from the server, filtered if `filters` is given.

    Raises
    ------
    requests.HTTPError
        If the request fails or the response format is incorrect.

    Notes
    -----
    - The function sends a GET request to `http://<ip>/api/ssd/list` and parses 
      the JSON response.
    - It is assumed that the response contains a `data` field holding the file list.
    """
    url = f"http://{ip}/api/ssd/list"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    file_list = data.get("data", [])

    if filters:
        lower_filter = [s.lower() for s in filters]
        file_list = [
            fname for fname in file_list
            if all(sub in fname.lower() for sub in lower_filter)
        ]

    return file_list


def download(
    ip: str,
    patterns: Optional[Union[str, List[str]]] = None,
    date: Optional[str] = None,
    convert: bool = True,
    archive: bool = True,
    output_path: Optional[str] = None,
    remove_from_server: bool = False,
) -> List[str]:
    """
    Download `.li` files from a Moku device and optionally convert, compress, and delete them.

    Parameters
    ----------
    ip : str
        IP address of the device (e.g., '10.128.100.198').
    patterns : str or list of str, optional
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
    list of str
        List of filenames that were successfully processed.

    Raises
    ------
    ValueError
        If neither `patterns` nor `date` is provided, or if `convert=True` but
        `mokucli` is not found in PATH.
    requests.HTTPError
        If download or delete requests fail.
    subprocess.CalledProcessError
        If file conversion fails.

    Notes
    -----
    - Requires `mokucli` to be installed and available in the system PATH if `convert=True`.
    - Files are matched by substring (partial matching supported).
    - The device API must support `DELETE` requests to `/api/ssd/delete/<filename>`.
    """
    if convert and not shutil.which("mokucli"):
        raise ValueError(
            "mokucli not found. Please install it from: "
            "https://liquidinstruments.com/software/utilities/"
        )

    files = list_files(ip)

    if patterns:
        if isinstance(patterns, str):
            patterns = [patterns]
        files_to_download = [
            f for f in files
            if any(pat in f for pat in patterns)
        ]
    elif date:
        pattern = re.compile(rf"{date}")
        files_to_download = [f for f in files if pattern.search(f)]
    else:
        raise ValueError("You must provide either `patterns` or `date`.")

    if not files_to_download:
        return []

    output_path = output_path or os.getcwd()
    os.makedirs(output_path, exist_ok=True)

    processed_files = []

    for filename in files_to_download:
        url = f"http://{ip}/api/ssd/download/{filename}"
        lifile = filename
        csvfile = lifile.replace(".li", ".csv")
        archive_name = f"{csvfile}.zip"

        # Download file
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(lifile, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        if convert:
            # Convert to CSV
            subprocess.run(["mokucli", "convert", lifile, "--format=csv"], check=True)

            if archive:
                archive_path = os.path.join(output_path, archive_name)
                with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    zipf.write(csvfile, arcname=os.path.basename(csvfile))
                os.remove(csvfile)
            else:
                shutil.move(csvfile, os.path.join(output_path, csvfile))

        if convert:
            os.remove(lifile)
        else:
            shutil.move(lifile, os.path.join(output_path, lifile))

        if remove_from_server:
            del_url = f"http://{ip}/api/ssd/delete/{filename}"
            response = requests.delete(del_url)
            response.raise_for_status()

        processed_files.append(filename)

    return processed_files


def upload(ip: str, paths: Union[str, List[str]]) -> Dict[str, bool]:
    """
    Upload one or more files to the Moku device's SSD.

    Parameters
    ----------
    ip : str
        IP address of the device (e.g., '10.128.100.198').
    paths : str or list of str
        Path to a local file or list of local file paths to upload.

    Returns
    -------
    dict of str to bool
        Dictionary mapping filenames to upload success status (True/False).

    Raises
    ------
    FileNotFoundError
        If a file path does not exist.
    requests.HTTPError
        If upload request fails.

    Notes
    -----
    - Uses HTTP POST with the file content as the body.
    - The upload endpoint is `/api/ssd/upload/<filename>`.
    - If the filename already exists on the device, it will be overwritten.
    """
    if isinstance(paths, str):
        paths = [paths]

    results = {}

    for file_path in paths:
        if not os.path.isfile(file_path):
            results[os.path.basename(file_path)] = False
            continue

        filename = os.path.basename(file_path)
        url = f"http://{ip}/api/ssd/upload/{filename}"

        with open(file_path, 'rb') as f:
            response = requests.post(url, data=f)

        response.raise_for_status()
        results[filename] = response.status_code == 200

    return results


def delete(
    ip: str,
    patterns: Optional[Union[str, List[str]]] = None,
    delete_all: bool = False,
    confirm: bool = False,
) -> List[str]:
    """
    Delete files from a Moku device, optionally by partial match, full list, or all files.

    Parameters
    ----------
    ip : str
        IP address of the device (e.g., '10.128.100.198').
    patterns : str or list of str, optional
        Partial filename string or list of substrings to match files. Ignored if `delete_all` is True.
    delete_all : bool, default False
        If True, delete all files on the device. Overrides `patterns`.
    confirm : bool, default False
        If False and files are found, raises ValueError to require explicit confirmation.
        Set to True to proceed without confirmation check.

    Returns
    -------
    list of str
        List of filenames that were successfully deleted.

    Raises
    ------
    ValueError
        If neither `patterns` nor `delete_all=True` is provided, or if `confirm=False`
        and files would be deleted.
    requests.HTTPError
        If delete request fails.

    Notes
    -----
    - For interactive use, set `confirm=True` or use the CLI/notebook wrappers.
    """
    files = list_files(ip)

    if delete_all:
        files_to_delete = files
    elif patterns:
        if isinstance(patterns, str):
            patterns = [patterns]
        files_to_delete = [f for f in files if any(pat in f for pat in patterns)]
    else:
        raise ValueError("Must specify `patterns` or set `delete_all=True`.")

    if not files_to_delete:
        return []

    if not confirm:
        raise ValueError(
            "Deletion requires confirmation. Set `confirm=True` to proceed, "
            f"or use CLI/notebook wrappers. Files to delete: {files_to_delete}"
        )

    deleted_files = []

    for f in files_to_delete:
        del_url = f"http://{ip}/api/ssd/delete/{f}"
        response = requests.delete(del_url)
        response.raise_for_status()
        deleted_files.append(f)

    return deleted_files


def parse_csv_file(
    filename: str,
    delimiter: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[int, int, int, List[str]]:
    """
    Parse a CSV file. It is potentially packaged in ZIP, TAR, GZ, or 7z format.

    Parameters
    ----------
    filename : str
        Location of the file
    delimiter : str, optional
        Delimiter to use when parsing. If None, auto-detection is attempted.
    logger : logging.Logger, optional
        Logger instance for debug messages.

    Returns
    -------
    num_cols : int
        Number of columns in the data
    num_rows : int
        Total number of rows (including headers)
    num_header_rows : int
        Number of detected header lines
    header : list
        List of header lines

    Raises
    ------
    ValueError
        If no valid data lines or header lines are found.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    def process_stream(file_obj):
        header_symbols = ['#', '%', '!', '@', ';', '&', '*', '/']
        header = []
        num_header_rows = 0
        num_rows = 0
        data_lines_sample = []
        num_cols = None

        # Wrap binary streams in text wrapper
        if isinstance(file_obj.read(0), bytes):
            file_obj = TextIOWrapper(file_obj, encoding='utf-8')
        file_obj.seek(0)

        for line in file_obj:
            num_rows += 1
            if any(line.startswith(symbol) for symbol in header_symbols):
                header.append(line)
                num_header_rows += 1
            else:
                # Capture a few non-header lines to detect delimiter
                if len(data_lines_sample) < 5 and line.strip():
                    data_lines_sample.append(line)
                # Try to determine number of columns from the first non-empty, non-header line
                if num_cols is None and line.strip():
                    try:
                        sniffed = csv.Sniffer().sniff(''.join(data_lines_sample))
                        detected_delimiter = sniffed.delimiter
                    except csv.Error:
                        detected_delimiter = delimiter if delimiter else ','
                    num_cols = len(line.strip().split(detected_delimiter))
        if num_cols is None:
            raise ValueError("No valid data lines found to determine column count.")
        return num_cols, num_rows, num_header_rows, header

    def process_file(path):
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path, 'r') as zip_ref:
                first_file_name = zip_ref.namelist()[0]
                with zip_ref.open(first_file_name, 'r') as f:
                    return process_stream(f)
        elif tarfile.is_tarfile(path):
            with tarfile.open(path, 'r') as tar_ref:
                first_member = tar_ref.getmembers()[0]
                with tar_ref.extractfile(first_member) as f:
                    return process_stream(f)
        elif path.endswith('.gz'):
            with gzip.open(path, 'rb') as f:
                return process_stream(f)
        elif path.endswith('.7z'):
            with SevenZipFile(path, 'r') as seven_zip_ref:
                first_file_name = seven_zip_ref.getnames()[0]
                with tempfile.TemporaryDirectory() as temp_dir:
                    seven_zip_ref.extract(path=temp_dir, targets=[first_file_name])
                    extracted_file = os.path.join(temp_dir, first_file_name)
                    with open(extracted_file, 'rb') as f:
                        return process_stream(f)
        else:
            with open(path, 'r', encoding='utf-8') as f:
                return process_stream(f)

    logger.debug(f"Reading from file: {filename}")
    num_cols, num_rows, num_header_rows, header = process_file(filename)

    if num_header_rows == 0:
        raise ValueError("No header lines detected. Ensure the file format is correct.")

    logger.debug(
        f"File contains {num_rows} total rows, {num_header_rows} header rows, "
        f"and {num_cols} columns"
    )
    return num_cols, num_rows, num_header_rows, header


def is_mat_file(file_path: str) -> bool:
    """
    Check if a file is a MATLAB .mat file by attempting to read its contents.

    Parameters
    ----------
    file_path : str
        Path to the file to check.

    Returns
    -------
    bool
        True if the file is a valid MATLAB .mat file, False otherwise.
    """
    try:
        scipy.io.whosmat(file_path)  # Try reading variable names in the file
        return True
    except Exception:
        return False


def is_li_file(file_path: str) -> bool:
    """
    Check if a file is a Liquid Instruments .li file by reading the header.

    Parameters
    ----------
    file_path : str
        Path to the file to check.

    Returns
    -------
    bool
        True if the file appears to be a .li file, False otherwise.
    """
    try:
        with open(file_path, 'rb') as f:
            header = f.read(8)  # Read first few bytes; adjust as needed
            # Example check: file starts with ASCII 'LI' or a known binary signature
            return header.startswith(b'LI')  # Adjust condition based on actual file format
    except Exception:
        return False


def read_lines(filename: str, num_lines: int) -> List[str]:
    """
    Read from file, return a number of lines as list.

    Parameters
    ----------
    filename : str
        Location of the file
    num_lines : int
        Number of lines to read from the file

    Returns
    -------
    list of str
        List of strings with `num_lines` lines from file (may be fewer if file is shorter).

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    IOError
        If the file cannot be read.
    """
    lines = []
    with open(filename, 'r', encoding='utf-8') as file:
        for _ in range(num_lines):
            line = file.readline()
            if not line:
                break
            lines.append(line.strip())
    return lines


def get_columns_with_nans(df) -> Dict[str, int]:
    """
    Find columns with NaNs in a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to analyze.

    Returns
    -------
    dict of str to int
        Dictionary mapping column names to their index positions for columns containing NaNs.
    """
    columns_with_nans = {}
    for column in df.columns:
        if df[column].isna().any():
            # Get the column number
            column_number = df.columns.get_loc(column)
            columns_with_nans[column] = column_number
    return columns_with_nans
