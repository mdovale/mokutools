# BSD 3-Clause License

# Copyright (c) 2025, Miguel Dovale

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

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

# This software may be subject to U.S. export control laws. By accepting this
# software, the user agrees to comply with all applicable U.S. export laws and
# regulations. User has the responsibility to obtain export licenses, or other
# export authority as may be required before exporting such information to
# foreign countries or providing access to foreign persons.

DELIMITER = ', ' # Data delimiter in the CSV files
NCOLS_PER_CHANNEL = 5 # Number of data columns per phasemeter channel

import numpy as np
import pandas as pd
from speckit import compute_spectrum as ltf
import speckit.dsp as dsp
from mokutools.moku_io.core import (
    list_files,
    download,
    parse_csv_file,
    resolve_mat_path,
    is_li_file,
)
import logging
import os
import subprocess
import ipywidgets as widgets
from IPython.display import display
import zipfile
import scipy.io


class MokuPhasemeterObject:
    """
    Class for loading, parsing, and analyzing data acquired from a Moku:Pro Phasemeter.

    This class handles `.csv`, `.mat`, or `.zip` archives containing a single `.mat` file.
    It extracts header and channel information,
    computes derived quantities (e.g., phase in radians), and can compute spectral density
    estimates for various time series.

    Args:
        filename (str): 
            Path to a `.csv`, `.mat`, or `.zip` (single `.mat` inside) data file from the
            Moku Phasemeter, or a partial file name when `ip` is provided.
        start_time (float, optional): 
            Start time (in seconds) for loading a subset of the data. Defaults to 0.
        duration (float, optional): 
            Duration (in seconds) of data to load. If not specified, loads until the end.
        prefix (str, optional): 
            String prefix to prepend to each column label in the data frame.
        spectrums (list, optional): 
            List of spectrum types to precompute (e.g., ['phase', 'frequency']).
        logger (logging.Logger, optional): 
            Logger for debug messages. If None, a default logger is used.
        ip (str, optional): 
            IP address of the Moku device to download the file from. If provided, `filename`
            is interpreted as a substring to match on the device.
        output_path (str, optional): 
            Directory where output files will be saved. Defaults to current directory.
        archive_file (bool): 
            If True, compress the converted `.csv` file into a `.zip`. Default is False.
        delete_original (bool): 
            If True, delete the original `.li` or `.mat` file. Default is False.
        remove_from_server (bool): 
            If True, delete the `.li` file from the device after processing. Default is False.
        *args, **kwargs: 
            Additional arguments passed to the spectral estimation function (`ltf`).

    Attributes:
        fs (float): 
            Sampling frequency in Hz.
        date (str): 
            Date string extracted from the file header.
        df (pandas.DataFrame): 
            Loaded and processed data frame.
        nchan (int): 
            Number of phasemeter channels detected.
        labels (list): 
            List of data column labels.
        ps (dict): 
            Dictionary of power spectral density results keyed by 'channel_metric'.
    """

    def __init__(self, filename=None, start_time=None, duration=None, prefix=None, spectrums=[], logger=None,
                 ip=None, output_path=None, archive_file=False, delete_original=False, remove_from_server=False, *args, **kwargs):

        if logger is None:
            logger = logging.getLogger(__name__)

        # Handle file download from a Moku server
        if ip is not None:
            if filename is None:
                raise ValueError("If 'ip' is provided, 'filename' must also be specified.")

            logger.debug(f"Fetching file list from Moku device at {ip}...")
            files = list_files(ip)
            matches = [f for f in files if filename in f]

            if len(matches) == 0:
                raise FileNotFoundError(f"No files matching '{filename}' found on device at {ip}.")
            elif len(matches) == 1:
                selected_file = matches[0]
                logger.debug(f"Single match found: {selected_file}")
            else:
                logger.info("Multiple matching files found. Choosing the first one.")
                selected_file = matches[0]

            logger.debug(f"Downloading selected file: {selected_file}")
            download(
                ip=ip,
                patterns=selected_file,
                convert=False,
                archive=False,
                output_path=output_path,
                remove_from_server=remove_from_server
            )
            self.filename = os.path.join(output_path or os.getcwd(), os.path.basename(selected_file))
            is_downloaded_file = True
        else:
            if filename is None:
                raise ValueError("Either 'filename' or both 'ip' and 'filename' must be specified.")
            self.filename = filename
            is_downloaded_file = False

        # Handle file conversions to CSV
        is_converted_file = False

        mat_path = resolve_mat_path(self.filename)
        if mat_path is not None:
            original_file = self.filename
            is_converted_file = True
            logger.debug(
                f"{self.filename} is MATLAB data, converting to CSV for further processing..."
            )
            self.filename = mat_to_csv(mat_path)

        elif is_li_file(self.filename):
            original_file = self.filename
            is_converted_file = True
            logger.debug(f"{self.filename} is a Liquid:Instruments binary file, converting to CSV for further processing...")
            subprocess.run(["mokucli", "convert", self.filename, "--format=csv"], check=True)
            self.filename = os.path.splitext(self.filename)[0] + ".csv"

        # Archive the converted file if requested
        if is_converted_file and archive_file:
            zip_path = self.filename + ".zip"
            logger.debug(f"Archiving {self.filename} to {zip_path}")
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(self.filename, arcname=os.path.basename(self.filename))
            os.remove(self.filename)  # Delete original CSV
            self.filename = zip_path

        # Delete the original file if requested
        if is_converted_file and delete_original:
            logger.debug(f"Removing {original_file}")
            os.remove(original_file)

        # Parse header only (avoids full file scan; single-pass load below)
        self.ncols, self.nrows, self.header_rows, self.header = parse_csv_file(
            self.filename, delimiter=DELIMITER, logger=logger, header_only=True
        )
        self.fs, self.date = parse_header(self.header, logger=logger)
        self.nchan = (self.ncols - 1) // NCOLS_PER_CHANNEL
        logger.debug(f"Detected {self.nchan} phasemeter channels")

        self.labels = self.data_labels()

        # Time slicing
        self.start_time = start_time if start_time is not None else 0.0
        self.start_row = int(start_time * self.fs) if start_time is not None else 0
        ndata_to_load = (
            int(duration * self.fs) if duration is not None else None
        )
        self.end_row = (
            self.start_row + ndata_to_load
            if ndata_to_load is not None
            else None
        )

        logger.debug(
            f"Attempting to load {ndata_to_load or 'all'} rows starting after "
            f"{self.start_time:.2f} s (row {self.start_row})"
        )
        logger.debug("Loading data, please wait...")

        # C engine with sep=',' + skipinitialspace parses ', ' faster than Python engine.
        # Explicit dtype skips inference and speeds up large-file loading.
        self.df = pd.read_csv(
            self.filename,
            sep=',',
            skipinitialspace=True,
            skiprows=self.header_rows + self.start_row,
            nrows=ndata_to_load,
            names=self.labels,
            dtype={label: np.float64 for label in self.labels},
            engine='c',
        )
        if (is_converted_file) and (not archive_file):
            os.remove(self.filename)

        self.ndata = len(self.df)
        self.end_row = (
            self.start_row + self.ndata if self.end_row is None else self.end_row
        )
        self.nrows = self.header_rows + self.start_row + self.ndata
        self.duration = self.ndata / self.fs

        logger.debug(f"    * Moku phasemeter data loaded successfully")
        logger.debug(f"    * Loaded {self.ndata} rows, {self.duration:.2f} seconds")
        logger.debug(f"\n{self.df.head()}")

        # Derived quantities (batched to avoid per-column copy)
        assign_dict = {}
        for i in range(self.nchan):
            assign_dict[f'{i+1}_phase'] = self.df[f'{i+1}_cycles'] * 2 * np.pi
            assign_dict[f'{i+1}_freq2phase'] = dsp.frequency2phase(
                self.df[f'{i+1}_freq'], self.fs
            )
        self.df = self.df.assign(**assign_dict)

        # Optional spectrum computation
        self.ps = {}
        if len(spectrums) > 0:
            self.spectrum(spectrums, *args, **kwargs)

        # Optional label prefixing (in-place column rename, no full copy)
        if prefix is not None:
            self.df.columns = [f"{prefix}{c}" for c in self.df.columns]

    def data_labels(self):
        """
        Generate column labels for the phasemeter data based on detected channels.

        Returns:
            list:
            A list of strings representing the expected CSV column headers, including
            time, set frequency, measured frequency, phase (in cycles), I and Q values
            for each channel.
        """
        labels = ['time']
        for i in range(self.nchan):
            labels += [f'{i+1}_set_freq', f'{i+1}_freq', f'{i+1}_cycles', f'{i+1}_i', f'{i+1}_q']
        return labels

    def spectrum(self, which='phase', channels=[], *args, **kwargs):
        """
        Compute and store power spectral density estimates for specified data channels.

        Args:
            which (str or list of str): 
                Type(s) of data to analyze. Options include:
                - 'phase', 'frequency', 'freq2phase' (interpreted per channel)
                - Or an exact name of a column in self.df
            channels (list or int, optional): 
                List of integer channel numbers (1-indexed) to analyze. 
                If empty, all channels are included.
            *args, **kwargs: 
                Passed directly to the spectral estimation function (`ltf`).

        Returns:
            None
            Updates the `self.ps` dictionary in-place with the computed spectra.

        Raises:
            ValueError:
                If the specified channel(s) do not exist in the loaded data.
        """
        in_channels = channels

        if isinstance(which, str):
            which = [which]

        # Handle single-column case if it exactly matches a column name
        if len(which) == 1 and which[0] in self.df.columns:
            col = which[0]
            self.ps[col] = ltf(self.df[col], fs=self.fs, *args, **kwargs)
            return

        if isinstance(channels, int):
            channels = [channels]
        elif not channels:
            channels = list(range(1, self.nchan + 1))
        else:
            channels = list(channels)

        channels = [ch for ch in channels if 1 <= ch <= self.nchan]

        if not channels:
            raise ValueError(f"A channel specified ({in_channels}) is not present")

        for i in channels:
            if any(key in which for key in ('frequency', 'freq', 'f')):
                self.ps[f'{i}_freq'] = ltf(self.df[f'{i}_freq'], fs=self.fs, *args, **kwargs)
            if any(key in which for key in ('phase', 'p')):
                self.ps[f'{i}_phase'] = ltf(self.df[f'{i}_phase'], fs=self.fs, *args, **kwargs)
            if any(key in which for key in ('frequency2phase', 'freq2phase', 'f2p')):
                self.ps[f'{i}_freq2phase'] = ltf(self.df[f'{i}_freq2phase'], fs=self.fs, *args, **kwargs)

def parse_header(file_header, row_fs=None, row_t0=None, fs_hint="rate", t0_hint="Acquired", logger=None):
    """
    Parse a Moku phasemeter CSV file header.

    Args:
        header (str): The file header
        row_fs (int, optional): Row number containing acquisition rate
        row_t0 (int, optional): Row number containing start time
        fs_hint (str, optional): String hint to locate the acquisition rate line (default: "rate")
        t0_hint (str, optional): String hint to locate the start time line (default: "Acquired")
    
    Returns:
        date (pd.Timestamp): Start time reported in the file
        fs (float): Sampling frequency
        num_header_lines (int): Number of detected header lines
        num_columns (int): Number of columns in the data
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    fs = None
    date = None
    num_header_rows = len(file_header)

    # First attempt to use row_fs if provided
    if row_fs is not None and row_fs <= num_header_rows:
        try:
            fs = float(file_header[row_fs-1].split(': ')[1].split(' ')[0])
        except (IndexError, ValueError):
            logger.warning(f"Failed to parse fs from row {row_fs}, falling back to hint search.")

    # If fs is not found, use hint search
    if fs is None:
        for line in file_header[:num_header_rows]:
            if fs_hint in line:
                try:
                    fs = float(line.split(': ')[1].split(' ')[0])
                    break
                except (IndexError, ValueError):
                    logger.warning(f"Failed to parse fs from line containing {fs_hint}.")

    logger.debug(f'Moku phasemeter metadata:')
    logger.debug(f'    fs = {fs}')

    # First attempt to use row_t0 if provided
    if row_t0 is not None and row_t0 <= num_header_rows:
        try:
            date = pd.to_datetime(file_header[row_t0-1].split(f'% {t0_hint} ')[1].strip())
        except (IndexError, ValueError):
            logger.warning(f"Failed to parse t0 from row {row_t0}, falling back to hint search.")

    # If t0 is not found, use hint search
    if date is None:
        for line in file_header[:num_header_rows]:
            if t0_hint in line:
                try:
                    date = pd.to_datetime(line.split(f'% {t0_hint} ')[1].strip())
                    break
                except (IndexError, ValueError):
                    logger.warning(f"Failed to parse t0 from line containing {t0_hint}.")

    logger.debug(f'    t0 = {date}')
    
    return fs, date


def _extract_moku_header_and_data(mat_data):
    """
    Extract header text and numeric data array from a Moku phasemeter .mat file.

    Supports the nested layouts produced by MATLAB/Moku exports (1x2 and 2x1
    object arrays inside ``moku[0, 0]``).
    """
    block = mat_data['moku'][0, 0]
    header_elem = None
    data_elem = None

    for item in np.ravel(block):
        arr = np.asarray(item)
        if arr.dtype == object and arr.size == 1:
            arr = np.asarray(arr.flat[0])
        if np.issubdtype(arr.dtype, np.character):
            header_elem = arr
        elif np.issubdtype(arr.dtype, np.number) and arr.ndim >= 2:
            data_elem = arr

    if header_elem is None or data_elem is None:
        try:
            header = str(mat_data['moku'][0][0][0][0][:-2])
            data_elem = mat_data['moku'][0][0][1]
            if isinstance(data_elem, np.ndarray) and data_elem.dtype == object:
                data_elem = np.asarray(data_elem.flat[0])
        except (IndexError, TypeError) as exc:
            raise ValueError(
                "Could not parse Moku phasemeter .mat structure in 'moku' variable"
            ) from exc
    else:
        header = str(np.ravel(header_elem)[0][:-2])

    return header, data_elem


def mat_to_csv(mat_file, out_file=None):
    """
    Convert a MATLAB `.mat` file generated by a Moku:Pro phasemeter into a CSV file.

    Args:
        mat_file (str): Path to the input `.mat` file, or a `.zip` archive containing
            exactly one `.mat` file, with the Moku data.
        out_file (str, optional): Path to the output CSV file. If not provided, 
        the function will save the CSV file with the same name as `mat_file` 
        but with a `.csv` extension.

    Returns:
        None
        The function writes the extracted data to a CSV file and does not return a value.

    Notes:
    ------
    - The function expects a ``moku`` variable with header and numeric data arrays
      nested under ``moku[0, 0]`` (layouts vary between MATLAB export versions).
    - The extracted header is assumed to be a string and is stripped of its last newline.
    - The data is saved with six decimal places of precision.

    """
    mat_path = resolve_mat_path(mat_file)
    if mat_path is None:
        raise ValueError(f"Not a MATLAB .mat file or zip with a single .mat member: {mat_file}")
    mat_data = scipy.io.loadmat(mat_path)
    header, data_array = _extract_moku_header_and_data(mat_data)

    if out_file is None:
        base, _ = os.path.splitext(mat_path)
        out_file = base + '.csv'

    with open(out_file, 'w', newline='') as f:
        f.write(header.rstrip() + '\n')
        for row in np.asarray(data_array):
            f.write(
                DELIMITER.join(f'{val:.14f}' for val in row) + '\n'
            )

    return out_file
