import numpy as np
import pandas as pd
from spectools.lpsd import ltf
import spectools.dsp as dsp
from mokutools.filetools import *
import logging

class MokuPhasemeterObject():
    """
    Class for loading, parsing, and analyzing data acquired from a Moku:Pro Phasemeter.

    This class handles `.csv` or `.mat` files, extracts header and channel information,
    computes derived quantities (e.g., phase in radians), and can compute spectral density
    estimates for various time series.

    Args:
        filename (str): 
            Path to the `.csv` or `.mat` data file acquired from the Moku Phasemeter.
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
    def __init__(self, filename, start_time=None, duration=None, prefix=None, spectrums=[], logger=None, *args, **kwargs):

        if logger is None:
            logger = logging.getLogger(__name__)

        if is_mat_file(filename):
            logger.debug(f"{filename} is a Matlab file, converting to CSV for further processing...")
            self.filename = moku_mat_to_csv(filename)
        else:
            self.filename = filename

        self.ncols, self.nrows, self.header_rows, self.header = parse_csv_file(self.filename, logger=logger)

        self.fs, self.date = parse_header(self.header, logger=logger)
        
        self.nchan = (self.ncols-1)//NCOLS_PER_CHANNEL
        
        logger.debug(f"Detected {self.nchan} phasemeter channels")

        self.labels = self.data_labels()

        self.start_time = start_time if start_time is not None else 0.0
        self.start_row = int(start_time*self.fs) if start_time is not None else 0
        self.end_row = self.start_row + int(duration*self.fs) if duration is not None else self.nrows - self.header_rows
        self.ndata = self.end_row - self.start_row
        self.duration = self.ndata/self.fs
        logger.debug(f"Attempting to load {self.duration:.2f} s ({self.ndata} rows) starting after {self.start_time:.2f} s (row {self.start_row})")
        logger.debug("Loading data, please wait...")

        self.df = pd.read_csv(
            self.filename, 
            delimiter=DELIMITER, 
            skiprows=self.header_rows + self.start_row, 
            nrows=self.ndata, 
            names=self.labels, 
            engine='python'
        )
        
        if len(self.df) != self.ndata:
            self.end_row = len(self.df) - 1
            self.ndata = len(self.df)
            
        self.duration = self.ndata / self.fs

        logger.debug(f"    * Moku phasemeter data loaded successfully")
        logger.debug(f"    * Loaded {self.ndata} rows, {self.duration} seconds")
        logger.debug(f"\n{self.df.head()}")

        for i in range(self.nchan):
            self.df[f'{i+1}_phase'] = self.df[f'{i+1}_cycles'] * 2 * np.pi

        for i in range(self.nchan):
            self.df[f'{i+1}_freq2phase'] = dsp.frequency2phase(self.df[f'{i+1}_freq'], self.fs)

        self.ps = {}

        if len(spectrums) > 0:
            self.spectrum(spectrums, *args, **kwargs)

        if prefix is not None:
            self.df = self.df.add_prefix(prefix)

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
                Type(s) of data to analyze. Options include 'phase', 'frequency', 'freq2phase'.
                Can be a string or list of strings.
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