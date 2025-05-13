import numpy as np
import pandas as pd
from spectools.lpsd import ltf
import spectools.dsp as dsp
from mokutools.filetools import *
import logging

class MokuPhasemeterObject():
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

        # Figure out data length
        self.start_time = start_time if start_time is not None else 0.0
        self.start_row = int(start_time*self.fs) if start_time is not None else 0
        self.end_row = self.start_row + int(duration*self.fs) if duration is not None else self.nrows - self.header_rows
        self.ndata = self.end_row - self.start_row
        self.duration = self.ndata/self.fs
        logger.debug(f"Attempting to load {self.duration:.2f} s ({self.ndata} rows) starting after {self.start_time:.2f} s (row {self.start_row})")
        logger.debug("Loading data, please wait...")

        # Read in file
        self.df = pd.read_csv(self.filename, delimiter=DELIMITER, skiprows=self.header_rows+self.start_row, nrows=self.ndata, names=self.labels, engine='python')
        
        if len(self.df) != self.ndata:
            self.end_row = len(self.df) - 1
            self.ndata = len(self.df)
            
        self.duration = self.ndata/self.fs

        logger.debug(f"    * Moku phasemeter data loaded successfully")
        logger.debug(f"    * Loaded {self.ndata} rows, {self.duration} seconds")
        logger.debug(f"\n{self.df.head()}")

        for i in range(self.nchan): # Convert phase in cycles to phase in radians
            self.df[f'{i+1}_phase'] = self.df[f'{i+1}_cycles']*2*np.pi

        for i in range(self.nchan): # Integrate frequency in Hz to find phase in radians
            self.df[f'{i+1}_freq2phase'] = dsp.frequency2phase(self.df[f'{i+1}_freq'], self.fs)

        self.ps = {}

        if len(spectrums) > 0:
            self.spectrum(spectrums, *args, **kwargs)

        if prefix is not None:
            self.df = self.df.add_prefix(prefix)

    def data_labels(self):
        labels = ['time']
        for i in range(self.nchan):
            labels = labels + [f'{i+1}_set_freq', f'{i+1}_freq', f'{i+1}_cycles', f'{i+1}_i', f'{i+1}_q']
        return labels

    def spectrum(self, which='phase', channels=[], *args, **kwargs):
        in_channels = channels

        if isinstance(which, str):
            which = [which]

        # Normalize channels
        if isinstance(channels, int):
            channels = [channels]
        elif not channels:  # If empty, default to all channels
            channels = list(range(1, self.nchan + 1))
        else:
            channels = list(channels)

        # Sanity check
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