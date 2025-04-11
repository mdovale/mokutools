import numpy as np
import pandas as pd
from spectools import ltf
from mokutools.filetools import *
import logging
logger = logging.getLogger(__name__)

def moku_phasemeter_labels(nchan):
    labels = ['time']
    for i in range(nchan):
        labels = labels + [f'{i+1}_set_f', f'{i+1}_freq', f'{i+1}_phase', f'{i+1}_i', f'{i+1}_q']
    return labels

class MokuPhasemeterObject():
    def __init__(self, filename, start_time=None, duration=None):

        if is_mat_file(filename):
            self.filename = moku_mat_to_csv(filename)
        else:
            self.filename = filename

        with open(self.filename, 'r', encoding='utf-8') as f:
            self.nrows = sum(1 for line in f)

        logger.debug(f"Detected {self.nrows} rows of data")

        self.date, self.fs, self.nrows_header, self.ncols = parse_header(filename)
        self.nchan = (self.ncols-1)//NCOLS_PER_CHANNEL
        
        logger.debug(f"Detected {self.nchan} phasemeter channels")

        self.labels = moku_phasemeter_labels(self.nchan)

        # Figure out data length
        self.start_time = start_time if start_time is not None else 0.0
        self.start_row = int(start_time*self.fs) if start_time is not None else 0
        self.end_row = self.start_row + int(duration*self.fs) if duration is not None else self.nrows - self.nrows_header
        self.ndata = self.end_row - self.start_row
        self.duration = self.ndata/self.fs
        logger.debug(f"Attempting to load {self.duration:.2f} s ({self.ndata} rows) starting after {self.start_time:.2f} s (row {self.start_row})")
        logger.debug("Loading data, please wait...")

        # Read in file
        self.df = pd.read_csv(self.filename, delimiter=DELIMITER, skiprows=self.nrows_header+self.start_row, nrows=self.ndata, names=self.labels, engine='python')
        logger.debug("    * Moku phasemeter data loaded successfully")

def frequency_spectral_analysis():
    return

def phase_spectral_analysis():
    return