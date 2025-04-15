import numpy as np
import pandas as pd
from spectools.lpsd import ltf
import spectools.dsp as dsp
from mokutools.filetools import *
import logging
logger = logging.getLogger(__name__)

def moku_phasemeter_labels(nchan):
    labels = ['time']
    for i in range(nchan):
        labels = labels + [f'{i+1}_set_freq', f'{i+1}_freq', f'{i+1}_cycles', f'{i+1}_i', f'{i+1}_q']
    return labels

class MokuPhasemeterObject():
    def __init__(self, filename, start_time=None, duration=None, prefix=None):

        if is_mat_file(filename):
            self.filename = moku_mat_to_csv(filename)
        else:
            self.filename = filename

        self.ncols, self.nrows, self.header_rows, self.header = parse_csv_file(filename)

        self.fs, self.date = parse_moku_phasemeter_header(self.header)
        
        self.nchan = (self.ncols-1)//NCOLS_PER_CHANNEL
        
        logger.debug(f"Detected {self.nchan} phasemeter channels")

        self.labels = moku_phasemeter_labels(self.nchan)

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

        for i in range(self.nchan): # Convert phase in cycles to phase in radians
            self.df[f'{i+1}_phase'] = self.df[f'{i+1}_cycles']*2*np.pi

        for i in range(self.nchan): # Integrate frequency in Hz to find phase in radians
            self.df[f'{i+1}_freq2phase'] = dsp.frequency2phase(self.df[f'{i+1}_freq'], self.fs)

        if prefix is not None:
            self.df = self.df.add_prefix(prefix)
        

def frequency_spectral_analysis():
    return

def phase_spectral_analysis():
    return