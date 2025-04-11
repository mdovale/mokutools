VERSION = '1.02'
"""
mokusync: Synchronization of two Moku:Pro phasemeter data streams

The two Moku share a clock, but their data streams are 
misaligned by a non-integer number of samples

Miguel Dovale (mdovale@arizona.edu)
Tucson, 2025
"""
from mokutools.filetools import *
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pytdi.dsp import timeshift
from spectools import lpsd
import multiprocessing
from synctools.sync import sync_signals
import logging
logger = logging.getLogger(__name__)

# : ===== User variables ====================
START_TIME = 100.0 # Seconds to crop from the beginning of the two files
DURATION = 0.5*3600 # Measurement duration to analyze
MASTER_COL = '2_phase' # Quantity on master device to use for sync (<channel>_<freq/phase>)
SLAVE_COL = '2_phase' # Quantity on slave device to use for sync (<channel>_<freq/phase>)
"""
The options for MASTER_COL and SLAVE_COL are '<ch>_<signal>':
    '1_phase', '2_phase', '3_phase', '4_phase'
    '1_freq', '2_freq', '3_freq', '4_freq'
"""
MODEL = 'fluc' # Model for TDIR-like synchronization ('fluc' or 'total', default: 'fluc')
DOMAIN = 'freq' # Domain for TDIR-like synchronization ('freq' or 'time', default: 'time')
SOLVER = 'Nelder-Mead' # Solver to use in scipy.optimize.minimize for TDIR-like synchronization
INTERP_ORDER = 121 # Interpolation order for TDIR-like synchronization (default: 121)
RESDIR = './results' # Where to store outputs
FILENAME = 'synced-data.csv' # Name of output data file
PLOTS = True # Generate plots or not
# Plot options
title = 'Moku-synchronization' # Title to use in plots
figsize = (6,4) # Figure size (inches)
dpi = 300 # Pixel density
fontsize = 8 # Font size
linewidth = 1.5 # Linewidth

# : ===== LPSD parameters ====================
p_lpsd = {"olap":"default",
          "bmin":1.0,
          "Lmin":1,
          "Jdes":500,
          "Kdes":100,
          "order":2,
          "win":np.kaiser,
          "psll":250}

# : ===== Main program ====================
def main():
    logger.debug(f"moku-sync v{VERSION} - Starting up...")
    
    # Multiprocessing
    pool = multiprocessing.Pool()
    p_lpsd["pool"] = pool

    # The program only looks for files in the current directory
    logger.debug(f"Looking for CSV files in {os.getcwd()}:")
    files_in_directory = os.listdir()
    file_list = sorted([file for file in files_in_directory if (file.endswith('.csv') or file.endswith('.mat'))])

    # Make results directory if it does not exist
    if not os.path.exists(RESDIR):
        os.makedirs(RESDIR)

    # Did we find files?
    if not file_list:
        logger.error("Error: No CSV or MAT files found in the current directory.")
        pool.close()
        pool.join()
        sys.exit(1)
    elif len(file_list) == 1:
        logger.error("Error: Not enough CSV or MAT files found in the current directory.")
        pool.close()
        pool.join()
        sys.exit(1)
    else:
        while True:
            display_menu(file_list)
            action, selected_files = get_user_choice(file_list)
            
            if action == 'Q':
                pool.close()
                pool.join()
                logger.debug('Done!')
                sys.exit(0)
            elif action == 'F':
                file1 = selected_files[0]
                file2 = selected_files[1]
                break

        logger.debug('Files selected:')
        logger.debug('    ' + file1 + ' (Master)')
        logger.debug('    ' + file2 + ' (Slave)')
    
    if is_mat_file(file1):
        logger.debug("Detected a MATLAB file, converting to CSV...")
        file1 = moku_mat_to_csv(file1)
        
    if is_mat_file(file2):
        logger.debug("Detected a MATLAB file, converting to CSV...")
        file2 = moku_mat_to_csv(file2)

    # Parse header of the two input files
    date1, fs, header1, cols1 = parse_header(file1)
    date2, fs2, header2, cols2 = parse_header(file2)

    labels1 = ['time']
    for i in range((cols1-1)//NCOLS_PER_CHANNEL):
        labels1 = labels1 + [f'1_{i+1}_set_f', f'1_{i+1}_freq', f'1_{i+1}_phase', f'1_{i+1}_i', f'1_{i+1}_q']

    labels2 = ['time']
    for i in range((cols2-1)//NCOLS_PER_CHANNEL):
        labels2 = labels2 + [f'2_{i+1}_set_f', f'2_{i+1}_freq', f'2_{i+1}_phase', f'2_{i+1}_i', f'2_{i+1}_q']

    # Check: are the files sampled at the same frequency?
    if fs != fs2:
        logger.error("Error: The input files report different sampling frequencies")
        pool.close()
        pool.join()
        sys.exit(1)

    # Figure out initial time offset from file metadata
    dt_datetime = date1 - date2 # Slave - Master
    dt_seconds = dt_datetime.total_seconds()
    if dt_seconds < 0:
        logger.debug(f"Time offset from metadata: {dt_seconds:.2f} s, Master device ahead")
    else:
        logger.debug(f"Time offset from metadata: {dt_seconds:.2f} s, Slave device ahead")
    if abs(dt_seconds) > 100.0:
        logger.warning(f"Warning: The initial time offset of {dt_seconds:.2f} seconds is very high!")

    # Figure out data length
    start_row = int(START_TIME*fs)
    end_row = start_row + int(DURATION*fs)
    n_rows = end_row - start_row
    real_duration = n_rows/fs
    logger.debug(f"Attempting to analyze {real_duration:.2f} s ({n_rows} rows) starting after {START_TIME:.2f} s (row {start_row})")

    # Data in
    logger.debug("Loading data, please wait...")

    # Read in file #1
    df1 = pd.read_csv(file1, delimiter=DELIMITER, skiprows=header1, nrows=end_row, names=labels1, engine='python')
    df1.drop(index=np.arange(start_row), inplace=True)
    df1.drop(labels='time', axis=1, inplace=True)
    logger.debug("    * Master device data loaded successfully")

    # Read in file #2
    df2 = pd.read_csv(file2, delimiter=DELIMITER, skiprows=header2, nrows=end_row, names=labels2, engine='python')
    df2.drop(index=np.arange(start_row), inplace=True)
    df2.drop(labels='time', axis=1, inplace=True)
    logger.debug("    * Slave device data loaded successfully")

    # Assert that the DataFrames are of the same length
    if len(df1) > len(df2):
        logger.error(f"Error: {file2} is shorter than required")
        pool.close()
        pool.join()
        sys.exit(1)
    elif len(df1) < len(df2):
        logger.error(f"Error: {file1} is shorter than required")
        pool.close()
        pool.join()
        sys.exit(1)

    # Find columns with NaNs in both DataFrames
    logger.debug("Looking for NaNs in data...")
    df1_nans = get_columns_with_nans(df1)
    df2_nans = get_columns_with_nans(df2)

    if len(df1_nans) > 0:
        for col, num in df1_nans.items():
            logger.warning(f"Warning: NaNs detected in {file1} column {num} ({col}) ")

    if len(df2_nans) > 0:
        for col, num in df2_nans.items():
            logger.warning(f"Warning: NaNs detected in {file2} column {num} ({col}) ")

    sig_master = '1_' + MASTER_COL
    if sig_master in df1_nans:
        logger.error("Error: The specified master device column contains NaNs")
        pool.close()
        pool.join()
        sys.exit(-1)

    sig_slave = '2_' + SLAVE_COL
    if sig_slave in df2_nans:
        logger.error("Error: The specified slave device column contains NaNs")
        pool.close()
        pool.join()
        sys.exit(-1)
    
    # Identify columns without NaNs in both DataFrames
    non_nan_columns_df1 = [col for col in df1.columns if col not in df1_nans]
    non_nan_columns_df2 = [col for col in df2.columns if col not in df2_nans]

    # Subset the DataFrames to only include non-NaN columns
    df1_non_nan = df1[non_nan_columns_df1]
    df2_non_nan = df2[non_nan_columns_df2]

    # Concatenate the DataFrames along columns (axis=1)
    logger.debug("Creating single DataFrame...")
    df = pd.concat([df1_non_nan.reset_index(drop=True), df2_non_nan.reset_index(drop=True)], axis=1)

    # Free up memory
    del df1, df2, df1_non_nan, df2_non_nan

    # Convert phase values from cycles to radians
    for sig in non_nan_columns_df1 + non_nan_columns_df2:
        if 'phase' in sig:
            df[sig] = df[sig]*2*np.pi 

    # When using phase signals for sync, convert them to frequency in Hertz by differentiation
    if 'phase' in sig_master:
        df[sig_master+'_to_freq'] = np.diff(df[sig_master], prepend=np.nan) / (2*np.pi/fs)
        sig_master = sig_master+'_to_freq'
        non_nan_columns_df1.append(sig_master)

    if 'phase' in sig_slave:
        df[sig_slave+'_to_freq'] = np.diff(df[sig_slave], prepend=np.nan) / (2*np.pi/fs)
        sig_slave = sig_slave+'_to_freq'
        non_nan_columns_df2.append(sig_slave)

    df = df.iloc[1:] # Remove first row of the DataFrame, it may contain NaNs from differentiation

    # Calculate and print RMS values of the two signals, useful for debugging
    rms1 = np.sqrt(np.mean(np.square(df[sig_master]-np.mean(df[sig_master]))))
    rms2 = np.sqrt(np.mean(np.square(df[sig_slave]-np.mean(df[sig_slave]))))
    logger.debug(f"    * RMS value of master signal: {rms1:.6}")
    logger.debug(f"    * RMS value of slave signal: {rms2:.6}")

    logger.debug("Calling synctools::sync_signals...")
    n_truncate = int(2*abs(dt_seconds*fs))
    unsync, sync = sync_signals([np.array(df[sig_master]), np.array(df[sig_slave])], 
                        fs, 
                        p_lpsd, 
                        [dt_seconds], 
                        MODEL, 
                        DOMAIN, 
                        SOLVER, 
                        INTERP_ORDER, 
                        n_truncate)

    dt = sync.timer_offsets[0]

    # Timeshift all signals from slave device
    logger.debug('Generating timeshifted outputs...')
    for sig in non_nan_columns_df2:
        df[sig+'_shifted'] = timeshift(np.array(df[sig]), np.full(len(df), dt*fs))

    # Truncate output
    df = df.iloc[n_truncate:-n_truncate] 

    # Form unsynced and synced signal combinations
    df['freq_unsynced'] = df[sig_master] - df[sig_slave]
    df['freq_synced'] = df[sig_master] - df[sig_slave+'_shifted']

    # Convert phase in radians to frequency in Hertz via integration
    df['phase_unsynced'] = (2*np.pi/fs)*np.cumsum(np.array(df['freq_unsynced']-np.mean(df['freq_unsynced'])))
    df['phase_synced'] = (2*np.pi/fs)*np.cumsum(np.array(df['freq_synced']-np.mean(df['freq_synced'])))

    # Linear detrend just in case
    df['phase_unsynced'] = df['phase_unsynced'] - np.polyval(np.polyfit(np.arange(len(df)), df['phase_unsynced'], 1), np.arange(len(df)))
    df['phase_synced'] = df['phase_synced'] - np.polyval(np.polyfit(np.arange(len(df)), df['phase_synced'], 1), np.arange(len(df)))

    df.insert(0, 'time', np.arange(len(df))/fs)

    # Save data to file
    logger.debug('Saving data...')

    metadata1 = read_lines(file1, header1-1)
    metadata2 = read_lines(file2, header2-1)

    metadata1[0] += ' (Master)'
    metadata2[0] += ' (Slave)'

    metadata = []
    for line in metadata1:
        metadata.append(line)
    for line in metadata2:
        metadata.append(line)
    metadata.append(f"% Synchronized with moku-sync v{VERSION}, dt = {dt:.14f} seconds ({dt*fs:.14f} samples)")

    output_file_path = os.path.join(RESDIR, FILENAME)

    # Ensure the directory exists
    os.makedirs(RESDIR, exist_ok=True)

    with open(output_file_path, 'w') as file:
        for line in metadata:
            file.write(line + '\n')

    with open(output_file_path, 'a') as file:
        df.to_csv(file, index=False)
    
    if PLOTS:
        # Time domain plots
        logger.debug('Plotting time series data...')
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.plot(df['time'], df['freq_unsynced'], linewidth=linewidth, label=r"Unsynced", color="gray");
        ax.plot(df['time'], df['freq_synced'], linewidth=linewidth, label=r"Synced", color="tomato");
        ax.set_xlabel("Time (s)", fontsize=fontsize);
        ax.set_ylabel("Frequency (Hz)", fontsize=fontsize);
        ax.set_title(title, fontsize=fontsize)
        ax.tick_params(labelsize=fontsize);
        ax.grid();
        ax.legend(loc='best', edgecolor='black', fancybox=True, shadow=True, framealpha=1, fontsize=fontsize, handlelength=2.5);
        fig.tight_layout();
        fig.savefig(os.path.join(RESDIR,'fig_freq_t.pdf'));
        logger.debug(f"    * Plot saved to {os.path.join(RESDIR,'fig_freq_t.pdf')}")

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.plot(df['time'], df['phase_unsynced'], linewidth=linewidth, label=r"Unsynced", color="gray");
        ax.plot(df['time'], df['phase_synced'], linewidth=linewidth, label=r"Synced", color="tomato");
        ax.set_xlabel("Time (s)", fontsize=fontsize);
        ax.set_ylabel("Phase (rad)", fontsize=fontsize);
        ax.set_title(title, fontsize=fontsize)
        ax.tick_params(labelsize=fontsize);
        ax.grid();
        ax.legend(loc='best', edgecolor='black', fancybox=True, shadow=True, framealpha=1, fontsize=fontsize, handlelength=2.5);
        fig.tight_layout();
        fig.savefig(os.path.join(RESDIR, 'fig_phase_t.pdf'));
        logger.debug(f"    * Plot saved to {os.path.join(RESDIR, 'fig_phase_t.pdf')}")

        # Compute spectrums with spectools
        logger.debug('Computing unsynced frequency spectrum...')
        psd_unsynced = lpsd.lpsd(np.array(df['freq_unsynced']), fs=fs,
            olap=p_lpsd['olap'], bmin=p_lpsd['bmin'], Lmin=p_lpsd['Lmin'], 
            Jdes=p_lpsd['Jdes'], Kdes=p_lpsd['Kdes'],
            order=p_lpsd['order'], win=p_lpsd['win'], psll=p_lpsd['psll'],
            return_type='object', pool=pool)
        
        logger.debug('Computing synced frequency spectrum...')
        psd_synced = lpsd.lpsd(np.array(df['freq_synced']), fs=fs,
            olap=p_lpsd['olap'], bmin=p_lpsd['bmin'], Lmin=p_lpsd['Lmin'], 
            Jdes=p_lpsd['Jdes'], Kdes=p_lpsd['Kdes'],
            order=p_lpsd['order'], win=p_lpsd['win'], psll=p_lpsd['psll'],
            return_type='object', pool=pool)
        logger.debug('Plotting spectrums...')
    
        # ASD plots
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.loglog(psd_unsynced.f, np.sqrt(psd_unsynced.Gxx), linewidth=linewidth, label=r"Unsynced", color="gray");
        ax.loglog(psd_synced.f, np.sqrt(psd_synced.Gxx), linewidth=linewidth, label=r"Synced", color="tomato");
        ax.set_xlim(psd_synced.f[0], psd_synced.f[-1])
        ax.set_xlabel("Fourier frequency (Hz)", fontsize=fontsize);
        ax.set_ylabel(r"Frequency ASD $\rm (Hz/Hz^{1/2})$", fontsize=fontsize);
        ax.set_title(title, fontsize=fontsize)
        ax.tick_params(labelsize=fontsize);
        ax.grid(which='both');
        ax.legend(loc='best', edgecolor='black', fancybox=True, shadow=True, framealpha=1, fontsize=fontsize, handlelength=2.5);
        fig.tight_layout();
        fig.savefig(os.path.join(RESDIR, 'fig_freq_asd.pdf'));
        logger.debug(f"    * Plot saved to {os.path.join(RESDIR, 'fig_freq_asd.pdf')}")

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.loglog(psd_unsynced.f, np.sqrt(psd_unsynced.Gxx)/psd_unsynced.f, linewidth=linewidth, label=r"Unsynced", color="gray");
        ax.loglog(psd_synced.f, np.sqrt(psd_synced.Gxx)/psd_synced.f, linewidth=linewidth, label=r"Synced", color="tomato");
        ax.set_xlim(psd_synced.f[0], psd_synced.f[-1])
        ax.set_xlabel("Fourier frequency (Hz)", fontsize=fontsize);
        ax.set_ylabel(r"Phase ASD $\rm (rad/Hz^{1/2})$", fontsize=fontsize);
        ax.set_title(title, fontsize=fontsize)
        ax.tick_params(labelsize=fontsize);
        ax.grid(which='both');
        ax.legend(loc='best', edgecolor='black', fancybox=True, shadow=True, framealpha=1, fontsize=fontsize, handlelength=2.5);
        fig.tight_layout();
        fig.savefig(os.path.join(RESDIR, 'fig_phase_asd.pdf'));
        logger.debug(f"    * Plot saved to {os.path.join(RESDIR, 'fig_phase_asd.pdf')}")

    pool.close()
    pool.join()
    logger.debug('Done!')




if __name__ == "__main__":
    main()