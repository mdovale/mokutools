import sys
import csv
import os
import re
import shutil
import zipfile
import requests
import subprocess
from io import TextIOWrapper
import ipywidgets as widgets
from IPython.display import display, clear_output
import scipy.io
import zipfile
import tarfile
import gzip
from py7zr import SevenZipFile
import numpy as np
import logging

SERVER_URL = "http://10.128.100.198/api/ssd"
DATA_DIR = "./data"

def get_file_list(ip):
    """
    Fetch the list of files available from the Moku server at the specified IP address.

    Args:
        ip (str): IP address of the device (e.g., '10.128.100.198').

    Returns:
        list: List of filenames available from the server.

    Notes:
    ------
    - The function sends a GET request to `http://<ip>/api/ssd/list` and parses 
      the JSON response.
    - If the request fails or the response format is incorrect, the function may 
      raise an exception.
    - It is assumed that the response contains a `data` field holding the file list.
    """
    url = f"http://{ip}/api/ssd/list"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    return data.get("data", [])

def download_files(ip, file_names=None, date=None, convert=True, archive=True, output_path=None, delete=False):
    """
    Download `.li` files from a Moku device and optionally convert, compress, and delete them.

    Args:
        ip (str): 
            IP address of the device (e.g., '10.128.100.198').
        file_names (str or list of str, optional): 
            Partial filename or list of partial strings to match files.
            If provided, the `date` argument is ignored.
        date (str, optional): 
            A date string in 'YYYYMMDD' format to filter filenames.
        convert (bool): 
            If True, convert the `.li` file to `.csv` using `mokucli`. Default is True.
        archive (bool): 
            If True, zip the `.csv` file. Applies only if `convert=True`. Default is True.
        output_path (str, optional): 
            Directory where output files will be saved. Defaults to current directory.
        delete (bool): 
            If True, delete the `.li` file from the device after processing. Default is False.

    Returns:
        None
            This function processes files as described but returns no value.

    Notes:
    ------
    - Requires `mokucli` to be installed and available in the system PATH if `convert=True`.
    - Either `file_names` or `date` must be specified.
    - Files are matched by substring (partial matching supported).
    - The device API must support `DELETE` requests to `/api/ssd/delete/<filename>`.
    """
    if convert and not shutil.which("mokucli"):
        print("❌ `mokucli` not found. Please install it from:")
        print("   https://liquidinstruments.com/software/utilities/")
        return

    files = get_file_list(ip)

    if file_names:
        if isinstance(file_names, str):
            file_names = [file_names]
        files_to_download = [
            f for f in files
            if any(pat in f for pat in file_names)
        ]
    elif date:
        pattern = re.compile(rf"{date}")
        files_to_download = [f for f in files if pattern.search(f)]
    else:
        raise ValueError("You must provide either `file_names` or `date`.")

    if not files_to_download:
        print("⚠️ No matching files found.")
        return

    output_path = output_path or os.getcwd()
    os.makedirs(output_path, exist_ok=True)

    for filename in files_to_download:
        url = f"http://{ip}/api/ssd/download/{filename}"
        lifile = filename
        csvfile = lifile.replace(".li", ".csv")
        archive_name = f"{csvfile}.zip"

        print(f"⬇️  Downloading {lifile}...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(lifile, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        if convert:
            print(f"🔄 Converting {lifile} to CSV...")
            subprocess.run(["mokucli", "convert", lifile, "--format=csv"], check=True)

            if archive:
                archive_path = os.path.join(output_path, archive_name)
                print(f"📦 Archiving {csvfile} to {archive_path}...")
                with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    zipf.write(csvfile, arcname=os.path.basename(csvfile))
                os.remove(csvfile)
            else:
                shutil.move(csvfile, os.path.join(output_path, csvfile))

        if convert:
            os.remove(lifile)
        else:
            shutil.move(lifile, os.path.join(output_path, lifile))

        if delete:
            print(f"🗑️  Deleting {filename} from device...")
            del_url = f"http://{ip}/api/ssd/delete/{filename}"
            response = requests.delete(del_url)
            if response.status_code == 200:
                print(f"✅ Deleted: {filename}")
            else:
                print(f"⚠️  Failed to delete: {filename} (status {response.status_code})")

        print(f"✅ Finished processing: {filename}")

def delete_files(ip, file_names=None, delete_all=False):
    """
    Delete files from a Moku device, optionally by partial match, full list, or all files.

    Args:
        ip (str): 
            IP address of the device (e.g., '10.128.100.198').
        file_names (str or list of str, optional): 
            Partial filename string or list of substrings to match files. Ignored if `delete_all` is True.
        delete_all (bool): 
            If True, delete all files on the device. Overrides `file_names`.

    Returns:
        None
    """
    files = get_file_list(ip)

    if delete_all:
        files_to_delete = files
    elif file_names:
        if isinstance(file_names, str):
            file_names = [file_names]
        files_to_delete = [f for f in files if any(pat in f for pat in file_names)]
    else:
        raise ValueError("Must specify `file_names` or set `delete_all=True`.")

    if not files_to_delete:
        print("⚠️ No matching files found for deletion.")
        return

    print("📋 The following files will be deleted:")
    for f in files_to_delete:
        print(f" - {f}")

    button_yes = widgets.Button(description="Yes, delete", button_style='danger')
    button_no = widgets.Button(description="No, cancel", button_style='success')
    output = widgets.Output()

    def delete_action(b):
        with output:
            clear_output()
            print("🚨 Deleting files...")
            for f in files_to_delete:
                del_url = f"http://{ip}/api/ssd/delete/{f}"
                response = requests.delete(del_url)
                if response.status_code == 200:
                    print(f"✅ Deleted: {f}")
                else:
                    print(f"⚠️ Failed to delete: {f} (status {response.status_code})")

    def cancel_action(b):
        with output:
            clear_output()
            print("❎ Deletion cancelled.")

    button_yes.on_click(delete_action)
    button_no.on_click(cancel_action)

    display(widgets.HBox([button_no, button_yes]))
    display(output)

def upload_files(ip, files):
    """
    Upload one or more files to the Moku device's SSD.

    Args:
        ip (str):
            IP address of the device (e.g., '10.128.100.198').
        files (str or list of str):
            Path to a local file or list of local file paths to upload.

    Returns:
        None

    Notes:
    ------
    - Uses HTTP POST with the file content as the body.
    - The upload endpoint is `/api/ssd/upload/<filename>`.
    - If the filename already exists on the device, it will be overwritten.
    """
    if isinstance(files, str):
        files = [files]

    for file_path in files:
        if not os.path.isfile(file_path):
            print(f"❌ File not found: {file_path}")
            continue

        filename = os.path.basename(file_path)
        url = f"http://{ip}/api/ssd/upload/{filename}"

        print(f"📤 Uploading {filename} to {ip}...")
        with open(file_path, 'rb') as f:
            response = requests.post(url, data=f)

        if response.status_code == 200:
            print(f"✅ Uploaded: {filename}")
        else:
            print(f"⚠️ Failed to upload {filename} (status {response.status_code})")

def read_lines(filename, num_lines):
    """
    Read from file, return a number of lines as list.

    Args:
        filename (string): location of the file
        num_lines (int): number of lines to read from the file

    Returns:
        lines (list): list of strings with `num_lines` lines from file
    """
    lines = []
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            for _ in range(num_lines):
                line = file.readline()
                if not line:
                    break
                lines.append(line.strip())
        return lines
    except Exception as e:
        print(f"read_lines error: {e}")
        sys.exit(1)

def is_mat_file(file_path):
    """Check if a file is a MATLAB .mat file by attempting to read its contents."""
    try:
        scipy.io.whosmat(file_path)  # Try reading variable names in the file
        return True
    except:
        return False

def moku_mat_to_csv(mat_file, out_file=None):
    """
    Convert a MATLAB `.mat` file generated by a Moku:Pro phasemeter into a CSV file.

    Args:
        mat_file (str): Path to the input `.mat` file containing the Moku data.
        out_file (str, optional): Path to the output CSV file. If not provided, 
        the function will save the CSV file with the same name as `mat_file` 
        but with a `.csv` extension.

    Returns:
        None
        The function writes the extracted data to a CSV file and does not return a value.

    Notes:
    ------
    - The function expects a specific structure within the MATLAB file: 
      `mat_data['moku'][0][0][0][0]` for the header and `mat_data['moku'][0][0][1]` 
      for the numerical data.
    - The extracted header is assumed to be a string and is stripped of its last newline.
    - The data is saved with six decimal places of precision.

    """
    mat_data = scipy.io.loadmat(mat_file)

    header = str(mat_data['moku'][0][0][0][0][:-2])

    data_array = mat_data['moku'][0][0][1]

    if out_file is None:
        out_file = mat_file + '.csv'

    with open(out_file, 'w', newline='') as f:
        np.savetxt(f, data_array, delimiter=', ', header=header, comments="", fmt="%.14f")

    return out_file

def parse_csv_file(filename, delimiter=None, logger=None):
    """
    Parse a CSV file. It is potentially packaged in ZIP, TAR, GZ, or 7z format.

    Args:
        filename (str): Location of the file

    Returns:
        num_cols (int): Number of columns in the data
        num_rows (int): Total number of rows (including headers)
        num_header_rows (int): Number of detected header lines
        header (list): List of header lines
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
                with seven_zip_ref.open(first_file_name) as f:
                    return process_stream(f)
        else:
            with open(path, 'r', encoding='utf-8') as f:
                return process_stream(f)

    logger.debug(f"Reading from file: {filename}")
    num_cols, num_rows, num_header_rows, header = process_file(filename)

    if num_header_rows == 0:
        raise ValueError("No header lines detected. Ensure the file format is correct.")

    logger.debug(f"File contains {num_rows} total rows, {num_header_rows} header rows, and {num_cols} columns")
    return num_cols, num_rows, num_header_rows, header

def get_columns_with_nans(df):
    """
    Find columns with NaNs in a DataFrame.

    Args: 
        df (DataFrame): the DataFrame

    Returns:
        columns_with_nans (dict): dictionary of columns with NaNs
    """
    columns_with_nans = {}
    for column in df.columns:
        if df[column].isna().any():
            # Get the column number
            column_number = df.columns.get_loc(column)
            columns_with_nans[column] = column_number
    return columns_with_nans

def display_menu(files):
    """
    Display a menu of options for the user.
    
    Args:
        files (list): list of files the user can choose
    """
    print("\nChoose two CSV files for processing (Master device first):")
    for idx, file in enumerate(files):
        print(f"{idx + 1}. {file}")
    print("Q. Quit")

def get_two_file_choice(files):
    """
    Get the user's choice of files.
    
    Args:
        files (list): list of potential choices

    Returns:
        ('Q', None) if user chooses to quit
        ('F', file_choices) if user chose two files
    """
    while True:
        choice = input("E.g., enter 1 2 if file #1 is Master and file #2 is Slave (Q to quit): ").strip().upper()
        if choice == 'Q':
            return 'Q', None
        
        try:
            choices = [int(ch) for ch in choice.split()]
            if len(choices) == 2 and all(1 <= ch <= len(files) for ch in choices):
                return 'F', [files[ch - 1] for ch in choices]
        except ValueError:
            pass

def get_single_file_choice(files):
    """
    Prompt user to select a single file from a list.

    Args:
        files (list): list of file name strings.

    Returns:
        str: filename selected by the user, or None if they quit.
    """
    while True:
        choice = input("Enter the number of the file to download (Q to quit): ").strip().upper()
        if choice == 'Q':
            return None
        try:
            idx = int(choice)
            if 1 <= idx <= len(files):
                return files[idx - 1]
        except ValueError:
            pass

def select_file_widget(files):
    """
    Display a dropdown widget for selecting a file and return the widget.
    The user is expected to read `.value` after selection.
    """
    dropdown = widgets.Dropdown(
        options=files,
        description='Select file:',
        layout=widgets.Layout(width='100%'),
        style={'description_width': 'initial'}
    )
    display(dropdown)
    return dropdown