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
Notebook UI widgets for Moku I/O operations.

This module provides Jupyter notebook widgets for interactive file operations
using ipywidgets.
"""

from typing import List, Optional, Union
import ipywidgets as widgets
from IPython.display import display, clear_output
from mokutools.moku_io.core import (
    list_files,
    download,
    upload,
    delete,
)


def select_file_widget(files: List[str]) -> widgets.Dropdown:
    """
    Display a dropdown widget for selecting a file and return the widget.
    The user is expected to read `.value` after selection.

    Parameters
    ----------
    files : list of str
        List of file names to choose from.

    Returns
    -------
    ipywidgets.Dropdown
        Dropdown widget for file selection.
    """
    dropdown = widgets.Dropdown(
        options=files,
        description='Select file:',
        layout=widgets.Layout(width='100%'),
        style={'description_width': 'initial'}
    )
    display(dropdown)
    return dropdown


def download_files_widget(
    ip: str,
    file_names: Optional[Union[str, List[str]]] = None,
    date: Optional[str] = None,
    convert: bool = True,
    archive: bool = True,
    output_path: Optional[str] = None,
    remove_from_server: bool = False,
) -> widgets.Output:
    """
    Interactive widget wrapper for download with progress display.

    Parameters
    ----------
    ip : str
        IP address of the device.
    file_names : str or list of str, optional
        Partial filename or list of partial strings to match files.
    date : str, optional
        A date string in 'YYYYMMDD' format to filter filenames.
    convert : bool, default True
        If True, convert the `.li` file to `.csv` using `mokucli`.
    archive : bool, default True
        If True, zip the `.csv` file. Applies only if `convert=True`.
    output_path : str, optional
        Directory where output files will be saved.
    remove_from_server : bool, default False
        If True, delete the `.li` file from the device after processing.

    Returns
    -------
    ipywidgets.Output
        Output widget displaying download progress and results.
    """
    output = widgets.Output()
    
    def download_action():
        with output:
            clear_output()
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
                print(f"❌ Error: {e}")
            except Exception as e:
                print(f"❌ Error processing files: {e}")
    
    download_action()
    return output


def upload_files_widget(ip: str, files: Union[str, List[str]]) -> widgets.Output:
    """
    Interactive widget wrapper for upload with progress display.

    Parameters
    ----------
    ip : str
        IP address of the device.
    files : str or list of str
        Path to a local file or list of local file paths to upload.

    Returns
    -------
    ipywidgets.Output
        Output widget displaying upload progress and results.
    """
    output = widgets.Output()
    
    def upload_action():
        with output:
            clear_output()
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
    
    upload_action()
    return output


def delete_files_widget(
    ip: str,
    file_names: Optional[Union[str, List[str]]] = None,
    delete_all: bool = False,
) -> None:
    """
    Interactive widget wrapper for delete with confirmation buttons.

    Parameters
    ----------
    ip : str
        IP address of the device.
    file_names : str or list of str, optional
        Partial filename string or list of substrings to match files.
    delete_all : bool, default False
        If True, delete all files on the device.
    """
    try:
        # First, get the list of files that would be deleted
        files = list_files(ip)
        
        if delete_all:
            files_to_delete = files
        elif file_names:
            if isinstance(file_names, str):
                file_names = [file_names]
            files_to_delete = [f for f in files if any(pat in f for pat in file_names)]
        else:
            print("❌ Error: Must specify `file_names` or set `delete_all=True`.")
            return

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
                try:
                    deleted = delete(ip, patterns=file_names, delete_all=delete_all, confirm=True)
                    for f in deleted:
                        print(f"✅ Deleted: {f}")
                except Exception as e:
                    print(f"❌ Error: {e}")

        def cancel_action(b):
            with output:
                clear_output()
                print("❎ Deletion cancelled.")

        button_yes.on_click(delete_action)
        button_no.on_click(cancel_action)

        display(widgets.HBox([button_no, button_yes]))
        display(output)
    except Exception as e:
        print(f"❌ Error: {e}")
