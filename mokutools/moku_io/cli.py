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
Command-line interface helpers for Moku I/O operations.

This module provides functions for interactive command-line use, including
menus, prompts, and formatted output. It also provides a Typer-based CLI
accessible via the `moku` command.
"""

import warnings
from typing import List, Optional, Tuple
import typer
from typer import Option, Argument

from mokutools.moku_io.core import (
    list_files,
    download,
    upload,
    delete,
)

# Create Typer app
app = typer.Typer(
    name="moku",
    help="Moku I/O command-line interface for file operations on Moku devices.",
    no_args_is_help=True,
)


# Internal helper functions
def print_menu(files: List[str]) -> None:
    """
    Display a menu of options for the user.
    
    Parameters
    ----------
    files : list of str
        List of files the user can choose from.
    """
    print("\nChoose two CSV files for processing (Master device first):")
    for idx, file in enumerate(files):
        print(f"{idx + 1}. {file}")
    print("Q. Quit")


def pick_two_files(files: List[str]) -> Tuple[str, Optional[List[str]]]:
    """
    Get the user's choice of files.
    
    Parameters
    ----------
    files : list of str
        List of potential choices

    Returns
    -------
    tuple
        ('Q', None) if user chooses to quit
        ('F', file_choices) if user chose two files
    """
    while True:
        choice = input(
            "E.g., enter 1 2 if file #1 is Master and file #2 is Slave (Q to quit): "
        ).strip().upper()
        if choice == 'Q':
            return 'Q', None
        
        try:
            choices = [int(ch) for ch in choice.split()]
            if len(choices) == 2 and all(1 <= ch <= len(files) for ch in choices):
                return 'F', [files[ch - 1] for ch in choices]
        except ValueError:
            pass


def pick_file(files: List[str]) -> Optional[str]:
    """
    Prompt user to select a single file from a list.

    Parameters
    ----------
    files : list of str
        List of file name strings.

    Returns
    -------
    str or None
        Filename selected by the user, or None if they quit.
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


# Renamed helper functions (shorter names)
def dl(
    ip: str,
    file_names: Optional[List[str]] = None,
    date: Optional[str] = None,
    convert: bool = True,
    archive: bool = True,
    output_path: Optional[str] = None,
    remove_from_server: bool = False,
) -> None:
    """
    Interactive wrapper for download with user-friendly output.

    Parameters
    ----------
    ip : str
        IP address of the device.
    file_names : list of str, optional
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
    """
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


def up(ip: str, files: List[str]) -> None:
    """
    Interactive wrapper for upload with user-friendly output.

    Parameters
    ----------
    ip : str
        IP address of the device.
    files : list of str
        Path to a local file or list of local file paths to upload.
    """
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


def rm(
    ip: str,
    file_names: Optional[List[str]] = None,
    delete_all: bool = False,
) -> None:
    """
    Interactive wrapper for delete with confirmation prompt.

    Parameters
    ----------
    ip : str
        IP address of the device.
    file_names : list of str, optional
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

        confirm = input("Are you sure you want to delete these files? (yes/no): ").strip().lower()
        if confirm in ['yes', 'y']:
            deleted = delete(ip, patterns=file_names, delete_all=delete_all, confirm=True)
            for f in deleted:
                print(f"✅ Deleted: {f}")
        else:
            print("❎ Deletion cancelled.")
    except Exception as e:
        print(f"❌ Error: {e}")


# Backward compatibility aliases with deprecation warnings
def download_cli(
    ip: str,
    file_names: Optional[List[str]] = None,
    date: Optional[str] = None,
    convert: bool = True,
    archive: bool = True,
    output_path: Optional[str] = None,
    remove_from_server: bool = False,
) -> None:
    """
    Deprecated: Use `dl` instead.
    
    Interactive wrapper for download with user-friendly output.
    """
    warnings.warn(
        "download_cli is deprecated. Use 'dl' instead.",
        DeprecationWarning,
        stacklevel=2
    )
    dl(
        ip=ip,
        file_names=file_names,
        date=date,
        convert=convert,
        archive=archive,
        output_path=output_path,
        remove_from_server=remove_from_server,
    )


def upload_cli(ip: str, files: List[str]) -> None:
    """
    Deprecated: Use `up` instead.
    
    Interactive wrapper for upload with user-friendly output.
    """
    warnings.warn(
        "upload_cli is deprecated. Use 'up' instead.",
        DeprecationWarning,
        stacklevel=2
    )
    up(ip=ip, files=files)


def delete_cli(
    ip: str,
    file_names: Optional[List[str]] = None,
    delete_all: bool = False,
) -> None:
    """
    Deprecated: Use `rm` instead.
    
    Interactive wrapper for delete with confirmation prompt.
    """
    warnings.warn(
        "delete_cli is deprecated. Use 'rm' instead.",
        DeprecationWarning,
        stacklevel=2
    )
    rm(
        ip=ip,
        file_names=file_names,
        delete_all=delete_all,
    )


# Typer CLI commands
@app.command("ls")
def cmd_ls(
    ip: str = Argument(..., help="IP address of the Moku device"),
) -> None:
    """
    List files on the device.
    """
    try:
        files = list_files(ip)
        if files:
            print(f"\nFound {len(files)} file(s) on device {ip}:")
            for filename in files:
                print(f"  - {filename}")
        else:
            print(f"\nNo files found on device {ip}")
    except Exception as e:
        typer.echo(f"❌ Error: {e}", err=True)
        raise typer.Exit(1)


@app.command("dl")
def cmd_dl(
    ip: str = Argument(..., help="IP address of the Moku device"),
    pattern: Optional[List[str]] = Option(
        None,
        "--pattern",
        "-p",
        help="Pattern(s) to match filenames (repeatable)",
    ),
    date: Optional[str] = Option(
        None,
        "--date",
        help="Date filter in YYYYMMDD format",
    ),
    convert: bool = Option(
        True,
        "--convert/--no-convert",
        help="Convert .li files to .csv",
    ),
    archive: bool = Option(
        True,
        "--archive/--no-archive",
        help="Create .zip archive (only if convert=True)",
    ),
    output_path: Optional[str] = Option(
        None,
        "--output-path",
        "-o",
        help="Output directory for downloaded files",
    ),
    rm: bool = Option(
        False,
        "--rm",
        help="Remove files from device after download",
    ),
    interactive: bool = Option(
        False,
        "--interactive",
        help="Use interactive mode",
    ),
) -> None:
    """
    Download files from the device.
    """
    try:
        if interactive:
            # Use the interactive wrapper
            dl(
                ip=ip,
                file_names=pattern,
                date=date,
                convert=convert,
                archive=archive,
                output_path=output_path,
                remove_from_server=rm,
            )
        else:
            # Direct call to core function
            if not pattern and not date:
                typer.echo("❌ Error: Must provide --pattern or --date", err=True)
                raise typer.Exit(1)
            
            processed = download(
                ip=ip,
                patterns=pattern,
                date=date,
                convert=convert,
                archive=archive,
                output_path=output_path,
                remove_from_server=rm,
            )
            
            if not processed:
                typer.echo("⚠️ No matching files found.")
            else:
                for filename in processed:
                    typer.echo(f"✅ Finished processing: {filename}")
    except ValueError as e:
        typer.echo(f"❌ Error: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"❌ Error processing files: {e}", err=True)
        raise typer.Exit(1)


@app.command("up")
def cmd_up(
    ip: str = Argument(..., help="IP address of the Moku device"),
    files: List[str] = Argument(..., help="Local file path(s) to upload"),
    interactive: bool = Option(
        False,
        "--interactive",
        help="Use interactive mode",
    ),
) -> None:
    """
    Upload files to the device.
    """
    try:
        if interactive:
            # Use the interactive wrapper
            up(ip=ip, files=files)
        else:
            # Direct call to core function
            results = upload(ip, files)
            for filename, success in results.items():
                if success:
                    typer.echo(f"✅ Uploaded: {filename}")
                else:
                    typer.echo(f"❌ Failed to upload: {filename}", err=True)
    except FileNotFoundError as e:
        typer.echo(f"❌ File not found: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"❌ Error uploading files: {e}", err=True)
        raise typer.Exit(1)


@app.command("rm")
def cmd_rm(
    ip: str = Argument(..., help="IP address of the Moku device"),
    pattern: Optional[List[str]] = Option(
        None,
        "--pattern",
        "-p",
        help="Pattern(s) to match filenames (repeatable)",
    ),
    all: bool = Option(
        False,
        "--all",
        help="Delete all files on the device",
    ),
    yes: bool = Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation prompt",
    ),
    interactive: bool = Option(
        True,
        "--interactive/--no-interactive",
        help="Use interactive confirmation (default: True)",
    ),
) -> None:
    """
    Delete files from the device.
    """
    try:
        # Determine if we should prompt for confirmation
        should_confirm = interactive and not yes
        
        if should_confirm:
            # Use the interactive wrapper
            rm(
                ip=ip,
                file_names=pattern,
                delete_all=all,
            )
        else:
            # Direct call to core function (with confirm=True)
            if not pattern and not all:
                typer.echo("❌ Error: Must provide --pattern or --all", err=True)
                raise typer.Exit(1)
            
            deleted = delete(
                ip=ip,
                patterns=pattern,
                delete_all=all,
                confirm=True,
            )
            
            if not deleted:
                typer.echo("⚠️ No matching files found for deletion.")
            else:
                for f in deleted:
                    typer.echo(f"✅ Deleted: {f}")
    except Exception as e:
        typer.echo(f"❌ Error: {e}", err=True)
        raise typer.Exit(1)


# Allow running as a module
if __name__ == "__main__":
    app()
