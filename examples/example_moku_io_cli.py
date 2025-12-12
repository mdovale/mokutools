#!/usr/bin/env python3
"""
Example script demonstrating the moku_io CLI functionality.

This script shows how to use the interactive CLI functions for:
- Listing files on a Moku device
- Downloading and converting files
- Uploading files
- Deleting files

Usage:
    python example_moku_io_cli.py

Note: Replace the IP address with your actual Moku device IP address.
"""

from mokutools.moku_io.cli import (
    download_cli,
    upload_cli,
    delete_cli,
    print_menu,
    pick_file,
    pick_two_files,
)
from mokutools.moku_io.core import list_files


def example_list_files(ip: str):
    """Example: List all files on the device."""
    print("\n" + "="*60)
    print("Example 1: Listing files on device")
    print("="*60)
    
    try:
        files = list_files(ip)
        if files:
            print(f"\nFound {len(files)} file(s) on device {ip}:")
            for idx, filename in enumerate(files, 1):
                print(f"  {idx}. {filename}")
        else:
            print(f"\nNo files found on device {ip}")
    except Exception as e:
        print(f"❌ Error listing files: {e}")


def example_list_files_with_filter(ip: str):
    """Example: List files matching a filter."""
    print("\n" + "="*60)
    print("Example 2: Listing files with filter")
    print("="*60)
    
    try:
        # Filter files containing "20250101" (date pattern)
        files = list_files(ip, filters=["20250101"])
        if files:
            print(f"\nFound {len(files)} file(s) matching filter '20250101':")
            for filename in files:
                print(f"  - {filename}")
        else:
            print("\nNo files found matching the filter")
    except Exception as e:
        print(f"❌ Error listing files: {e}")


def example_interactive_download_by_pattern(ip: str):
    """Example: Download files matching a pattern."""
    print("\n" + "="*60)
    print("Example 3: Interactive download by pattern")
    print("="*60)
    
    # Download files matching a pattern
    # This will convert .li to .csv and optionally archive
    download_cli(
        ip=ip,
        file_names=["example"],  # Match files containing "example"
        convert=True,             # Convert .li to .csv
        archive=True,             # Create .zip archive
        output_path="./downloads",  # Save to downloads directory
        remove_from_server=False,   # Keep files on device
    )


def example_interactive_download_by_date(ip: str):
    """Example: Download files by date."""
    print("\n" + "="*60)
    print("Example 4: Interactive download by date")
    print("="*60)
    
    # Download files from a specific date (YYYYMMDD format)
    download_cli(
        ip=ip,
        date="20250101",          # Date filter
        convert=True,
        archive=True,
        output_path="./downloads",
        remove_from_server=False,
    )


def example_interactive_upload(ip: str):
    """Example: Upload files to device."""
    print("\n" + "="*60)
    print("Example 5: Interactive file upload")
    print("="*60)
    
    # Upload one or more files
    files_to_upload = [
        "test_file.csv",
        # Add more file paths as needed
    ]
    
    upload_cli(ip=ip, files=files_to_upload)


def example_interactive_delete_by_pattern(ip: str):
    """Example: Delete files matching a pattern (with confirmation)."""
    print("\n" + "="*60)
    print("Example 6: Interactive delete by pattern")
    print("="*60)
    
    # Delete files matching a pattern (will prompt for confirmation)
    delete_cli(
        ip=ip,
        file_names=["temp"],  # Match files containing "temp"
        delete_all=False,
    )


def example_interactive_delete_all(ip: str):
    """Example: Delete all files (with confirmation)."""
    print("\n" + "="*60)
    print("Example 7: Interactive delete all files")
    print("="*60)
    
    # Delete all files (will prompt for confirmation)
    # WARNING: This will delete all files on the device!
    delete_cli(
        ip=ip,
        delete_all=True,
    )


def example_interactive_menu_single_choice(ip: str):
    """Example: Interactive menu for single file selection."""
    print("\n" + "="*60)
    print("Example 8: Interactive menu - single file choice")
    print("="*60)
    
    try:
        files = list_files(ip)
        if not files:
            print("No files available on device")
            return
        
        print("\nAvailable files:")
        for idx, file in enumerate(files, 1):
            print(f"{idx}. {file}")
        
        selected = pick_file(files)
        if selected:
            print(f"\n✅ Selected file: {selected}")
        else:
            print("\n❎ Selection cancelled")
    except Exception as e:
        print(f"❌ Error: {e}")


def example_interactive_menu_two_choice(ip: str):
    """Example: Interactive menu for two file selection."""
    print("\n" + "="*60)
    print("Example 9: Interactive menu - two file choice")
    print("="*60)
    
    try:
        files = list_files(ip)
        if not files:
            print("No files available on device")
            return
        
        print_menu(files)
        choice, file_list = pick_two_files(files)
        
        if choice == 'F' and file_list:
            print(f"\n✅ Selected files:")
            print(f"  Master: {file_list[0]}")
            print(f"  Slave: {file_list[1]}")
        else:
            print("\n❎ Selection cancelled")
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """Main function to run examples."""
    # Replace with your actual Moku device IP address
    DEVICE_IP = "10.128.100.188"
    
    print("\n" + "="*60)
    print("Moku I/O CLI Examples")
    print("="*60)
    print(f"\nUsing device IP: {DEVICE_IP}")
    print("\nNote: Some examples require actual files on the device.")
    print("Modify DEVICE_IP and uncomment the examples you want to run.\n")
    
    # Uncomment the examples you want to run:
    
    # Basic operations
    example_list_files(DEVICE_IP)
    example_list_files_with_filter(DEVICE_IP)
    
    # Download operations (uncomment to test)
    example_interactive_download_by_pattern(DEVICE_IP)
    example_interactive_download_by_date(DEVICE_IP)
    
    # Upload operations (uncomment to test)
    example_interactive_upload(DEVICE_IP)
    
    # Delete operations (uncomment with caution!)
    # example_interactive_delete_by_pattern(DEVICE_IP)
    # example_interactive_delete_all(DEVICE_IP)  # WARNING: Deletes all files!
    
    # Interactive menu examples
    example_interactive_menu_single_choice(DEVICE_IP)
    example_interactive_menu_two_choice(DEVICE_IP)
    
    print("\n" + "="*60)
    print("Examples completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
