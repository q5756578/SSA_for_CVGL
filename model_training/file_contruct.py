"""
File Organization and Filtering Script for Satellite Imagery

This script provides functionality to find and copy PNG files based on specific keywords
from a source directory to a destination directory. It's particularly useful for organizing
and filtering satellite imagery files based on their coordinates or other identifiers.

The script walks through all subdirectories of the source folder, identifies PNG files
that contain any of the specified keywords in their filenames, and copies them to a
designated destination folder while preserving their original filenames.

Example Usage:
    python file_contruct.py
    # This will search for PNG files containing specific satellite coordinates
    # in the /VIGOR directory and copy them to ./filtered_pngs
"""

import os
import shutil

def find_and_copy_png(source_folder, keywords, destination_folder):
    """
    Find and copy PNG files containing specified keywords from source folder to destination folder.
    
    This function recursively searches through the source directory and its subdirectories
    for PNG files that contain any of the specified keywords in their filenames. When found,
    these files are copied to the destination folder while preserving their original names
    and metadata.
    
    Args:
        source_folder (str): Path to the source directory to search in. This should be an
                           absolute or relative path to the directory containing the PNG files.
        keywords (list): List of keywords to match in filenames. Each keyword should be a
                        string that might appear in the target PNG filenames.
        destination_folder (str): Path to the destination directory where matching files
                                will be copied. If this directory doesn't exist, it will
                                be created.
    
    Returns:
        None
    
    Note:
        - The function preserves file metadata during copying using shutil.copy2
        - Only files with .png extension (case-insensitive) are considered
        - A file is copied if it contains ANY of the specified keywords
        - Progress is printed to console for each copied file
    """
    # Create destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
        print(f"Created destination folder: {destination_folder}")
    
    # Counter for tracking number of files copied
    files_copied = 0
    
    # Walk through all directories and files in source folder
    for root, _, files in os.walk(source_folder):
        for filename in files:
            # Check if file is PNG and contains any of the keywords
            if filename.lower().endswith(".png") and any(keyword in filename for keyword in keywords):
                source_path = os.path.join(root, filename)
                destination_path = os.path.join(destination_folder, filename)
                # Copy file while preserving metadata
                shutil.copy2(source_path, destination_path)
                files_copied += 1
                print(f"Copied: {filename}")
    
    return files_copied

if __name__ == "__main__":
    # Source directory containing the PNG files
    # This should be the root directory of your VIGOR dataset
    source_folder = '/VIGOR'
    
    # List of keywords to match in filenames
    # These are satellite image coordinates in the format: satellite_latitude_longitude
    # The coordinates correspond to specific locations in Seattle and San Francisco
    keywords = [
        "satellite_47.58192628904255_-122.32581300575343",  # Seattle area coordinates
        "satellite_47.57142797055815_-122.32143500767124",  # Seattle area coordinates
        "satellite_47.5927526799796_-122.32970455960427",   # Seattle area coordinates
        'satellite_37.76552644828475_-122.43935207479669'   # San Francisco area coordinates
    ]
    
    # Destination folder for filtered PNG files
    # Will be created in the current working directory if it doesn't exist
    destination_folder = os.path.join(os.getcwd(), "filtered_pngs")
    
    # Execute the file finding and copying process
    total_copied = find_and_copy_png(source_folder, keywords, destination_folder)
    print(f"\nSummary:")
    print(f"- Total files copied: {total_copied}")
    print(f"- Destination folder: {destination_folder}")
    print(f"- Keywords matched: {len(keywords)}")
 


