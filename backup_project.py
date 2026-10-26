"""
LoomFlow - Sports Project Backup Utility
Creates a clean, robust zip backup of the project on a local drive.
Includes all source code, configs, .env, workflows, autosaves, and precious generated data states
(sports/ datasets, intermediate states, NBA historical archives, etc.), while excluding
heavy re-installable dependencies (node_modules, .venv) and synthetic benchmark dummies.
"""

import os
import sys
import zipfile
import datetime

# Directories to exclude anywhere in the tree
EXCLUDE_DIRS = {
    'node_modules',
    '.venv',
    'venv',
    '__pycache__',
    '.git',
    '.pytest_cache',
    '.vibe_cache',
}

# Specific files to exclude (e.g. huge synthetic benchmark files)
EXCLUDE_FILES = {
    'dummy_10m.csv',
    'dummy_10m.parquet',
}

# File extensions to exclude
EXCLUDE_EXTS = {
    '.pyc',
    '.tmp',
}


def create_backup(source_dir: str = '.', dest_dir: str = r'D:\Project') -> str:
    source_dir = os.path.abspath(source_dir)
    os.makedirs(dest_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"LoomFlow-Sports-Backup-{timestamp}.zip"
    zip_path = os.path.join(dest_dir, zip_filename)

    base_folder_in_zip = "LoomFlow - Sports"

    print(f"==================================================")
    print(f"Creating project backup: {zip_path}")
    print(f"Source: {source_dir}")
    print(f"Destination: {zip_path}")
    print(f"==================================================")

    total_files = 0
    total_uncompressed_bytes = 0

    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for root, dirs, files in os.walk(source_dir):
            # Prune excluded directories in-place so os.walk doesn't descend into them
            dirs[:] = [
                d for d in dirs
                if d not in EXCLUDE_DIRS and not d.endswith('.egg-info')
            ]

            for file in files:
                if file in EXCLUDE_FILES:
                    continue

                _, ext = os.path.splitext(file)
                if ext.lower() in EXCLUDE_EXTS:
                    continue

                # Exclude root zip files or temporary logs if any
                if ext.lower() == '.zip':
                    continue

                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, source_dir)
                archive_name = os.path.join(base_folder_in_zip, rel_path)

                try:
                    file_size = os.path.getsize(full_path)
                    zf.write(full_path, arcname=archive_name)
                    total_files += 1
                    total_uncompressed_bytes += file_size
                except Exception as e:
                    print(f"Warning: could not add {rel_path}: {e}")

    # Integrity verification
    print("\nVerifying zip integrity...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        bad_file = zf.testzip()
        if bad_file:
            raise RuntimeError(f"Integrity check failed on: {bad_file}")

    compressed_size = os.path.getsize(zip_path)

    print("\n==================================================")
    print("BACKUP COMPLETED SUCCESSFULLY!")
    print(f"Saved to:             {zip_path}")
    print(f"Total files archived: {total_files}")
    print(f"Original uncompressed size: {total_uncompressed_bytes / (1024 * 1024):.2f} MB")
    print(f"Final zip file size:        {compressed_size / (1024 * 1024):.2f} MB")
    print(f"Compression ratio:          {(1 - compressed_size / total_uncompressed_bytes) * 100:.1f}% reduction")
    print("==================================================")

    return zip_path


if __name__ == "__main__":
    target_dir = sys.argv[1] if len(sys.argv) > 1 else r"D:\Project"
    create_backup(source_dir='.', dest_dir=target_dir)
