import os
import json
import argparse
import sys
from process_verification_utils import verify_single_record_mapping, MASK_TOKEN_PATTERN


def verify_data_in_directory(directory_path: str) -> bool:
    """
    Verifies the data consistency (original, masked, mapping) within a single directory.

    Args:
        directory_path: Path to the directory containing the three data files.

    Returns:
        True if all records in the directory pass verification, False otherwise.
        Prints detailed errors for failing records.
    """
    original_file = os.path.join(directory_path, 'original_data.txt')
    masked_file = os.path.join(directory_path, 'masked_data.txt')
    mapping_file = os.path.join(directory_path, 'mapping.jsonl')

    # Check if all three expected files exist
    if not all(os.path.exists(f) for f in [original_file, masked_file, mapping_file]):
        # This check should ideally happen before calling this function,
        # but included here as a safeguard.
        print(f"Skipping directory '{directory_path}': Missing required files.")
        return False # Indicate failure due to missing files

    print(f"  Verifying files in: {directory_path}")

    errors_found_in_dir = False
    dir_error_count = 0

    try:
        with open(original_file, 'r', encoding='utf-8') as f_orig, \
                open(masked_file, 'r', encoding='utf-8') as f_masked, \
                open(mapping_file, 'r', encoding='utf-8') as f_map:

            # Iterate through files line by line simultaneously
            for line_num, (original_line, masked_line, mapping_line_str) in enumerate(zip(f_orig, f_masked, f_map), 1):

                original_line_stripped = original_line.strip()
                masked_line_stripped = masked_line.strip()
                mapping_line_str_stripped = mapping_line_str.strip()

                mapping_dict = {}
                is_line_parsable = True

                # Attempt to parse mapping JSON first
                if mapping_line_str_stripped:
                    try:
                        mapping_dict = json.loads(mapping_line_str_stripped)
                    except json.JSONDecodeError as e:
                        print(f"    Error on line {line_num} in {mapping_file}: JSON parsing failed - {e}")
                        is_line_parsable = False
                        errors_found_in_dir = True
                        dir_error_count += 1
                        # Continue to next line, skip verification

                # If mapping wasn't unexpectedly empty, check for masked tokens without mapping
                if is_line_parsable and masked_line_stripped and MASK_TOKEN_PATTERN.search(masked_line_stripped) and not mapping_dict:
                    # If masked line has tokens but mapping is empty/missing, it's an error
                    print(f"    Error on line {line_num}: Masked tokens found in {masked_file} but mapping is empty.")
                    is_line_parsable = False # Treat as not parsable/valid for verification
                    errors_found_in_dir = True
                    dir_error_count += 1


                # --- Call the single record verification function ---
                # Only call if mapping was successfully parsed (or was empty as expected)
                if is_line_parsable:
                    # Note: verify_single_record_mapping expects stripped strings and a dict
                    if not verify_single_record_mapping(original_line_stripped, masked_line_stripped, mapping_dict):
                        print(f"    Error on line {line_num}: Mapping consistency check failed.")
                        # Optional: Print details if needed for debugging
                        # print(f"      Original : {original_line_stripped}")
                        # print(f"      Masked   : {masked_line_stripped}")
                        # print(f"      Mapping  : {mapping_dict}")
                        errors_found_in_dir = True
                        dir_error_count += 1


    except FileNotFoundError:
        # Should not happen if initial check passes, but for safety
        print(f"    Error: File not found during processing in {directory_path}. Skipping.")
        errors_found_in_dir = True
        dir_error_count += 1 # Count as at least one issue
    except Exception as e:
        print(f"    An unexpected error occurred while processing {directory_path}: {e}")
        errors_found_in_dir = True
        dir_error_count += 1 # Count as at least one issue


    if errors_found_in_dir:
        print(f"  Verification FAILED for '{directory_path}' (found {dir_error_count} errors).")
        return False
    else:
        print(f"  Verification PASSED for '{directory_path}'.")
        return True


def verify_all_subdirectories(root_directory: str):
    """
    Finds all subdirectories within a root path and verifies the data files
    (original_data.txt, masked_data.txt, mapping.jsonl) within each.

    Args:
        root_directory: The starting directory to search for subdirectories.
    """
    print(f"Starting verification process in root directory: {root_directory}\n")

    total_dirs_checked = 0
    failed_dirs_count = 0
    verified_dirs_count = 0

    # Walk through the directory tree
    # dirpath: current directory path
    # dirnames: list of subdirectories in dirpath
    # filenames: list of files in dirpath
    for dirpath, dirnames, filenames in os.walk(root_directory):

        # Check if the current directory contains the three required files
        required_files = ['original_data.txt', 'masked_data.txt', 'mapping.jsonl']
        if all(f in filenames for f in required_files):
            total_dirs_checked += 1
            print(f"Found data directory: {dirpath}")

            # Verify the data within this specific directory
            is_verified = verify_data_in_directory(dirpath)

            if not is_verified:
                failed_dirs_count += 1
            else:
                verified_dirs_count += 1

            print("-" * 20) # Separator for clarity


    print("\n--- Verification Summary ---")
    print(f"Total data directories found and checked: {total_dirs_checked}")
    print(f"Directories where verification PASSED: {verified_dirs_count}")
    print(f"Directories where verification FAILED: {failed_dirs_count}")

    if total_dirs_checked == 0:
        print("No data directories containing all three required files were found.")


# --- Main execution block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Verify data consistency in subdirectories.')
    parser.add_argument('--root_dir', type=str, default="data", help='Root directory to search for data subdirectories.')

    args = parser.parse_args()

    # Ensure the root directory exists
    if not os.path.isdir(args.root_dir):
        print(f"Error: Root directory not found at {args.root_dir}")
        sys.exit(1)

    # Run the verification process
    verify_all_subdirectories(args.root_dir)