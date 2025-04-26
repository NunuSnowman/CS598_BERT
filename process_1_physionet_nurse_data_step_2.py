import json
import re
import os
# Import the verification utility function and the pattern from the separate file
from process_verification_utils import verify_single_record_mapping, MASK_TOKEN_PATTERN

def process_and_partially_reidentify(text_file_path: str, res_file_path: str, mapping_file_path: str, output_dir: str):
    """
    Verifies physionet formatted data line by line and processes verified records.
    For verified records, partially re-identifies masked data in id.res based on mapping.jsonl,
    but keeps masks containing 'Name' or 'name'.
    Saves original, partially re-identified, original masked, and mapping data to the specified output directory.

    Args:
        text_file_path: Path to the original text file (id.text).
        res_file_path: Path to the masked text file (id.res).
        mapping_file_path: Path to the mapping file (mapping.jsonl).
        output_dir: Directory to save verified and partially re-identified data.
    """
    # Check if input files exist
    if not all(os.path.exists(f) for f in [text_file_path, res_file_path, mapping_file_path]):
        print("Error: One or more input files not found.")
        if not os.path.exists(text_file_path): print(f" - {text_file_path} not found")
        if not os.path.exists(res_file_path): print(f" - {res_file_path} not found")
        if not os.path.exists(mapping_file_path): print(f" - {mapping_file_path} not found")
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    output_text_path = os.path.join(output_dir, 'original_data.txt')
    output_res_path = os.path.join(output_dir, 'masked_data.txt') # This will store partially re-identified data
    output_mapping_path = os.path.join(output_dir, 'mapping.jsonl')
    output_original_res_path = os.path.join(output_dir, 'raw_masked.text') # Store original res for verified lines

    print(f"Processing files:\n - {text_file_path}\n - {res_file_path}\n - {mapping_file_path}\n")
    print(f"Saving verified and partially re-identified data to: {output_dir}\n")

    total_lines = 0
    verified_records_count = 0

    try:
        with open(text_file_path, 'r', encoding='utf-8') as text_file, \
                open(res_file_path, 'r', encoding='utf-8') as res_file, \
                open(mapping_file_path, 'r', encoding='utf-8') as mapping_file, \
                open(output_text_path, 'w', encoding='utf-8') as out_text_file, \
                open(output_res_path, 'w', encoding='utf-8') as out_res_file, \
                open(output_mapping_path, 'w', encoding='utf-8') as out_mapping_file, \
                open(output_original_res_path, 'w', encoding='utf-8') as out_original_res_file: # New output file

            # Iterate through input files line by line simultaneously
            for line_num, (original_line, masked_line, mapping_line_str) in enumerate(zip(text_file, res_file, mapping_file), 1):

                total_lines += 1

                # Strip leading/trailing whitespace, especially newlines
                original_line_stripped = original_line.strip()
                masked_line_stripped = masked_line.strip()
                mapping_line_str_stripped = mapping_line_str.strip()

                mapping_dict = {}
                is_mapping_valid = True # Flag to track mapping validity for this line

                # Attempt to parse mapping JSON first
                if mapping_line_str_stripped:
                    try:
                        mapping_dict = json.loads(mapping_line_str_stripped)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Error parsing JSON on line {line_num} in {mapping_file_path}. Skipping record. Error: {e}")
                        is_mapping_valid = False # Mark as invalid
                elif masked_line_stripped and MASK_TOKEN_PATTERN.search(masked_line_stripped):
                    # If masked line has tokens but mapping is empty/missing, it's invalid
                    print(f"Warning: Line {line_num}: Masked tokens found in {res_file_path} but mapping is empty. Skipping record.")
                    is_mapping_valid = False


                # --- Call the external single record verification function ---
                # Only proceed with detailed check if JSON parsed and mapping wasn't unexpectedly empty
                if is_mapping_valid:
                    # Note: verify_single_record_mapping expects stripped strings and a dict
                    if not verify_single_record_mapping(original_line_stripped, masked_line_stripped, mapping_dict):
                        print(f"Warning: Line {line_num}: Verification failed (mapping mismatch or missing/extra tokens). Skipping record.")
                        is_mapping_valid = False # Mark as invalid


                # --- Process Valid Records ---
                if is_mapping_valid:
                    verified_records_count += 1
                    partially_reidentified_line = masked_line_stripped

                    # --- Partial Re-identification Step ---
                    # Iterate through the mapping for this line
                    for mask_token, original_value in mapping_dict.items():
                        # Check if the mask token (case-insensitive) contains "name"
                        if "name" not in mask_token.lower():
                            # If it doesn't contain "name", replace it with the original value
                            partially_reidentified_line = partially_reidentified_line.replace(mask_token, str(original_value))

                    # Write the original, partially re-identified, original masked, and mapping lines to output files
                    out_text_file.write(original_line_stripped + '\n')
                    out_res_file.write(partially_reidentified_line + '\n') # Partially re-identified
                    out_original_res_file.write(masked_line_stripped + '\n') # Original masked
                    out_mapping_file.write(mapping_line_str_stripped + '\n') # Original mapping

    except Exception as e:
        print(f"\nAn unexpected error occurred during processing: {e}")

    finally:
        print(f"\nFinished processing {total_lines} lines.")
        print(f"Successfully verified and saved {verified_records_count} records to '{output_dir}'.")
        print(f"Skipped {total_lines - verified_records_count} records due to verification failures.")


# --- Function to verify the data in the output directory ---
def verify_verified_data(verified_dir: str):
    """
    Verifies the consistency between id.text, id.res.original_masked, and mapping.jsonl
    in the verified directory using the single record verification function.
    Note: This function does not verify the correctness of the id.res (partially re-identified) file itself.

    Args:
        verified_dir: Path to the directory containing the files to verify.
    """
    text_file_path = os.path.join(verified_dir, 'id.text')
    original_res_file_path = os.path.join(verified_dir, 'id.res.original_masked')
    mapping_file_path = os.path.join(verified_dir, 'mapping.jsonl')
    # We don't need id.res (partially re-identified) for THIS SPECIFIC verification check

    # Check if required files exist in the verified directory
    if not all(os.path.exists(f) for f in [text_file_path, original_res_file_path, mapping_file_path]):
        print(f"Error: Required files not found in '{verified_dir}' for verification.")
        if not os.path.exists(text_file_path): print(f" - {text_file_path} not found")
        if not os.path.exists(original_res_file_path): print(f" - {original_res_file_path} not found")
        if not os.path.exists(mapping_file_path): print(f" - {mapping_file_path} not found")
        return

    print(f"\nVerifying data in '{verified_dir}'...")
    print(f"(Checking id.text, id.res.original_masked, and mapping.jsonl consistency)\n")

    total_lines = 0
    errors_found = False

    try:
        with open(text_file_path, 'r', encoding='utf-8') as text_file, \
                open(original_res_file_path, 'r', encoding='utf-8') as original_res_file, \
                open(mapping_file_path, 'r', encoding='utf-8') as mapping_file:

            # Iterate through files line by line simultaneously
            for line_num, (original_line, original_masked_line, mapping_line_str) in enumerate(zip(text_file, original_res_file, mapping_file), 1):

                total_lines += 1

                # Strip leading/trailing whitespace
                original_line_stripped = original_line.strip()
                original_masked_line_stripped = original_masked_line.strip()
                mapping_line_str_stripped = mapping_line_str.strip()

                mapping_dict = {}
                # Attempt to parse mapping JSON
                if mapping_line_str_stripped:
                    try:
                        mapping_dict = json.loads(mapping_line_str_stripped)
                    except json.JSONDecodeError as e:
                        print(f"Error parsing JSON on line {line_num} in {mapping_file_path}: {e}")
                        errors_found = True
                        continue # Skip verification for this line
                elif original_masked_line_stripped and MASK_TOKEN_PATTERN.search(original_masked_line_stripped):
                    # If masked line has tokens but mapping is empty/missing, it's an error
                    print(f"Error on line {line_num}: Masked tokens found in {original_res_file_path} but mapping is empty.")
                    errors_found = True
                    continue

                # --- Call the single record verification function ---
                # We use the original text, original masked text, and the mapping dict
                if not verify_single_record_mapping(original_line_stripped, original_masked_line_stripped, mapping_dict):
                    print(f"Error on line {line_num}: Mapping consistency check failed.")
                    # Optional: Print details if needed for debugging
                    # print(f"  Original : {original_line_stripped}")
                    # print(f"  Masked   : {original_masked_line_stripped}")
                    # print(f"  Mapping  : {mapping_dict}")
                    errors_found = True


    except Exception as e:
        print(f"\nAn unexpected error occurred during verification: {e}")
        errors_found = True

    finally:
        if not errors_found:
            print(f"\nVerification of data in '{verified_dir}' successful! All {total_lines} records checked.")
        else:
            print(f"\nVerification of data in '{verified_dir}' finished with errors on {total_lines} records checked.")


# --- Main execution block ---
if __name__ == "__main__":
    # Define the paths to your input files
    input_text_file = 'tmp/physionet_nurse_filtered/id.text'
    input_res_file = 'tmp/physionet_nurse_filtered/id.res'
    input_mapping_file = 'tmp/physionet_nurse_filtered/mapping.jsonl'

    # Define the output directory
    output_directory = 'data/physionet_nurse'

    print("Starting data processing and partial re-identification...")
    process_and_partially_reidentify(input_text_file, input_res_file, input_mapping_file, output_directory)

    print("\nProcessing complete. Starting verification of output files...")
    # Run the verification on the newly created files in the output directory
    verify_verified_data(output_directory)