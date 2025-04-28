import re
import os
import json
import argparse
import dataclasses # Import dataclasses for converting MaskResult to dict

MASK_TOKEN_PATTERN = re.compile(r"\[\*\*(.*?)\*\*\]")

def read_and_split_records(file_path):
    """
    Reads file content and splits it into records.

    Args:
        file_path (str): The path to the file.

    Returns:
        list: A list containing all records, with START_OF_RECORD and END_OF_RECORD markers removed.
              Returns an empty list if the file does not exist or an error occurs.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Use non-capturing group (?:...) for the split pattern
            records = re.split(r"START_OF_RECORD=\d+\|\|\|\|\d+\|\|\|\|", content)
            cleaned_records = []
            for record in records:
                if record.strip():
                    # Remove the trailing END_OF_RECORD marker
                    cleaned_record = re.sub(r"\|\|\|\|END_OF_RECORD", "", record).strip()
                    cleaned_records.append(cleaned_record)
            return cleaned_records
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []
    except Exception as e:
        print(f"An error occurred during file reading: {e}")
        return []

def process_and_filter_records(res_records, text_records):
    """
    Processes pairs of res and text records, performs mapping, and filters
    records where not all masks are successfully mapped.

    Args:
        res_records (list): List of records from the .res file.
        text_records (list): List of records from the .text file.

    Returns:
        tuple: (matched_res_records, matched_text_records, matched_mask_results)
               Lists containing records and the list of MaskResult objects that passed the filtering.
    """
    matched_res_records = []
    matched_text_records = []
    matched_mask_results = [] # This will store lists of MaskResult objects

    if len(res_records) != len(text_records):
        print("Error: Number of records in res and text files do not match. Cannot process.")
        return [], [], []

    # Assuming RecordProcessor is available globally or imported
    try:
        from record_processor import RecordProcessor
    except ImportError:
        print("Error: Could not import RecordProcessor. Please ensure record_processor.py is accessible.")
        return [], [], []


    for i in range(len(res_records)):
        res_record = res_records[i]
        text_record = text_records[i]

        # Skip empty records
        if not res_record.strip() or not text_record.strip():
            print(f"Warning: Skipping empty record {i+1}.")
            continue

        processor = RecordProcessor(res_record, text_record)
        mask_results = processor.masks # Get the list of MaskResult objects

        # Filtering Logic: A record is matched if all masks found in the res_record
        # are present in the mask_results with valid (non-None, non-empty text) mappings.
        all_masks_in_record = re.findall(MASK_TOKEN_PATTERN, res_record)
        unique_mask_contents = set(all_masks_in_record)
        total_unique_masks = len(unique_mask_contents)

        # Create a set of mask labels that were successfully mapped to non-empty text
        successfully_mapped_labels = {
            result.label for result in mask_results if result.text is not None and result.text.strip() != ""
        }

        # Check if all unique mask contents from the res_record have a corresponding
        # successfully mapped label in the mask_results.
        # Note: This assumes the label in MaskResult is the content inside the mask [**label**].
        is_matched = False
        if total_unique_masks == 0 and not mask_results:
            # Case: No masks in res_record, and mask_results is empty (correctly)
            is_matched = True
        elif total_unique_masks > 0:
            # Check if all unique mask contents from res_record are present as labels
            # in the successfully mapped results.
            is_matched = all(mask_content in successfully_mapped_labels for mask_content in unique_mask_contents)


        if is_matched:
            print(f"Record {i+1} is matched ({len(successfully_mapped_labels)}/{total_unique_masks} unique masks mapped).")
            matched_res_records.append(res_record)
            matched_text_records.append(text_record)
            matched_mask_results.append(mask_results) # Append the list of MaskResult objects
        else:
            print(f"Record {i+1} is NOT fully matched (found {total_unique_masks} unique masks, {len(successfully_mapped_labels)} successfully mapped). Skipping.")

    return matched_res_records, matched_text_records, matched_mask_results


def partially_reidentify_record(masked_record: str, mask_results: list):
    """
    Applies mapping from MaskResult objects to a masked record, except for masks containing 'Name'/'name'.

    Args:
        masked_record (str): The record from the .res file.
        mask_results (list[MaskResult]): The list of MaskResult objects for this record.

    Returns:
        str: The partially re-identified record.
    """
    reidentified_record = masked_record
    # Create a mapping from the full mask string to the original text from MaskResult objects
    mapping = {result.masked_text: result.text for result in mask_results}

    # Sort mapping keys by length in descending order to avoid partial replacement of longer masks
    sorted_mask_tokens = sorted(mapping.keys(), key=len, reverse=True)

    for mask_token in sorted_mask_tokens:
        # Extract mask content from the token format [**mask_content**]
        match = MASK_TOKEN_PATTERN.match(mask_token)
        if match:
            mask_content = match.group(1)
            original_value = mapping.get(mask_token, "") # Get original value, default to empty string if not found

            # Check if the mask content (case-insensitive) contains "name"
            if "name" not in mask_content.lower():
                # If it doesn't contain "name", replace the token in the record with the original value
                reidentified_record = reidentified_record.replace(mask_token, str(original_value))
            # If it contains "name", the mask token is kept in reidentified_record (no replacement)

    return reidentified_record

def save_processed_data(output_dir: str, text_records: list[str], partially_reidentified_records: list[str], res_records: list[str], mask_results_list: list[list]):
    """
    Saves the processed data to a JSONL file.

    Args:
        output_dir (str): The directory to save the output file.
        text_records (list[str]): List of original text records.
        partially_reidentified_records (list[str]): List of partially re-identified records.
        res_records (list[str]): List of original masked records (.res).
        mask_results_list (list[list[MaskResult]]): List of lists of MaskResult objects,
                                                     one list per record.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    output_file_path = os.path.join(output_dir, "processed_data.jsonl")

    print(f"Writing processed data to {output_file_path}")

    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            # Iterate through the matched records and their corresponding data
            for i in range(len(res_records)):
                record_data = {
                    "res_record": res_records[i],
                    "text_record": text_records[i],
                    # Convert list of MaskResult objects to a list of dictionaries
                    "mask_info": [dataclasses.asdict(mr) for mr in mask_results_list[i]]
                }
                # Write each record as a JSON object on a new line
                f.write(json.dumps(record_data, ensure_ascii=False) + '\n')
        print("Data successfully saved.")
    except Exception as e:
        print(f"An error occurred while saving data: {e}")


# --- Main execution block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process and partially re-identify PhysioNet Nurse data.')
    parser.add_argument('--in_dir', type=str, default="downloaded/physionet.org/files/deidentifiedmedicaltext/1.0/", help='Path to the input directory containing id.res and id.text.')
    parser.add_argument('--out_dir', type=str, default="data/physionet_nurse", help='Path to the directory to save processed data.')

    # Parse known arguments to ignore the ones injected by the execution environment
    args, unknown = parser.parse_known_args()

    res_file_path = os.path.join(args.in_dir, "id.res")
    text_file_path = os.path.join(args.in_dir, "id.text")
    output_directory = args.out_dir

    # --- Step 1: Read and Split Records ---
    print(f"Reading records from {res_file_path} and {text_file_path}...")
    res_records = read_and_split_records(res_file_path)
    text_records = read_and_split_records(text_file_path)
    print(f"Read {len(res_records)} res records and {len(text_records)} text records.")

    # --- Step 2 & 3: Process, Map, and Filter Records ---
    print("\nProcessing and filtering records...")
    # process_and_filter_records now returns a list of lists of MaskResult objects
    matched_res_records, matched_text_records, matched_mask_results = process_and_filter_records(res_records, text_records)
    print(f"\nFinished processing. {len(matched_res_records)} records passed filtering.")

    # --- Step 4: Partially Re-identify Filtered Records ---
    partially_reidentified_records = []
    if matched_res_records:
        print("\nPartially re-identifying filtered records...")
        for i in range(len(matched_res_records)):
            original_masked_record = matched_res_records[i]
            # Pass the list of MaskResult objects for the current record
            mask_results_for_record = matched_mask_results[i]
            reidentified_record = partially_reidentify_record(original_masked_record, mask_results_for_record)
            partially_reidentified_records.append(reidentified_record)
        print(f"Generated partially re-identified data for {len(partially_reidentified_records)} records.")

    # --- Step 5: Save Processed Data ---
    if matched_res_records: # Only save if there are matched records
        print(f"\nSaving processed data to {output_directory}...")
        # Pass the list of lists of MaskResult objects to save_processed_data
        save_processed_data(output_directory, matched_text_records, partially_reidentified_records, matched_res_records, matched_mask_results)
        print("\nProcessing and saving complete.")
    else:
        print("\nNo records were fully matched, skipping save operation.")
