import re
import os
import json
import argparse

# Assuming MASK_TOKEN_PATTERN is required for checking for masks
# Include the pattern here for a self-contained script
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

class RecordProcessor:
    def __init__(self, res_record: str, text_record: str):
        self.res_record = res_record
        self.text_record = text_record
        self.mapping = {}
        self._parsed_segments = None # Stores the parsed segments of res_record

    def _parse_res_segments(self):
        """
        Parses the structure of res_record, splitting it into alternating non-mask text segments and mask groups.

        Returns:
            list: A list of (segment_type, data) tuples.
                  segment_type is "non_mask" or "mask_group".
                  data is a non-mask text string or a list of mask contents.
        """
        segments = []
        current_res_pos = 0
        mask_matches = list(re.finditer(MASK_TOKEN_PATTERN, self.res_record))
        mask_idx = 0

        while current_res_pos < len(self.res_record):
            next_mask = mask_matches[mask_idx] if mask_idx < len(mask_matches) else None

            if next_mask and next_mask.start() == current_res_pos:
                # Current position is the start of a mask, process a mask group
                mask_group_contents = []
                temp_pos = current_res_pos # Temporary pointer to find consecutive masks

                # Find all consecutive masks
                while mask_idx < len(mask_matches) and mask_matches[mask_idx].start() == temp_pos:
                    m = mask_matches[mask_idx]
                    mask_group_contents.append(m.group(1)) # Extract mask content
                    temp_pos = m.end() # Temporary pointer skips the current mask's length
                    mask_idx += 1 # Advance to the next potential mask_match

                segments.append(("mask_group", mask_group_contents))
                current_res_pos = temp_pos # Main pointer skips the entire mask group

            else:
                # Current position is the start of non-mask text (or end of string after all masks processed)
                non_mask_start = current_res_pos
                next_mask_start = next_mask.start() if next_mask else len(self.res_record)
                non_mask_text = self.res_record[non_mask_start : next_mask_start]

                # Only non-empty non-mask segments are meaningful as anchors or content
                if non_mask_text:
                    segments.append(("non_mask", non_mask_text))

                current_res_pos = next_mask_start # Main pointer skips this non-mask segment

        return segments

    def _distribute_text_to_masks(self, mask_contents_list: list[str], text_segment_for_group: str):
        """
        Helper: Distributes a segment of original text to a list of mask contents.
        """
        if not mask_contents_list:
            if text_segment_for_group.strip():
                print(f"Warning: Found text segment '{text_segment_for_group[:50]}...' but no corresponding masks in distribution step.")
            return

        words = text_segment_for_group.split()
        num_masks = len(mask_contents_list)

        if num_masks == 1:
            # If there is only one mask, assign the entire text segment to it
            self.mapping[mask_contents_list[0]] = text_segment_for_group
        else:
            # If there are multiple consecutive masks, apply the word-based average distribution heuristic
            words_per_mask_base = len(words) // num_masks
            extra_words = len(words) % num_masks
            word_idx = 0

            for mask_content in mask_contents_list:
                num_words_this_mask = words_per_mask_base + (1 if extra_words > 0 else 0)
                if extra_words > 0:
                    extra_words -= 1

                # Ensure we don't go out of bounds
                end_idx = min(word_idx + num_words_this_mask, len(words))
                assigned_text_words = words[word_idx : end_idx]
                assigned_text = " ".join(assigned_text_words)

                self.mapping[mask_content] = assigned_text
                word_idx += len(assigned_text_words) # Move the index by the actual number of words assigned

    def map_record(self) -> dict:
        """
        Executes the record parsing and alignment mapping process.

        Returns:
            dict: A dictionary mapping mask content to original text. Returns empty dict if alignment fails.
        """
        self._parsed_segments = self._parse_res_segments()
        text_current_pos = 0 # Pointer, tracks position in text_record

        for i in range(len(self._parsed_segments)):
            segment_type, segment_data = self._parsed_segments[i]

            if segment_type == "non_mask":
                anchor = segment_data
                anchor_match_start_in_text = self.text_record.find(anchor, text_current_pos)

                if anchor_match_start_in_text == -1:
                    print(f"Warning: Anchor '{anchor[:50]}...' not found in text after pos {text_current_pos}. Alignment likely broken for this record.")
                    # Clear mapping to indicate failure for this record
                    self.mapping = {}
                    return self.mapping # Interrupt processing

                anchor_match_end_in_text = anchor_match_start_in_text + len(anchor)

                if i > 0:
                    prev_segment_type, prev_segment_data = self._parsed_segments[i-1]
                    if prev_segment_type == "mask_group":
                        text_segment_for_group = self.text_record[text_current_pos : anchor_match_start_in_text]
                        self._distribute_text_to_masks(prev_segment_data, text_segment_for_group)

                text_current_pos = anchor_match_end_in_text

            else: # segment_type == "mask_group"
                pass # Wait for the next anchor

        # --- Handle the last segment ---
        if self._parsed_segments and self._parsed_segments[-1][0] == "mask_group":
            last_mask_group_contents = self._parsed_segments[-1][1]
            remaining_text = self.text_record[text_current_pos:]
            self._distribute_text_to_masks(last_mask_group_contents, remaining_text)

        # Optional: Check for trailing text if the last segment was non_mask
        elif text_current_pos < len(self.text_record):
            trailing_text = self.text_record[text_current_pos:]
            if trailing_text.strip():
                print(f"Warning: Trailing non-empty text found in text_record but no corresponding segments in res_record: '{trailing_text[:50]}...'")

        return self.mapping

def process_and_filter_records(res_records, text_records):
    """
    Processes pairs of res and text records, performs mapping, and filters
    records where not all masks are successfully mapped.

    Args:
        res_records (list): List of records from the .res file.
        text_records (list): List of records from the .text file.

    Returns:
        tuple: (matched_res_records, matched_text_records, matched_mappings)
               Lists containing records and mappings that passed the filtering.
    """
    matched_res_records = []
    matched_text_records = []
    matched_mappings = []

    if len(res_records) != len(text_records):
        print("Error: Number of records in res and text files do not match. Cannot process.")
        return [], [], []

    for i in range(len(res_records)):
        res_record = res_records[i]
        text_record = text_records[i]

        # Skip empty records
        if not res_record.strip() or not text_record.strip():
            print(f"Warning: Skipping empty record {i+1}.")
            continue

        processor = RecordProcessor(res_record, text_record)
        mapping = processor.map_record()

        # Filtering Logic: A record is matched if all masks found in the res_record
        # are present as keys in the mapping with a non-None value.
        all_masks_in_record = re.findall(MASK_TOKEN_PATTERN, res_record)
        total_masks = len(set(all_masks_in_record)) # Use set to count unique mask contents

        # Check if mapping contains all unique masks found in the res_record and their values are not None
        is_matched = False
        if total_masks == 0 and not mapping:
            # Case: No masks in res_record, and mapping is empty (correctly)
            is_matched = True
        elif total_masks > 0 and mapping:
            # Case: Masks exist, check if all are mapped
            mapped_count = sum(1 for mask_content in set(all_masks_in_record) if mask_content in mapping and mapping[mask_content] is not None)
            is_matched = (total_masks == mapped_count)
        # Else: total_masks > 0 but mapping is empty (map_record failed), is_matched remains False

        if is_matched:
            print(f"Record {i+1} is matched ({total_masks}/{total_masks} masks mapped).")
            matched_res_records.append(res_record)
            matched_text_records.append(text_record)
            # Modify the mapping to use the full token format [**mask_content**] as keys for saving
            formatted_mapping = {f"[**{key}**]": value for key, value in mapping.items()}
            matched_mappings.append(formatted_mapping)
        else:
            print(f"Record {i+1} is NOT fully matched (found {total_masks} masks, {len(mapping)} mapped). Skipping.")

    return matched_res_records, matched_text_records, matched_mappings


def partially_reidentify_record(masked_record: str, mapping: dict) -> str:
    """
    Applies mapping to a masked record, except for masks containing 'Name'/'name'.

    Args:
        masked_record (str): The record from the .res file.
        mapping (dict): The mapping from mask token (e.g., '[**Name**]') to original text.
                        Keys are expected to be in the format '[**mask_content**]'.

    Returns:
        str: The partially re-identified record.
    """
    reidentified_record = masked_record
    # Sort mapping keys by length in descending order to avoid partial replacement of longer masks
    # when a shorter mask content is a substring of a longer one.
    # The mapping keys are already in '[**mask_content**]' format from process_and_filter_records
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


def save_processed_data(output_dir: str, original_text_records: list[str], partially_reidentified_records: list[str], original_masked_records: list[str], mappings: list[dict]):
    """
    Saves the processed data to the specified output directory.

    Args:
        output_dir (str): Directory to save the output files.
        original_text_records (list[str]): List of original text records.
        partially_reidentified_records (list[str]): List of partially re-identified records.
        original_masked_records (list[str]): List of original masked records (from id.res).
        mappings (list[dict]): List of mapping dictionaries.
    """
    os.makedirs(output_dir, exist_ok=True)

    output_text_path = os.path.join(output_dir, 'original_data.txt')
    output_res_path = os.path.join(output_dir, 'masked_data.txt') # Partially re-identified
    output_original_res_path = os.path.join(output_dir, 'raw_masked.text') # Original masked
    output_mapping_path = os.path.join(output_dir, 'mapping.jsonl')

    try:
        with open(output_text_path, 'w', encoding='utf-8') as f:
            for record in original_text_records:
                f.write(record.replace('\n', '\t') + '\n')
        print(f"Successfully saved {len(original_text_records)} original text records to {output_text_path}")

        with open(output_res_path, 'w', encoding='utf-8') as f:
            for record in partially_reidentified_records:
                f.write(record.replace('\n', '\t') + '\n')
        print(f"Successfully saved {len(partially_reidentified_records)} partially re-identified records to {output_res_path}")

        with open(output_original_res_path, 'w', encoding='utf-8') as f:
            for record in original_masked_records:
                f.write(record.replace('\n', '\t') + '\n')
        print(f"Successfully saved {len(original_masked_records)} original masked records to {output_original_res_path}")


        with open(output_mapping_path, 'w', encoding='utf-8') as f:
            for mapping in mappings:
                json_string = json.dumps(mapping, ensure_ascii=False)
                f.write(json_string + '\n')
        print(f"Successfully saved {len(mappings)} mappings to {output_mapping_path}")

    except Exception as e:
        print(f"An error occurred during saving: {e}")


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
    matched_res_records, matched_text_records, matched_mappings = process_and_filter_records(res_records, text_records)
    print(f"\nFinished processing. {len(matched_res_records)} records passed filtering.")

    # --- Step 4: Partially Re-identify Filtered Records ---
    partially_reidentified_records = []
    if matched_res_records:
        print("\nPartially re-identifying filtered records...")
        for i in range(len(matched_res_records)):
            original_masked_record = matched_res_records[i]
            mapping = matched_mappings[i] # Use the already formatted mapping with [**mask**] keys
            reidentified_record = partially_reidentify_record(original_masked_record, mapping)
            partially_reidentified_records.append(reidentified_record)
        print(f"Generated partially re-identified data for {len(partially_reidentified_records)} records.")

    # --- Step 5: Save Processed Data ---
    if matched_res_records: # Only save if there are matched records
        print(f"\nSaving processed data to {output_directory}...")
        save_processed_data(output_directory, matched_text_records, partially_reidentified_records, matched_res_records, matched_mappings)
        print("\nProcessing and saving complete.")
    else:
        print("\nNo records were fully matched, skipping save operation.")