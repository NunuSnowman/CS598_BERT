import re
import os
import json

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
        Step 1: Parses the structure of res_record, splitting it into alternating non-mask text segments and mask groups.

        Returns:
            list: A list of (segment_type, data) tuples.
                  segment_type is "non_mask" or "mask_group".
                  data is a non-mask text string or a list of mask contents.
        """
        segments = []
        current_res_pos = 0
        mask_matches = list(re.finditer(r"\[\*\*(.*?)\*\*\]", self.res_record))
        mask_idx = 0

        while current_res_pos < len(self.res_record):
            next_mask = mask_matches[mask_idx] if mask_idx < len(mask_matches) else None

            if next_mask and next_mask.start() == current_res_pos:
                # Current position is the start of a mask, process a mask group
                mask_group_contents = []
                # group_start = current_res_pos # Start position of the mask group in res_record
                temp_pos = current_res_pos # Temporary pointer to find consecutive masks

                # Find all consecutive masks
                while mask_idx < len(mask_matches) and mask_matches[mask_idx].start() == temp_pos:
                    m = mask_matches[mask_idx]
                    mask_group_contents.append(m.group(1)) # Extract mask content
                    temp_pos = m.end() # Temporary pointer skips the current mask's length
                    mask_idx += 1 # Advance to the next potential mask_match

                # group_end = temp_pos # End position of the mask group in res_record
                segments.append(("mask_group", mask_group_contents))
                current_res_pos = temp_pos # Main pointer skips the entire mask group

            else:
                # Current position is the start of non-mask text (or end of string after all masks processed)
                non_mask_start = current_res_pos
                next_mask_start = next_mask.start() if next_mask else len(self.res_record)
                non_mask_text = self.res_record[non_mask_start : next_mask_start]

                # Only non-empty non-mask segments are meaningful as anchors or content
                # If it's an empty string, it might be the gap between consecutive masks. We don't store it as a separate "non_mask" segment.
                # This simplifies the subsequent alignment logic as we only focus on non-empty anchors.
                if non_mask_text:
                    segments.append(("non_mask", non_mask_text))

                current_res_pos = next_mask_start # Main pointer skips this non-mask segment

        return segments

    def _distribute_text_to_masks(self, mask_contents_list: list[str], text_segment_for_group: str):
        """
        Step 3 Helper: Distributes a segment of original text to a list of mask contents.
        """
        if not mask_contents_list:
            if text_segment_for_group.strip():
                # This should not happen unless parsing or alignment logic is flawed
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
            # Optional logging:
            # print(f"Info: Splitting text '{text_segment_for_group[:50]}...' ({len(words)} words) among {num_masks} masks: {mask_contents_list}.")

            for mask_content in mask_contents_list:
                num_words_this_mask = words_per_mask_base + (1 if extra_words > 0 else 0)
                if extra_words > 0:
                    extra_words -= 1

                end_idx = min(word_idx + num_words_this_mask, len(words))
                assigned_text_words = words[word_idx : end_idx]
                assigned_text = " ".join(assigned_text_words)

                self.mapping[mask_content] = assigned_text
                word_idx += len(assigned_text.split()) # Use split().len() for robustness in counting words assigned

            # Optional warnings if the total assigned words don't match the original words (should not happen with current logic)
            # if word_idx < len(words):
            #      print(f"Warning: Word count mismatch during splitting.")

    def map_record(self) -> dict:
        """
        Main Step: Executes the record parsing and alignment mapping process.

        Returns:
            dict: A dictionary mapping mask content to original text.
        """
        # Step 1: Parse the structure of res_record
        self._parsed_segments = self._parse_res_segments()
        # print(f"Parsed segments: {self._parsed_segments}")

        # Step 2: Iterate through the parsed segments and align with text_record
        text_current_pos = 0 # Pointer, tracks position in text_record

        for i in range(len(self._parsed_segments)):
            segment_type, segment_data = self._parsed_segments[i]

            if segment_type == "non_mask":
                # Current segment is a non-mask anchor
                anchor = segment_data

                # Find the anchor in text_record starting from the current position
                anchor_match_start_in_text = self.text_record.find(anchor, text_current_pos)

                if anchor_match_start_in_text == -1:
                    # Anchor not found, indicates misalignment
                    print(f"Warning: Anchor '{anchor[:50]}...' not found in text after pos {text_current_pos}. Alignment likely broken for this record.")
                    # Cannot align, return the partial mapping found so far
                    return self.mapping # Interrupt processing and return partial result

                anchor_match_end_in_text = anchor_match_start_in_text + len(anchor)

                # text_segment_for_prev_group is the text in text_record,
                # from the end of the previous anchor position (text_current_pos) to the start of the current anchor position (anchor_match_start_in_text).
                # This text corresponds to the mask group in res_record that was the *previous* segment (if it exists).

                if i > 0: # Only if this is not the first segment
                    prev_segment_type, prev_segment_data = self._parsed_segments[i-1]
                    # If the previous segment was a mask group (which should be the case if the current is non_mask, given how segments are parsed)
                    if prev_segment_type == "mask_group":
                        text_segment_for_group = self.text_record[text_current_pos : anchor_match_start_in_text]
                        # Step 3: Distribute text to the previous mask group
                        self._distribute_text_to_masks(prev_segment_data, text_segment_for_group)
                    # Else: If the previous was also non_mask, it means there was no mask group between two non-empty anchors in res. text_segment_for_group should be empty, _distribute_text_to_masks will not be called, which is correct.

                # Update the lookup position in text_record to the end of the currently matched anchor
                text_current_pos = anchor_match_end_in_text

            else: # segment_type == "mask_group"
                # Current segment is a mask group.
                # The corresponding original text segment will be determined when the *next* non-mask anchor is found.
                # text_current_pos at this point is the start position of the text corresponding to this mask group.
                pass # No action needed here, wait for the next anchor

        # --- Handle the last segment ---
        # Check if the last segment was a mask_group
        if self._parsed_segments and self._parsed_segments[-1][0] == "mask_group":
            last_mask_group_contents = self._parsed_segments[-1][1]
            # The corresponding text is all text in text_record from text_current_pos to the end
            remaining_text = self.text_record[text_current_pos:]
            # Step 3: Distribute text to the last mask group
            self._distribute_text_to_masks(last_mask_group_contents, remaining_text)

        # Optional: Check if there is any remaining text in text_record that wasn't mapped (if the last segment was non_mask)
        elif text_current_pos < len(self.text_record):
            trailing_text = self.text_record[text_current_pos:]
            if trailing_text.strip():
                print(f"Warning: Trailing non-empty text found in text_record but no corresponding segments in res_record: '{trailing_text[:50]}...'")


        # Step 4: Return the final mapping result
        return self.mapping

import argparse

# Ensure the main execution block uses the new OOD class
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pseudo de-identify DocRED data by masking person names.')
    parser.add_argument('--in_path', type=str, default="downloaded/physionet.org/files/deidentifiedmedicaltext/1.0/", help='Path to the directory containing DocRED data files.')
    parser.add_argument('--out_path', type=str, default="tmp/physionet_nurse_filtered", help='Path to the directory to save processed data.')

    args = parser.parse_args()
    res_file = os.path.join(args.in_path,"id.res")
    text_file = os.path.join(args.in_path,"id.text")
    filtered_output_dir = args.out_path

    res_records = read_and_split_records(res_file)
    text_records = read_and_split_records(text_file)

    # Lists to store matched data
    matched_res_records = []
    matched_text_records = []
    matched_mappings = []

    if len(res_records) == len(text_records) and res_records:
        for i in range(len(res_records)):
            print(f"--- Processing Record {i+1} ---")
            res_record = res_records[i]
            text_record = text_records[i]

            # Skip empty records, but do NOT include them in matched data
            if not res_record.strip() or not text_record.strip():
                print(f"Warning: Skipping empty record {i+1} - will not be included in filtered output.")
                print("-" * 20)
                continue

            # Use the object-oriented processor
            processor = RecordProcessor(res_record, text_record)
            mapping = processor.map_record() # This now includes all masks, with None for unmapped

            # --- Filtering Logic ---
            all_masks_in_record = re.findall(r"\[\*\*(.*?)\*\*\]", res_record)
            total_masks = len(set(all_masks_in_record))

            # Count how many masks were successfully mapped (value is not None)
            mapped_count = sum(1 for value in mapping.values() if value is not None)

            # Decide if the record is matched: total masks equals successfully mapped masks
            # This includes records with 0 masks (0 == 0)
            is_matched = (total_masks == mapped_count)

            print(f"Record {i+1} Mapping (Total Masks: {total_masks}, Mapped: {mapped_count}):")
            if mapping or total_masks == 0: # Print mapping if any masks exist or if it's a no-mask record
                # Try to print mappings in the order masks appear in res_record for better readability
                all_masks_in_order = re.findall(r"\[\*\*(.*?)\*\*\]", res_record)
                printed_masks = set()
                for mask_content in all_masks_in_order:
                    # mapping.get(mask_content, None) is safe here as map_record ensures all masks are keys
                    actual_text = mapping.get(mask_content, None)
                    print(f"  Mask: '{mask_content}' -> Actual: '{actual_text}'")
                    printed_masks.add(mask_content) # Add to set to track printed masks

                # Check if any masks in the mapping weren't in the findall result (shouldn't happen)
                for mask_content in mapping.keys():
                    if mask_content not in printed_masks:
                        print(f"  Mask: '{mask_content}' -> Actual: '{mapping[mask_content]}' (Warning: Mask not found by findall?)")
            elif not mapping and total_masks > 0:
                # This case happens if map_record returns an empty mapping but there were masks in the record
                print("  Mapping process failed and returned an empty mapping.")


            # --- Save matched records ---
            if is_matched:
                print(f"Record {i+1} is matched ({mapped_count}/{total_masks}). It will be saved.")
                matched_res_records.append(res_record)
                matched_text_records.append(text_record)
                matched_mappings.append(mapping)
            else:
                print(f"Record {i+1} is NOT fully matched ({mapped_count}/{total_masks}). It will be skipped.")

            print("-" * 20)

    elif not res_records or not text_records:
        print("Could not read records from one or both files.")
    else:
        print(f"Error: Number of records in {res_file} ({len(res_records)}) and {text_file} ({len(text_records)}) do not match. Cannot proceed with mapping.")


    # --- Save collected matched data ---
    if matched_res_records: # Only save if there's data
        filtered_res_file = os.path.join(filtered_output_dir, "id.res")
        filtered_text_file = os.path.join(filtered_output_dir, "id.text")
        filtered_mapping_file = os.path.join(filtered_output_dir, "mapping.jsonl")

        # Create the output directory if it doesn't exist
        os.makedirs(filtered_output_dir, exist_ok=True)

        # Save matched res records (one per line)
        try:
            with open(filtered_res_file, 'w', encoding='utf-8') as f:
                for record in matched_res_records:
                    f.write(record.replace('\n', '\t') + '\n') # Write record followed by newline
            print(f"Successfully saved {len(matched_res_records)} matched res records to {filtered_res_file}")
        except Exception as e:
            print(f"Error saving matched res records to {filtered_res_file}: {e}")

        # Save matched text records (one per line)
        try:
            with open(filtered_text_file, 'w', encoding='utf-8') as f:
                for record in matched_text_records:
                    f.write(record.replace('\n', '\t') + '\n') # Write record followed by newline
            print(f"Successfully saved {len(matched_text_records)} matched text records to {filtered_text_file}")
        except Exception as e:
            print(f"Error saving matched text records to {filtered_text_file}: {e}")

        # Save matched mappings (one JSON object per line)
        try:
            with open(filtered_mapping_file, 'w', encoding='utf-8') as f:
                for record_mapping in matched_mappings:
                    # Convert each record's mapping dictionary to a JSON string
                    modified_mapping = {f"[**{key}**]": value for key, value in record_mapping.items()}
                    json_string = json.dumps(modified_mapping, ensure_ascii=False)
                    # Write the JSON string followed by a newline
                    f.write(json_string + '\n')
            print(f"Successfully saved {len(matched_mappings)} matched mappings to {filtered_mapping_file}")
        except Exception as e:
            print(f"Error saving matched mappings to {filtered_mapping_file}: {e}")

    else:
        print("No records were fully matched. No filtered data saved.")