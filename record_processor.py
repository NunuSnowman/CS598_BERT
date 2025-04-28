import re
import dataclasses
import json # For pretty printing results

from common import MaskInfo, ProcessedRecord
from record_validator import validate_mapping

# Assuming MASK_TOKEN_PATTERN is defined elsewhere, e.g.:
# Modified pattern to capture the full mask string (group 1) and the label (group 2)
MASK_TOKEN_PATTERN = re.compile(r"(\[\*\*(.*?)\*\*\])")

class RecordProcessor:
    def __init__(self, res_record: str, text_record: str):
        self.res_record = res_record
        self.text_record = text_record
        self.masks: list[MaskInfo] = self._map_record()
        self._parsed_segments = None # Stores the parsed segments of res_record

    def _parse_res_segments(self):
        """
        Parses the structure of res_record, splitting it into alternating non-mask text segments and mask groups.

        Returns:
            list: A list of (segment_type, data) tuples.
                  segment_type is "non_mask" or "mask_group".
                  data is a non-mask text string or a list of (full_mask_string, mask_content) tuples.
        """
        segments = []
        current_res_pos = 0
        # Use the modified pattern to get both the full mask and the inner label
        mask_matches = list(re.finditer(MASK_TOKEN_PATTERN, self.res_record))
        mask_idx = 0

        while current_res_pos < len(self.res_record):
            # Determine the start of the next mask or the end of the res_record
            next_mask_start = len(self.res_record)
            next_mask_match = None
            if mask_idx < len(mask_matches):
                m = mask_matches[mask_idx]
                if m.start() >= current_res_pos:
                    next_mask_start = m.start()
                    next_mask_match = m
                else: # Should not happen with correct regex and indexing, but safety
                    mask_idx += 1
                    continue

            if next_mask_start > current_res_pos:
                # Current position is the start of non-mask text
                non_mask_text = self.res_record[current_res_pos : next_mask_start]
                # Only non-empty non-mask segments are meaningful as anchors or content
                if non_mask_text:
                    segments.append(("non_mask", non_mask_text))
                current_res_pos = next_mask_start # Move past the non-mask segment

            # Check if the current position is now the start of a mask
            if next_mask_match and next_mask_match.start() == current_res_pos:
                # Process a mask group
                mask_group_details = [] # Stores tuples: (full_mask_string, mask_content)
                temp_pos = current_res_pos # Temporary pointer to find consecutive masks

                # Find all consecutive masks starting exactly at temp_pos
                current_mask_match_idx = mask_idx # Start checking from the current mask_idx
                while current_mask_match_idx < len(mask_matches):
                    m = mask_matches[current_mask_match_idx]
                    if m.start() == temp_pos:
                        mask_group_details.append((m.group(1), m.group(2))) # Capture full mask and label
                        temp_pos = m.end() # Temporary pointer skips the current mask's length
                        current_mask_match_idx += 1 # Advance to the next potential mask_match
                    else:
                        break # Stop if the next match is not consecutive

                segments.append(("mask_group", mask_group_details))
                current_res_pos = temp_pos # Main pointer skips the entire mask group
                mask_idx = current_mask_match_idx # Update main mask_idx to where the inner loop stopped

        return segments


    def _distribute_text_to_masks(self, mask_details_list: list[tuple[str, str]], text_segment_start_in_text: int, text_segment_end_in_text: int):
        """
        Helper: Distributes a segment of original text to a list of mask details and records results.
        mask_details_list is a list of (full_mask_string, mask_content) tuples.
        Records results directly to self._results as MaskResult objects.

        If num_masks == 1, the entire text segment is assigned to the single mask.
        If num_masks > 1 (consecutive masks), the text segment is split based on word count.
        """
        full_text_segment = self.text_record[text_segment_start_in_text : text_segment_end_in_text]
        words = full_text_segment.split()
        num_masks = len(mask_details_list)

        if not num_masks:
            if full_text_segment.strip():
                print(f"Warning: Found text segment '{full_text_segment[:50]}...' but no corresponding masks in distribution step.")
            return

        if num_masks == 1:
            # If there is only one mask, assign the entire text segment to it
            full_mask_string, mask_label = mask_details_list[0]
            # The start and end are simply the boundaries of the text segment provided
            self.masks.append(MaskInfo(
                label=mask_label,
                text=full_text_segment,
                start=text_segment_start_in_text,
                end=text_segment_end_in_text,
                masked_text=full_mask_string
            ))
        else:
            # If there are multiple consecutive masks, apply the word-based average distribution heuristic
            words_per_mask_base = len(words) // num_masks
            extra_words = len(words) % num_masks
            word_idx = 0 # Index in the 'words' list
            current_char_pos_in_segment = 0 # Character position within the full_text_segment for text assignment

            for full_mask_string, mask_label in mask_details_list:
                num_words_this_mask = words_per_mask_base + (1 if extra_words > 0 else 0)
                if extra_words > 0:
                    extra_words -= 1

                # Get the words for this mask
                assigned_text_words = words[word_idx : word_idx + num_words_this_mask]
                assigned_text = " ".join(assigned_text_words)

                # Calculate start and end indices within the original text_record
                # Find the start of the assigned text within the full_text_segment
                # Use the character position to narrow the search range for find
                search_start_in_segment = current_char_pos_in_segment
                start_in_segment = full_text_segment.find(assigned_text, search_start_in_segment)

                if start_in_segment == -1 and assigned_text.strip():
                    # Fallback/Warning if exact match not found, try to approximate start based on word index
                    print(f"Warning: Could not find assigned text '{assigned_text[:50]}...' within segment '{full_text_segment[:50]}...' for mask '{mask_label}'. Approximating position.")
                    # Re-calculate assigned_text just in case
                    assigned_text = " ".join(words[word_idx : word_idx + num_words_this_mask])
                    # Approximate start based on summing lengths of previous words and spaces
                    # Account for potential leading space if not the very first word
                    approx_start = sum(len(w) + 1 for w in words[:word_idx]) # +1 for space after each word
                    # Clamp approximation to segment bounds
                    start_in_segment = max(0, min(approx_start, len(full_text_segment)))


                # Calculate end index (exclusive)
                # If start_in_segment is -1, the assigned text wasn't found/approximated,
                # so setting end to -1 implies an invalid range.
                end_in_segment = start_in_segment + len(assigned_text) if start_in_segment != -1 else -1 # End is start + length


                # Record the result
                self.masks.append(MaskInfo(
                    label=mask_label,
                    text=assigned_text,
                    # Add the segment's start offset to get the absolute position in text_record
                    start=text_segment_start_in_text + start_in_segment if start_in_segment != -1 else -1, # Indicate failure with -1 or handle differently
                    end=text_segment_start_in_text + end_in_segment if start_in_segment != -1 else -1,
                    masked_text=full_mask_string
                ))

                # Update indices for the next mask
                word_idx += num_words_this_mask
                # Update character position based on the end of the assigned text in the segment
                if start_in_segment != -1:
                    current_char_pos_in_segment = end_in_segment
                else:
                    # If find failed, try to estimate the next character position based on word count
                    current_char_pos_in_segment = sum(len(w) + 1 for w in words[:word_idx])


    def _map_record(self) -> list[MaskInfo]:
        """
        Executes the record parsing and alignment process, generating detailed results.

        Returns:
            list[MaskInfo]: A list of MaskResult objects for each identified mask.
                              Returns an empty list if alignment fails critically.
        """
        self.masks = [] # Reset results for each call
        self._parsed_segments = self._parse_res_segments()
        text_current_pos = 0 # Pointer, tracks position in text_record

        for i in range(len(self._parsed_segments)):
            segment_type, segment_data = self._parsed_segments[i]

            if segment_type == "non_mask":
                anchor = segment_data
                anchor_match_start_in_text = self.text_record.find(anchor, text_current_pos)

                if anchor_match_start_in_text == -1:
                    print(f"Warning: Anchor '{anchor[:50]}...' not found in text after pos {text_current_pos}. Alignment likely broken for this record.")
                    self.masks = [] # Clear results to indicate failure
                    return self.masks # Interrupt processing

                anchor_match_end_in_text = anchor_match_start_in_text + len(anchor)

                # If the previous segment was a mask group, the text between the previous
                # text_current_pos and the start of this anchor is the text for that group.
                if i > 0:
                    prev_segment_type, prev_segment_data = self._parsed_segments[i-1]
                    if prev_segment_type == "mask_group":
                        text_segment_start_for_group = text_current_pos
                        text_segment_end_for_group = anchor_match_start_in_text
                        # Only process if the text segment is not empty
                        if text_segment_start_for_group < text_segment_end_for_group:
                            self._distribute_text_to_masks(prev_segment_data, text_segment_start_for_group, text_segment_end_for_group)
                        elif prev_segment_data and self.text_record[text_segment_start_for_group:text_segment_end_for_group].strip():
                            # Case where text segment is empty but there were masks, could indicate a mismatch
                            print(f"Warning: Zero-length text segment found for mask group at res_pos corresponding to segment {i-1} but non-empty text expected near text_pos {text_current_pos}. Mask labels: {[lbl for _, lbl in prev_segment_data]}.")
                            # Decide how to handle: assign empty string, indicate failure, etc.
                            # For now, _distribute_text_to_masks will handle empty input gracefully.
                            self._distribute_text_to_masks(prev_segment_data, text_segment_start_for_group, text_segment_end_for_group)


                text_current_pos = anchor_match_end_in_text

            else: # segment_type == "mask_group"
                # Mask groups are processed when the *next* non-mask segment (anchor) is found,
                # or at the end of the record if the last segment is a mask group.
                pass

        # --- Handle the last segment if it's a mask group ---
        if self._parsed_segments and self._parsed_segments[-1][0] == "mask_group":
            last_mask_group_details = self._parsed_segments[-1][1]
            remaining_text_start = text_current_pos
            remaining_text_end = len(self.text_record)
            # Only process if there is text to distribute
            if remaining_text_start < remaining_text_end:
                self._distribute_text_to_masks(last_mask_group_details, remaining_text_start, remaining_text_end)
            elif last_mask_group_details and self.text_record[remaining_text_start:remaining_text_end].strip():
                # Case where remaining text segment is empty but there were masks
                print(f"Warning: Zero-length remaining text segment found for last mask group starting near text_pos {text_current_pos}. Mask labels: {[lbl for _, lbl in last_mask_group_details]}.")
                self._distribute_text_to_masks(last_mask_group_details, remaining_text_start, remaining_text_end)


        # Optional: Check for trailing text if the last segment was non_mask
        elif text_current_pos < len(self.text_record):
            trailing_text = self.text_record[text_current_pos:]
            if trailing_text.strip():
                print(f"Warning: Trailing non-empty text found in text_record but no corresponding segments in res_record: '{trailing_text[:50]}...' starting at index {text_current_pos}")
                # Decide if this constitutes an alignment failure or just unexpected extra text.

        return self.masks

    def get_processed_record(self) -> ProcessedRecord:
        """Returns the processed record as a ProcessedRecord object."""
        return ProcessedRecord(
            res_record=self.res_record,
            text_record=self.text_record,
            masks=self.masks
        )
# --- Test Cases ---
if __name__ == "__main__":

    def run_test_case(name: str, res: str, text: str):
        processor = RecordProcessor(res, text)
        print("Results:", validate_mapping(processor))
        # print(json.dumps([dataclasses.asdict(r) for r in results], indent=2))

    # Test Case 1: Basic case
    run_test_case(
        "Basic Case",
        "Hello [**First Name**] [**Last Name**], your account number is [**Account Number**].",
        "Hello John Doe, your account number is 12345."
    )

    # Test Case 2: Consecutive Masks
    run_test_case(
        "Consecutive Masks",
        "Names: [**Name1**][**Name2**]. ID: [**ID**].",
        "Names: AliceBob. ID: xyz."
    )

    # Test Case 3: Consecutive Masks with Space
    run_test_case(
        "Consecutive Masks with Space",
        "Names: [**Name1**] [**Name2**]. ID: [**ID**].",
        "Names: Alice Bob. ID: xyz."
    )

    # Test Case 5: Mask at the end
    run_test_case(
        "Mask at End",
        "The end is [**The End**]",
        "The end is near."
    )

    # Test Case 6: Multiple mask groups, interspersed text
    run_test_case(
        "Multiple Groups",
        "[**Part1**] and [**Part2**] then [**Part3**].",
        "First part and second part then third part."
    )

    # Test Case 7: No Masks
    run_test_case(
        "No Masks",
        "This record has no masks.",
        "This record has no masks."
    )

    # Test Case 9: Text longer than expected after last anchor
    run_test_case(
        "Trailing Text",
        "Name: [**Name**].",
        "Name: Bob."
    )

    # Test Case 10: Punctuation adjacent to masks
    run_test_case(
        "Punctuation Adjacent",
        "Is this [**Question**]? Yes!",
        "Is this a query? Yes!"
    )

    # Test Case 13: Empty res and text
    run_test_case(
        "Empty Res and Text",
        "",
        ""
    )

    # Test Case 15: Multiple consecutive masks with varying space in text
    run_test_case(
        "Consecutive Masks Varying Space",
        "Info: [**Data1**] [**Data2**] [**Data3**] Done.",
        "Info: abc def ghi Done." # Note double space and no space
    )

    # Test Case 16: Single mask matching multiple words (explicit test)
    run_test_case(
        "Single Mask Multi-Word",
        "Full name: [**Patient Full Name**]. Date of Birth: [**DOB**].",
        "Full name: Dr. Jane Smith Jr.. Date of Birth: 01/01/1990."
    )