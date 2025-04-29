from dataclasses import dataclass

from transformers import BertTokenizer, BertForTokenClassification, BertTokenizerFast
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
from typing import List, Tuple, Optional

from common import ProcessedRecord, MaskInfo # Assuming common.py contains these definitions

# --- Configuration ---
MODEL_NAME = 'bert-base-uncased'
MAX_LENGTH = 128 # Max sequence length for tokenization and padding
TOKEN_OVERLAP = 64
BATCH_SIZE = 8   # Increased batch size for efficiency
NUM_EPOCHS = 50   # Train for more epochs
LEARNING_RATE = 1e-4
SAVE_DIRECTORY = "./tmp/saved_models"

label_map = {'O': 0, 'B-NAME': 1, 'I-NAME': 2,
             'B-LOCATION': 3, 'I-LOCATION': 4,
             'B-DATE': 5, 'I-DATE': 6}
id_to_label = {v: k for k, v in label_map.items()} # Create reverse mapping
num_labels = len(label_map)

def create_processed_record(text_record: str, masks: List[Tuple[int, int, str]]) -> ProcessedRecord:
    mask_info_list: List[MaskInfo] = []
    for start, end, label in masks:
        # Extract the text covered by the mask
        matched_text = text_record[start:end]
        # Generate a plausible masked_text string based on the label
        mask_string = f"[**{label}**]" # Example format

        mask_info = MaskInfo(
            start=start,
            end=end,
            label=label,
            text=matched_text,
            masked_text=mask_string
        )
        mask_info_list.append(mask_info)

    return ProcessedRecord(
        res_record="",  # Requested to be empty
        text_record=text_record,
        mask_info=mask_info_list
    )

# --- Example Usage of the Helper ---

# Creating the train_data using the helper
train_data: List[ProcessedRecord] = [
    create_processed_record("Dr. John Doe will see the patient.", [(4, 12, "NAME")]), # "John Doe"
    create_processed_record("The appointment is with Jane Smith tomorrow.", [(28, 38, "NAME")]), # "Jane Smith"
    create_processed_record("Report for Patient ID 12345.", []), # No entities
    create_processed_record("Contact Dr. Emily White at the clinic.", [(12, 23, "NAME")]), # "Emily White"
    create_processed_record("This is a normal sentence.", []),
    create_processed_record("Please refer to the notes from Dr. Brown.", [(33, 38, "NAME")]), # "Brown"
    create_processed_record("Mr. David Lee is the attending physician.", [(4, 13, "NAME")]), # "David Lee"
    create_processed_record("No specific name mentioned here.", []),
    create_processed_record("The nurse, Mary Johnson, provided the care.", [(11, 23, "NAME")]), # "Mary Johnson"
    create_processed_record("Patient admitted by Dr. Robert Green.", [(24, 36, "NAME")]), # "Robert Green"
    create_processed_record("Consultation with Dr. Alice Williams.", [(21, 36, "NAME")]), # "Alice Williams"
    create_processed_record("Patient transferred to Dr. Michael Brown's care.", [(25, 38, "NAME")]), # "Michael Brown"
    create_processed_record("Examined by Dr. Sarah Davis.", [(16, 27, "NAME")]), # "Sarah Davis"

    # Adding LOCATION and DATE examples
    create_processed_record("The meeting is in London.", [(19, 25, "LOCATION")]), # "London"
    create_processed_record("We will arrive on Monday.", [(18, 24, "DATE")]), # "Monday"
    create_processed_record("Visited Paris, France last year.", [(8, 13, "LOCATION"), (15, 21, "LOCATION")]), # "Paris", "France"
    create_processed_record("The deadline is December 31, 2024.", [(17, 33, "DATE")]), # "December 31, 2024"
    create_processed_record("He lives in New York City.", [(12, 25, "LOCATION")]), # "New York City"
    create_processed_record("The event is scheduled for next Tuesday.", [(28, 39, "DATE")]), # "next Tuesday"
    create_processed_record("Patient report from London, UK dated January 5, 2023.", [(20, 26, "LOCATION"), (28, 31, "LOCATION"), (38, 51, "DATE")]), # "London", "UK", "January 5, 2023"
    create_processed_record("Travel to Rome on the 15th.", [(9, 13, "LOCATION"), (21, 27, "DATE")]), # "Rome", "15th"
    create_processed_record("Appointment is on Friday afternoon.", [(18, 24, "DATE")]), # "Friday"
    create_processed_record("Location: Seattle. Date: 2025-04-27.", [(10, 17, "LOCATION"), (24, 34, "DATE")]), # "Seattle", "2025-04-27"
    create_processed_record("Dr. Chen is relocating to Boston next month.", [(28, 34, "LOCATION"), (35, 45, "DATE")]), # "Boston", "next month"
    create_processed_record("Meeting with Mr. Adams in Berlin on Wednesday.", [(20, 25, "NAME"), (30, 36, "LOCATION"), (40, 49, "DATE")]), # "Adams", "Berlin", "Wednesday"
]

# Creating the test_data using the helper
test_data: List[ProcessedRecord] = [
    create_processed_record("The patient name is Michael Clark.", [(24, 35, "NAME")]), # "Michael Clark"
    create_processed_record("Follow up with Dr. Susan Davis.", [(20, 31, "NAME")]), # "Susan Davis"
    create_processed_record("General health information.", []), # No entity
    create_processed_record("Check the records for Alice.", [(24, 29, "NAME")]), # "Alice"
    create_processed_record("Meeting with Dr. Williams and Dr. Jones.", [(19, 28, "NAME"), (37, 42, "NAME")]), # "Williams", "Jones"
    create_processed_record("Report filed by Dr. Evans.", [(21, 26, "NAME")]), # "Evans"
    create_processed_record("The primary contact is Mr. Thomas Green.", [(28, 39, "NAME")]), # "Thomas Green"

    # Test cases with LOCATION and DATE
    create_processed_record("Visited Washington D.C. last week.", [(9, 24, "LOCATION"), (25, 34, "DATE")]), # "Washington D.C.", "last week"
    create_processed_record("Conference held in Tokyo on March 10.", [(21, 26, "LOCATION"), (30, 39, "DATE")]), # "Tokyo", "March 10"
    create_processed_record("Dr. Wilson will be in Chicago until Friday.", [(4, 10, "NAME"), (25, 32, "LOCATION"), (39, 45, "DATE")]), # "Wilson", "Chicago", "Friday"
    create_processed_record("The report is due by 2024-12-01.", [(21, 31, "DATE")]), # "2024-12-01"
    create_processed_record("Sent details to Mr. White in Sydney.", [(24, 29, "NAME"), (33, 39, "LOCATION")]), # "White", "Sydney"
    create_processed_record("The event is scheduled for tomorrow in London.", [(28, 35, "DATE"), (39, 45, "LOCATION")]), # "tomorrow", "London"
    create_processed_record("Patient arrived from New York on July 4th.", [(22, 30, "LOCATION"), (34, 43, "DATE")]), # "New York", "July 4th"
]


def process_data_label(data: List[ProcessedRecord], tokenizer: BertTokenizerFast, max_length: int):
    """
    Tokenizes texts from ProcessedRecord and aligns labels with tokens using the B-I-O scheme,
    handling texts longer than max_length by splitting them into overlapping segments based on tokens.

    Args:
        data (List[ProcessedRecord]): List of ProcessedRecord objects.
        tokenizer (BertTokenizerFast): The BERT tokenizer.
        max_length (int): Maximum sequence length for tokenization and padding.

    Returns:
        dict: Contains padded input_ids, attention_masks, and labels tensors.
    """
    input_ids_list = []
    attention_masks_list = []
    labels_list = []

    global label_map
    overlap_tokens = TOKEN_OVERLAP

    for record in data:
        text = record.text_record
        mask_info = record.mask_info # This is a list of MaskInfo objects

        # Tokenize the entire text first
        encoded_inputs_full = tokenizer(
            text,
            # Do not pad or truncate here, we'll handle splitting manually
            padding="do_not_pad",
            truncation=False,
            return_tensors="pt",
            return_offsets_mapping=True, # Get character offsets for token-label alignment
            add_special_tokens=False # Do not add special tokens yet
        )

        full_input_ids = encoded_inputs_full['input_ids'][0].tolist()
        full_offset_mapping = encoded_inputs_full['offset_mapping'][0].tolist()
        full_attention_mask = encoded_inputs_full['attention_mask'][0].tolist() # Get full attention mask

        full_sequence_labels = [label_map['O']] * len(full_input_ids)

        for mask in mask_info:
            char_start = mask.start
            char_end = mask.end
            entity_type = mask.label
            is_first_token_of_entity = True

            for token_idx in range(len(full_input_ids)):
                token_char_start, token_char_end = full_offset_mapping[token_idx]

                # Check for overlap between token's character span and the entity's character span
                token_starts_within_entity = token_char_start >= char_start and token_char_start < char_end
                entity_starts_within_token = char_start >= token_char_start and char_start < token_char_end

                if token_starts_within_entity or entity_starts_within_token:
                    if is_first_token_of_entity:
                        b_label = 'B-' + entity_type
                        if b_label in label_map:
                            full_sequence_labels[token_idx] = label_map[b_label]
                            is_first_token_of_entity = False
                        else:
                            full_sequence_labels[token_idx] = label_map['O']
                            is_first_token_of_entity = False
                            print(f"Warning: Label '{b_label}' not found in label_map for entity type '{entity_type}'. Assigning 'O'.")
                    else:
                        i_label = 'I-' + entity_type
                        if i_label in label_map:
                            full_sequence_labels[token_idx] = label_map[i_label]
                        else:
                            full_sequence_labels[token_idx] = label_map['O']
                            print(f"Warning: Label '{i_label}' not found in label_map for entity type '{entity_type}'. Assigning 'O'.")


        # Split into overlapping segments based on tokens
        token_segments = []
        label_segments = []
        attention_mask_segments = []


        start_token = 0
        while start_token < len(full_input_ids):
            end_token = start_token + max_length

            # Ensure the segment does not exceed the full text token length
            current_segment_input_ids = full_input_ids[start_token:end_token]
            current_segment_labels = full_sequence_labels[start_token:end_token]
            current_segment_attention_mask = full_attention_mask[start_token:end_token]

            # Add special tokens ([CLS] and [SEP]) and pad to max_length
            # The tokenizer handles adding special tokens and padding correctly when called on a list of ids
            # However, for label alignment, we need to manage this manually or adjust after tokenization.
            # A simpler approach here is to pad the segments and labels manually.

            # Pad the segment and labels
            padding_length = max_length - len(current_segment_input_ids)
            padded_input_ids = current_segment_input_ids + [tokenizer.pad_token_id] * padding_length
            padded_labels = current_segment_labels + [label_map['O']] * padding_length # Pad with O labels
            padded_attention_mask = current_segment_attention_mask + [0] * padding_length # Pad attention mask with 0

            # Prepend CLS and append SEP
            # Note: This simplifies the token indices for labels.
            # A more robust approach would involve aligning labels with the tokenizer's output.
            # For minimal changes, we'll add special tokens manually and adjust labels.
            padded_input_ids = [tokenizer.cls_token_id] + padded_input_ids[:-1] # Replace the last padding token with SEP
            padded_labels = [label_map['O']] + padded_labels[:-1] # CLS label is O
            padded_attention_mask = [1] + padded_attention_mask[:-1] # CLS attention is 1

            padded_input_ids[-1] = tokenizer.sep_token_id # Set the last token to SEP
            padded_labels[-1] = label_map['O'] # SEP label is O
            padded_attention_mask[-1] = 1 # SEP attention is 1


            token_segments.append(padded_input_ids)
            label_segments.append(padded_labels)
            attention_mask_segments.append(padded_attention_mask)


            # Move start_token for the next segment, considering overlap
            if end_token >= len(full_input_ids):
                break # Reached the end of the full text
            start_token += max_length - overlap_tokens


        # Append all segments for this record to the main lists
        input_ids_list.extend(token_segments)
        labels_list.extend(label_segments)
        attention_masks_list.extend(attention_mask_segments)


    # Convert lists of lists to PyTorch tensors
    # Ensure all segments have the same length (max_length) before stacking
    input_ids_list = [torch.tensor(ids, dtype=torch.long) for ids in input_ids_list]
    attention_masks_list = [torch.tensor(mask, dtype=torch.long) for mask in attention_masks_list]
    labels_list = [torch.tensor(lbls, dtype=torch.long) for lbls in labels_list]


    return {
        'input_ids': torch.stack(input_ids_list),
        'attention_masks': torch.stack(attention_masks_list),
        'labels': torch.stack(labels_list)
    }