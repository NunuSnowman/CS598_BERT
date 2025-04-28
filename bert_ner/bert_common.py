

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
BATCH_SIZE = 8   # Increased batch size for efficiency
NUM_EPOCHS = 50   # Train for more epochs
LEARNING_RATE = 2e-5
SAVE_DIRECTORY = "./saved_ner_model"

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
    Tokenizes texts from ProcessedRecord and aligns labels with tokens using the B-I-O scheme.

    Args:
        data (List[ProcessedRecord]): List of ProcessedRecord objects.
        tokenizer (BertTokenizerFast): The BERT tokenizer.
        max_length (int): Maximum sequence length for padding and truncation.

    Returns:
        dict: Contains padded input_ids, attention_masks, and labels tensors.
    """
    input_ids = []
    attention_masks = []
    labels = []

    # Access the globally defined label_map
    global label_map

    for record in data:
        text = record.text_record
        mask_info = record.mask_info # This is a list of MaskInfo objects

        # Tokenize the text
        encoded_inputs = tokenizer(
            text,
            padding="max_length", # Pad to max_length
            truncation=True,      # Truncate to max_length
            max_length=max_length,
            return_tensors="pt",  # Return PyTorch tensors
            return_offsets_mapping=True # Get character offsets for token-label alignment
        )

        # Get the character offsets for each token in the sequence
        offset_mapping = encoded_inputs['offset_mapping'][0].tolist()

        # Initialize token-level labels for this sequence with 'O' (label 0)
        sequence_labels = [label_map['O']] * max_length # Initialize with 'O' up to max_length

        # Iterate through each defined mask_info in the current record
        for mask in mask_info:
            char_start = mask.start
            char_end = mask.end
            entity_type = mask.label
            # Keep track if we've found the first token of this entity
            is_first_token_of_entity = True

            # Iterate through tokens to find which ones correspond to the current entity span
            for token_idx in range(len(encoded_inputs['input_ids'][0])):
                token_char_start, token_char_end = offset_mapping[token_idx]

                # Skip special tokens ([CLS], [SEP], [PAD]) and their (0,0) offsets
                # Check token ID to be sure it's a special token
                # A more robust check for special tokens often involves looking at the token ID
                special_token_ids = set([tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id])
                if encoded_inputs['input_ids'][0][token_idx].item() in special_token_ids:
                    # If it's a special token, its label should always be O (or ignored in loss)
                    # Our initialization already sets it to O, so we just continue
                    continue

                # Check if the token's character span overlaps with the entity's character span
                # A common and effective check: Does the token's start character fall within the entity span?
                # This correctly handles subwords at the beginning of an entity.
                # We also need to handle cases where the token's end character is within the entity span,
                # or where the entity spans across the token. A simplified check is below:
                # If the token starts within the entity OR the entity starts within the token
                token_starts_within_entity = token_char_start >= char_start and token_char_start < char_end
                entity_starts_within_token = char_start >= token_char_start and char_start < token_char_end

                if token_starts_within_entity or entity_starts_within_token:
                    # This token is part of the current entity
                    if is_first_token_of_entity:
                        # This is the first token corresponding to this entity span, assign B- label
                        # Ensure the label exists in the label_map (e.g., "B-" + "NAME")
                        b_label = 'B-' + entity_type
                        if b_label in label_map:
                            sequence_labels[token_idx] = label_map[b_label]
                            is_first_token_of_entity = False # Subsequent tokens for this entity get I-
                        else:
                            # Handle cases where the B- label is not in the map, maybe default to 'O'
                            sequence_labels[token_idx] = label_map['O']
                            is_first_token_of_entity = False # Prevent I- if B- wasn't found
                            # Optional: Add a warning or log here if a label is unexpected
                            print(f"Warning: Label '{b_label}' not found in label_map for entity type '{entity_type}'. Assigning 'O'.")

                    else:
                        # This token is part of an entity that has already started, assign I- label
                        # Ensure the label exists in the label_map (e.g., "I-" + "NAME")
                        i_label = 'I-' + entity_type
                        if i_label in label_map:
                            sequence_labels[token_idx] = label_map[i_label]
                        else:
                            # Handle cases where the I- label is not in the map, maybe default to 'O'
                            sequence_labels[token_idx] = label_map['O']
                            # Optional: Add a warning or log here if a label is unexpected
                            print(f"Warning: Label '{i_label}' not found in label_map for entity type '{entity_type}'. Assigning 'O'.")

                # Note: The offset mapping can be tricky with BERT's subword tokenization.
                # The logic above is a common approximation. For highly precise alignment,
                # more complex logic might be needed, potentially considering token_char_end.


        # Convert the list of label integers to a PyTorch tensor
        sequence_labels_tensor = torch.tensor(sequence_labels, dtype=torch.long)

        # Append the processed inputs and labels to the lists
        input_ids.append(encoded_inputs['input_ids'][0])
        attention_masks.append(encoded_inputs['attention_mask'][0])
        labels.append(sequence_labels_tensor)

    # Stack the lists of tensors into single tensors for the dataset
    return {
        'input_ids': torch.stack(input_ids),
        'attention_masks': torch.stack(attention_masks),
        'labels': torch.stack(labels)
    }