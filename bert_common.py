from dataclasses import dataclass

from transformers import BertTokenizer, BertForTokenClassification, BertTokenizerFast
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
from typing import List, Tuple, Optional

from common import ProcessedRecord, MaskInfo

# --- Configuration ---
MAX_LENGTH = 128 # Max sequence length for tokenization and padding
TOKEN_OVERLAP = 32
BATCH_SIZE = 8   # Increased batch size for efficiency
NUM_EPOCHS = 5   # Train for more epochs
LEARNING_RATE = 5e-5
WARM_UP_RATIO = 0.4
IGNORE_INDEX = -100 # Value to use for ignoring loss on specific tokens
SAVE_DIRECTORY = "./tmp/saved_models"
SAVE_MODEL_EVERY_N_EPOCH = 0
bert_print_debug_log = False
model_name = 'bert-base-uncased'
use_multiple_classes = True
multi_class_label_map = {'O': 0, 'B-NAME': 1, 'I-NAME': 2,
                         'B-LOCATION': 3, 'I-LOCATION': 4,
                         'B-DATE': 5, 'I-DATE': 6, 'B-CONTACT': 7, 'I-CONTACT': 8, }
label_map = multi_class_label_map

id_to_label = {v: k for k, v in label_map.items()} # Create reverse mapping
num_labels = len(label_map)

def set_classify_type(use_multi_class: bool):
    global use_multiple_classes
    global label_map
    global id_to_label
    global num_labels
    use_multiple_classes = use_multi_class
    if use_multi_class:
        label_map = multi_class_label_map
    else:
        label_map = {'O': 0, 'B-PHI': 1, 'I-PHI': 2}
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

def process_data_label(data: List[ProcessedRecord], tokenizer: BertTokenizerFast, max_length: int):
    """
    Tokenizes texts from ProcessedRecord and aligns labels with tokens using the B-I-O scheme,
    handling texts longer than max_length by splitting them into overlapping segments based on tokens.
    Also modifies labels so that labels for sub-tokens and special tokens are set to -100
    for ignoring them in loss calculation, using the WordPiece prefix to identify sub-tokens.

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

    # Get the wordpiece prefix from the tokenizer (e.g., '##' for BERT)
    wordpieces_prefix = "##"

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

        # Get the list of tokens from their IDs to check for prefixes
        full_tokens = tokenizer.convert_ids_to_tokens(full_input_ids)

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
                            if b_label != 'B-O' and bert_print_debug_log:
                                print(f"Warning: Label '{b_label}' not found in label_map for entity type '{entity_type}'. Assigning 'O'.")
                    else:
                        i_label = 'I-' + entity_type
                        if i_label in label_map:
                            full_sequence_labels[token_idx] = label_map[i_label]
                        else:
                            full_sequence_labels[token_idx] = label_map['O']
                            if i_label != 'I-O' and bert_print_debug_log:
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
            current_segment_tokens = full_tokens[start_token:end_token]


            segment_labels_for_loss = list(current_segment_labels)

            for token_idx in range(len(current_segment_input_ids)):
                token = current_segment_tokens[token_idx]
                token_id = current_segment_input_ids[token_idx]

                # Check if the token is a special token or a sub-token using the prefix
                # Special tokens include CLS, SEP, PAD, etc.
                # Sub-tokens typically start with the wordpieces_prefix (e.g., '##')
                if token_id in tokenizer.all_special_ids or \
                        token.startswith(wordpieces_prefix):
                    segment_labels_for_loss[token_idx] = IGNORE_INDEX


            # --- Add Special Tokens and Pad Segments ---
            padding_length = max_length - len(current_segment_input_ids)

            # Pad input_ids, attention_mask, and labels for loss
            padded_input_ids = current_segment_input_ids + [tokenizer.pad_token_id] * padding_length
            padded_attention_mask = current_segment_attention_mask + [0] * padding_length
            # Pad labels for loss with IGNORE_INDEX
            padded_labels_for_loss = segment_labels_for_loss + [IGNORE_INDEX] * padding_length

            padded_input_ids = [tokenizer.cls_token_id] + padded_input_ids[:-1] # Replace the last padding token with SEP
            padded_labels_for_loss = [IGNORE_INDEX] + padded_labels_for_loss[:-1] # CLS label is O
            padded_attention_mask = [1] + padded_attention_mask[:-1] # CLS attention is 1

            # [SEP] token at the end
            padded_input_ids[-1] = tokenizer.sep_token_id
            padded_attention_mask[-1] = 1
            # The label for [SEP] is set to IGNORE_INDEX
            padded_labels_for_loss[-1] = IGNORE_INDEX

            # Ensure padding tokens introduced by adding special tokens also have IGNORE_INDEX
            for i in range(len(padded_input_ids)):
                # Check if the token is a padding token by its ID
                if padded_input_ids[i] == tokenizer.pad_token_id:
                    padded_labels_for_loss[i] = IGNORE_INDEX


            token_segments.append(padded_input_ids)
            label_segments.append(padded_labels_for_loss) # Use the modified labels for loss
            attention_mask_segments.append(padded_attention_mask)


            # Move start_token for the next segment, considering overlap
            if end_token >= len(full_input_ids):
                break # Reached the end of the full text
            start_token += max_length - overlap_tokens


        # Append all segments for this record to the main lists
        input_ids_list.extend(token_segments)
        labels_list.extend(label_segments) # Use the modified labels list
        attention_masks_list.extend(attention_mask_segments)


    # Convert lists of lists to PyTorch tensors
    # Ensure all segments have the same length (max_length) before stacking
    input_ids_list = [torch.tensor(ids, dtype=torch.long) for ids in input_ids_list]
    attention_masks_list = [torch.tensor(mask, dtype=torch.long) for mask in attention_masks_list]
    labels_list = [torch.tensor(lbls, dtype=torch.long) for lbls in labels_list]


    return {
        'input_ids': torch.stack(input_ids_list),
        'attention_masks': torch.stack(attention_masks_list),
        'labels': torch.stack(labels_list) # Return the modified labels tensor
    }