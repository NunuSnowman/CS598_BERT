from dataclasses import dataclass

from transformers import BertTokenizer, BertForTokenClassification, BertTokenizerFast
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
from typing import List, Tuple, Optional

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

@dataclass
class Entity:
    """Represents a named entity with start and end character indices and a label."""
    start: int
    end: int
    label: str
    original_text: Optional[str] = None
    original_label: Optional[str] = None

TextWithEntities = Tuple[str, List[Entity]]
LabeledData = List[TextWithEntities]


# --- Prepare Data (English Examples with Entity Spans) ---
# Data format: (text, list_of_entities)
# Each entity: (start_char_index, end_char_index, entity_type) - end_char_index is exclusive
train_data: LabeledData = [
    ("Dr. John Doe will see the patient.", [Entity(start=4, end=12, label="NAME")]), # "John Doe"
    ("The appointment is with Jane Smith tomorrow.", [Entity(start=28, end=38, label="NAME")]), # "Jane Smith"
    ("Report for Patient ID 12345.", []), # No entities
    ("Contact Dr. Emily White at the clinic.", [Entity(start=12, end=23, label="NAME")]), # "Emily White"
    ("This is a normal sentence.", []),
    ("Please refer to the notes from Dr. Brown.", [Entity(start=33, end=38, label="NAME")]), # "Brown"
    ("Mr. David Lee is the attending physician.", [Entity(start=4, end=13, label="NAME")]), # "David Lee"
    ("No specific name mentioned here.", []),
    ("The nurse, Mary Johnson, provided the care.", [Entity(start=11, end=23, label="NAME")]), # "Mary Johnson"
    ("Patient admitted by Dr. Robert Green.", [Entity(start=24, end=36, label="NAME")]), # "Robert Green"
    ("Consultation with Dr. Alice Williams.", [Entity(start=21, end=36, label="NAME")]), # "Alice Williams"
    ("Patient transferred to Dr. Michael Brown's care.", [Entity(start=25, end=38, label="NAME")]), # "Michael Brown"
    ("Examined by Dr. Sarah Davis.", [Entity(start=16, end=27, label="NAME")]), # "Sarah Davis"

    # Adding LOCATION and DATE examples
    ("The meeting is in London.", [Entity(start=19, end=25, label="LOCATION")]), # "London"
    ("We will arrive on Monday.", [Entity(start=18, end=24, label="DATE")]), # "Monday"
    ("Visited Paris, France last year.", [Entity(start=8, end=13, label="LOCATION"), Entity(start=15, end=21, label="LOCATION")]), # "Paris", "France"
    ("The deadline is December 31, 2024.", [Entity(start=17, end=33, label="DATE")]), # "December 31, 2024"
    ("He lives in New York City.", [Entity(start=12, end=25, label="LOCATION")]), # "New York City"
    ("The event is scheduled for next Tuesday.", [Entity(start=28, end=39, label="DATE")]), # "next Tuesday"
    ("Patient report from London, UK dated January 5, 2023.", [Entity(start=20, end=26, label="LOCATION"), Entity(start=28, end=31, label="LOCATION"), Entity(start=38, end=51, label="DATE")]), # "London", "UK", "January 5, 2023"
    ("Travel to Rome on the 15th.", [Entity(start=9, end=13, label="LOCATION"), Entity(start=21, end=27, label="DATE")]), # "Rome", "15th"
    ("Appointment is on Friday afternoon.", [Entity(start=18, end=24, label="DATE")]), # "Friday"
    ("Location: Seattle. Date: 2025-04-27.", [Entity(start=10, end=17, label="LOCATION"), Entity(start=24, end=34, label="DATE")]), # "Seattle", "2025-04-27"
    ("Dr. Chen is relocating to Boston next month.", [Entity(start=28, end=34, label="LOCATION"), Entity(start=35, end=45, label="DATE")]), # "Boston", "next month"
    ("Meeting with Mr. Adams in Berlin on Wednesday.", [Entity(start=20, end=25, label="NAME"), Entity(start=30, end=36, label="LOCATION"), Entity(start=40, end=49, label="DATE")]), # "Adams", "Berlin", "Wednesday"
]

test_data: LabeledData = [
    ("The patient name is Michael Clark.", [Entity(start=24, end=35, label="NAME")]), # "Michael Clark"
    ("Follow up with Dr. Susan Davis.", [Entity(start=20, end=31, label="NAME")]), # "Susan Davis"
    ("General health information.", []), # No entity
    ("Check the records for Alice.", [Entity(start=24, end=29, label="NAME")]), # "Alice"
    ("Meeting with Dr. Williams and Dr. Jones.", [Entity(start=19, end=28, label="NAME"), Entity(start=37, end=42, label="NAME")]), # "Williams", "Jones"
    ("Report filed by Dr. Evans.", [Entity(start=21, end=26, label="NAME")]), # "Evans"
    ("The primary contact is Mr. Thomas Green.", [Entity(start=28, end=39, label="NAME")]), # "Thomas Green"

    # Test cases with LOCATION and DATE
    ("Visited Washington D.C. last week.", [Entity(start=9, end=24, label="LOCATION"), Entity(start=25, end=34, label="DATE")]), # "Washington D.C.", "last week"
    ("Conference held in Tokyo on March 10.", [Entity(start=21, end=26, label="LOCATION"), Entity(start=30, end=39, label="DATE")]), # "Tokyo", "March 10"
    ("Dr. Wilson will be in Chicago until Friday.", [Entity(start=4, end=10, label="NAME"), Entity(start=25, end=32, label="LOCATION"), Entity(start=39, end=45, label="DATE")]), # "Wilson", "Chicago", "Friday"
    ("The report is due by 2024-12-01.", [Entity(start=21, end=31, label="DATE")]), # "2024-12-01"
    ("Sent details to Mr. White in Sydney.", [Entity(start=24, end=29, label="NAME"), Entity(start=33, end=39, label="LOCATION")]), # "White", "Sydney"
    ("The event is scheduled for tomorrow in London.", [Entity(start=28, end=35, label="DATE"), Entity(start=39, end=45, label="LOCATION")]), # "tomorrow", "London"
    ("Patient arrived from New York on July 4th.", [Entity(start=22, end=30, label="LOCATION"), Entity(start=34, end=43, label="DATE")]), # "New York", "July 4th"
]

def process_data(data: LabeledData, tokenizer, label_map: dict[str:int], max_length: int):
    """
    Tokenizes texts and aligns labels with tokens using the B-I-O scheme.

    Args:
        data (list): List of (text, list_of_entities) tuples.
        tokenizer: The BERT tokenizer.
        label_map (dict): Mapping from label strings to integers.
        max_length (int): Maximum sequence length for padding and truncation.

    Returns:
        dict: Contains padded input_ids, attention_masks, and labels tensors.
    """
    input_ids = []
    attention_masks = []
    labels = []

    for text, entities in data:
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

        # Iterate through each defined entity in the original text
        for entity in entities:
            char_start = entity.start
            char_end = entity.end
            entity_type = entity.label
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
                        sequence_labels[token_idx] = label_map['B-' + entity_type]
                        is_first_token_of_entity = False # Subsequent tokens for this entity get I-
                    else:
                        # This token is part of an entity that has already started, assign I- label
                        sequence_labels[token_idx] = label_map['I-' + entity_type]
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