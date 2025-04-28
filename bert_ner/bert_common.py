from transformers import BertTokenizer, BertForTokenClassification, BertTokenizerFast
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
from typing import List, Tuple


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

Entity = Tuple[int, int, str]
TextWithEntities = Tuple[str, List[Entity]]
LabeledData = List[TextWithEntities]


# --- Prepare Data (English Examples with Entity Spans) ---
# Data format: (text, list_of_entities)
# Each entity: (start_char_index, end_char_index, entity_type) - end_char_index is exclusive
train_data: LabeledData = [
    ("Dr. John Doe will see the patient.", [(4, 12, "NAME")]), # "John Doe"
    ("The appointment is with Jane Smith tomorrow.", [(28, 38, "NAME")]), # "Jane Smith"
    ("Report for Patient ID 12345.", []), # No entities
    ("Contact Dr. Emily White at the clinic.", [(12, 23, "NAME")]), # "Emily White"
    ("This is a normal sentence.", []),
    ("Please refer to the notes from Dr. Brown.", [(33, 38, "NAME")]), # "Brown"
    ("Mr. David Lee is the attending physician.", [(4, 13, "NAME")]), # "David Lee"
    ("No specific name mentioned here.", []),
    ("The nurse, Mary Johnson, provided the care.", [(11, 23, "NAME")]), # "Mary Johnson"
    ("Patient admitted by Dr. Robert Green.", [(24, 36, "NAME")]), # "Robert Green"
    ("Consultation with Dr. Alice Williams.", [(21, 36, "NAME")]), # "Alice Williams"
    ("Patient transferred to Dr. Michael Brown's care.", [(25, 38, "NAME")]), # "Michael Brown"
    ("Examined by Dr. Sarah Davis.", [(16, 27, "NAME")]), # "Sarah Davis"

    # Adding LOCATION and DATE examples
    ("The meeting is in London.", [(19, 25, "LOCATION")]), # "London"
    ("We will arrive on Monday.", [(18, 24, "DATE")]), # "Monday"
    ("Visited Paris, France last year.", [(8, 13, "LOCATION"), (15, 21, "LOCATION")]), # "Paris", "France"
    ("The deadline is December 31, 2024.", [(17, 33, "DATE")]), # "December 31, 2024"
    ("He lives in New York City.", [(12, 25, "LOCATION")]), # "New York City"
    ("The event is scheduled for next Tuesday.", [(28, 39, "DATE")]), # "next Tuesday"
    ("Patient report from London, UK dated January 5, 2023.", [(20, 26, "LOCATION"), (28, 31, "LOCATION"), (38, 51, "DATE")]), # "London", "UK", "January 5, 2023"
    ("Travel to Rome on the 15th.", [(9, 13, "LOCATION"), (21, 27, "DATE")]), # "Rome", "15th"
    ("Appointment is on Friday afternoon.", [(18, 24, "DATE")]), # "Friday"
    ("Location: Seattle. Date: 2025-04-27.", [(10, 17, "LOCATION"), (24, 34, "DATE")]), # "Seattle", "2025-04-27"
    ("Dr. Chen is relocating to Boston next month.", [(28, 34, "LOCATION"), (35, 45, "DATE")]), # "Boston", "next month"
    ("Meeting with Mr. Adams in Berlin on Wednesday.", [(20, 25, "NAME"), (30, 36, "LOCATION"), (40, 49, "DATE")]), # "Adams", "Berlin", "Wednesday"
]

test_data: LabeledData = [
    ("The patient name is Michael Clark.", [(24, 35, "NAME")]), # "Michael Clark"
    ("Follow up with Dr. Susan Davis.", [(20, 31, "NAME")]), # "Susan Davis"
    ("General health information.", []), # No entity
    ("Check the records for Alice.", [(24, 29, "NAME")]), # "Alice"
    ("Meeting with Dr. Williams and Dr. Jones.", [(19, 28, "NAME"), (37, 42, "NAME")]), # "Williams", "Jones"
    ("Report filed by Dr. Evans.", [(21, 26, "NAME")]), # "Evans"
    ("The primary contact is Mr. Thomas Green.", [(28, 39, "NAME")]), # "Thomas Green"

    # Test cases with LOCATION and DATE
    ("Visited Washington D.C. last week.", [(9, 24, "LOCATION"), (25, 34, "DATE")]), # "Washington D.C.", "last week"
    ("Conference held in Tokyo on March 10.", [(21, 26, "LOCATION"), (30, 39, "DATE")]), # "Tokyo", "March 10"
    ("Dr. Wilson will be in Chicago until Friday.", [(4, 10, "NAME"), (25, 32, "LOCATION"), (39, 45, "DATE")]), # "Wilson", "Chicago", "Friday"
    ("The report is due by 2024-12-01.", [(21, 31, "DATE")]), # "2024-12-01"
    ("Sent details to Mr. White in Sydney.", [(24, 29, "NAME"), (33, 39, "LOCATION")]), # "White", "Sydney"
    ("The event is scheduled for tomorrow in London.", [(28, 35, "DATE"), (39, 45, "LOCATION")]), # "tomorrow", "London"
    ("Patient arrived from New York on July 4th.", [(22, 30, "LOCATION"), (34, 43, "DATE")]), # "New York", "July 4th"
]
