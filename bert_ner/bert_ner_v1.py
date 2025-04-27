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

Entity = Tuple[int, int, str]
TextWithEntities = Tuple[str, List[Entity]]
LabeledData = List[TextWithEntities]


label_map = {'O': 0, 'B-NAME': 1, 'I-NAME': 2,
             'B-LOCATION': 3, 'I-LOCATION': 4,
             'B-DATE': 5, 'I-DATE': 6}
id_to_label = {v: k for k, v in label_map.items()} # Create reverse mapping
num_labels = len(label_map)

# --- Load Model and Tokenizer ---
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
# Initialize model with the correct number of labels for token classification
model = BertForTokenClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)

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

# --- Data Processing Function ---
def process_data(data: LabeledData, tokenizer, label_map, max_length: int):
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
        for char_start, char_end, entity_type in entities:
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

# --- Process Training Data ---
print("Processing training data...")
train_processed = process_data(train_data, tokenizer, label_map, MAX_LENGTH)

# --- Create TensorDataset and DataLoader ---
train_dataset = TensorDataset(
    train_processed['input_ids'],
    train_processed['attention_masks'],
    train_processed['labels']
)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- Training Setup ---
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
# Check for GPU and move the model and data to the appropriate device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

# --- Training Loop ---
print(f"\nTraining on device: {device}")
model.train() # Set the model to training mode
for epoch in range(NUM_EPOCHS):
    total_loss = 0
    # Iterate over batches from the DataLoader
    for step, batch in enumerate(train_dataloader):
        # Move batch tensors to the chosen device
        batch = tuple(t.to(device) for t in batch)
        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'labels': batch[2]} # Provide labels for loss calculation

        # Zero out any previously calculated gradients
        optimizer.zero_grad()

        # Perform a forward pass
        outputs = model(**inputs)

        # Get the loss (calculated internally by BertForTokenClassification when labels are provided)
        loss = outputs.loss

        # Perform a backward pass to calculate gradients
        loss.backward()

        # Update model parameters
        optimizer.step()

        total_loss += loss.item()

        # Print loss periodically
        if (step + 1) % 10 == 0 or step == 0: # Print first step and every 10 steps
            print(f"  Epoch {epoch+1}, Step {step+1}, Loss: {loss.item():.4f}")


    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch+1} Complete, Average Loss: {avg_loss:.4f}")

# --- Testing (Evaluation with Labels) ---
print("\n--- Testing (Evaluation) ---")
model.eval() # Set the model to evaluation mode

# --- Process Test Data ---
print("Processing test data...")
# Use the labeled test data here
test_processed = process_data(test_data, tokenizer, label_map, MAX_LENGTH)

# --- Create Test TensorDataset and DataLoader ---
test_dataset = TensorDataset(
    test_processed['input_ids'],
    test_processed['attention_masks'],
    test_processed['labels'] # Now includes true labels
)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE) # No need to shuffle for testing

# Lists to store all true and predicted labels for evaluation
all_true_labels = []
all_predicted_labels = []

with torch.no_grad(): # Disable gradient calculation for inference
    # Iterate over batches from the Test DataLoader
    for step, batch in enumerate(test_dataloader):
        # Move batch tensors to the chosen device
        batch = tuple(t.to(device) for t in batch)
        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1]}
        true_labels = batch[2] # Get the true labels for this batch

        # Perform a forward pass
        outputs = model(**inputs)

        # Get the predicted label with the highest score for each token
        predictions = torch.argmax(outputs.logits, dim=-1)

        # --- Collect true and predicted labels for evaluation ---
        # We need to flatten the tensors and filter out special tokens/padding
        # The attention mask is useful here to identify actual tokens
        attention_mask = inputs['attention_mask']

        # Iterate through each sequence in the batch
        for i in range(predictions.shape[0]):
            # Filter out predictions and true labels corresponding to padding (mask == 0)
            # and potentially special tokens if you want to exclude them from metrics
            # A common approach is to use the attention mask to keep only real tokens
            # We also typically ignore labels for special tokens themselves ('O' or -100)
            # If your process_data assigns 'O' (0) to special tokens, filtering by mask is sufficient.
            # If you use ignore_index=-100 in your loss, you might filter based on that as well.

            seq_predictions = predictions[i][attention_mask[i] == 1].cpu().numpy()
            seq_true_labels = true_labels[i][attention_mask[i] == 1].cpu().numpy()

            # Extend the global lists
            all_predicted_labels.extend(seq_predictions)
            all_true_labels.extend(seq_true_labels)


# --- Calculate and Print Evaluation Metrics ---
# This part requires a library like scikit-learn
try:
    from sklearn.metrics import classification_report

    # Map integer IDs back to label strings for the report
    # Ensure you have the correct mapping and potentially filter out the 'O' label
    # from the report if you only want metrics for the named entities.
    # The classification_report can handle this if you provide the target_names
    # and potentially specify labels to include.

    # Create a list of label names in the correct order based on label_map
    # Filter out 'O' if you want metrics only for entity types (B-*, I-*)
    label_names = [id_to_label[i] for i in range(num_labels)]
    # Example: If you want to exclude 'O' (label 0) from the report:
    # target_names_for_report = [name for name in label_names if name != 'O']
    # labels_to_include = [label_map[name] for name in target_names_for_report]

    print("\nClassification Report:")
    print(classification_report(all_true_labels, all_predicted_labels,
                                labels=[i for i in range(num_labels) if id_to_label[i] != 'O'], # Exclude 'O' from metrics
                                target_names=[id_to_label[i] for i in range(num_labels) if id_to_label[i] != 'O'],
                                zero_division=0)) # Handle cases where a label has no true/predicted instances

except ImportError:
    print("\nScikit-learn not found. Install it (`pip install scikit-learn`) to see evaluation metrics.")
    print("Skipping detailed classification report.")

# Optional: Print a few examples with predicted vs. true labels
print("\n--- Example Predictions vs. True Labels ---")