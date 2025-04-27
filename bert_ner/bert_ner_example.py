from transformers import BertTokenizer, BertForTokenClassification, BertTokenizerFast
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np

# --- Configuration ---
MODEL_NAME = 'bert-base-uncased'
MAX_LENGTH = 128 # Max sequence length for tokenization and padding
BATCH_SIZE = 8   # Increased batch size for efficiency
NUM_EPOCHS = 50   # Train for more epochs
LEARNING_RATE = 2e-5

# --- Define Labels and Mapping ---
# O: Outside of a named entity
# B-NAME: Beginning of a person's name
# I-NAME: Inside a person's name
label_map = {'O': 0, 'B-NAME': 1, 'I-NAME': 2}
id_to_label = {0: 'O', 1: 'B-NAME', 2: 'I-NAME'}
num_labels = len(label_map)

# --- Load Model and Tokenizer ---
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
# Initialize model with the correct number of labels for token classification
model = BertForTokenClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)

# --- Prepare Data (English Examples with Entity Spans) ---
# Data format: (text, list_of_entities)
# Each entity: (start_char_index, end_char_index, entity_type) - end_char_index is exclusive
train_data = [
    ("Dr. John Doe will see the patient.", [(4, 12, "NAME")]), # "John Doe"
    ("The appointment is with Jane Smith tomorrow.", [(28, 38, "NAME")]), # "Jane Smith"
    ("Report for Patient ID 12345.", []), # No NAME entities
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
]

test_data = [
    "The patient name is Michael Clark.", # Michael Clark
    "Follow up with Dr. Susan Davis.", # Susan Davis
    "General health information.", # No name
    "Check the records for Alice.", # Alice
    "Meeting with Dr. Williams and Dr. Jones.", # Williams, Jones
    "Report filed by Dr. Evans.", # Evans
    "The primary contact is Mr. Thomas Green.", # Thomas Green
]

# --- Data Processing Function ---
def process_data(data, tokenizer, label_map, max_length):
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
                if token_char_start == 0 and token_char_end == 0 and \
                        encoded_inputs['input_ids'][0][token_idx].item() in \
                        [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]:
                    continue

                # Check if the token's character span overlaps with the entity's character span
                # A common and effective check: Does the token's start character fall within the entity span?
                # This correctly handles subwords at the beginning of an entity.
                if token_char_start >= char_start and token_char_start < char_end:
                    # This token is part of the current entity
                    if is_first_token_of_entity:
                        # This is the first token corresponding to this entity span, assign B- label
                        sequence_labels[token_idx] = label_map['B-' + entity_type]
                        is_first_token_of_entity = False # Subsequent tokens for this entity get I-
                    else:
                        # This token is part of an entity that has already started, assign I- label
                        sequence_labels[token_idx] = label_map['I-' + entity_type]

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

# --- Testing (Prediction) ---
print("\n--- Testing (Prediction) ---")
model.eval() # Set the model to evaluation mode

with torch.no_grad(): # Disable gradient calculation for inference
    for text in test_data:
        # Tokenize the test text
        inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=MAX_LENGTH).to(device)

        # Perform a forward pass without labels
        outputs = model(**inputs)

        # Get the predicted label with the highest score for each token
        predictions = torch.argmax(outputs.logits, dim=-1)

        # Convert token IDs back to tokens for display
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

        # Convert predicted label IDs back to label strings
        predicted_labels = [id_to_label[p.item()] for p in predictions[0]]

        print(f"Text: {text}")
        print("Predicted Labels:")
        # Print each token and its predicted label
        for token, label in zip(tokens, predicted_labels):
            # Skip padding tokens for cleaner output
            if token == tokenizer.pad_token:
                continue
            print(f"  {token} --> {label}")
        print("-" * 30)