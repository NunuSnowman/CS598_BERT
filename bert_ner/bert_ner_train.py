from bert_ner.bert_common import *

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
model = BertForTokenClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)


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

if __name__ == "__main__":
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
    print(f"\nSaving model to {SAVE_DIRECTORY}...")
    model.save_pretrained(SAVE_DIRECTORY)
    print("Model saved.")
