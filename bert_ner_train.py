from bert_common import *

def train_model(data: [ProcessedRecord], tokenizer, model, save_directory=SAVE_DIRECTORY):
    print("Processing training data...")
    train_processed = process_data_label(data, tokenizer, MAX_LENGTH)

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
    print(f"\nSaving model to {save_directory}...")
    import os
    os.makedirs(save_directory, exist_ok=True)
    model.save_pretrained(save_directory)
    print("Model saved.")

if __name__ == "__main__":
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    model = BertForTokenClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)
    train_model(test_data, tokenizer, model, save_directory=SAVE_DIRECTORY)

