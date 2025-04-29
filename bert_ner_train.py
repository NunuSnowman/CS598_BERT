import torch
from transformers import BertTokenizerFast, BertForTokenClassification, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset
import os
from typing import List, Tuple, Optional, Callable

# Assuming these imports are available from bert_common.py
from bert_common import ProcessedRecord, MaskInfo, MODEL_NAME, MAX_LENGTH, TOKEN_OVERLAP, BATCH_SIZE, NUM_EPOCHS, \
    LEARNING_RATE, SAVE_DIRECTORY, label_map, id_to_label, num_labels, create_processed_record, \
    process_data_label

# Import the evaluation function from bert_ner_test.py
from bert_ner_test import evaluate_model  # Assuming evaluate_model is in bert_ner_test.py


def train_model(
        data: [ProcessedRecord],
        tokenizer: BertTokenizerFast,
        model: BertForTokenClassification,
        save_directory: str = SAVE_DIRECTORY
):
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

    total_steps = len(train_dataloader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),  # 10% warmup
        num_training_steps=total_steps
    )
    # --- Training Loop ---
    print(f"\nTraining on device: {device}")
    model.train()  # Set the model to training mode
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        # Iterate over batches from the DataLoader
        for step, batch in enumerate(train_dataloader):
            # Move batch tensors to the chosen device
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[2]}  # Provide labels for loss calculation

            # Zero out any previously calculated gradients
            optimizer.zero_grad()

            # Perform a forward pass
            outputs = model(**inputs)

            # Get the loss (calculated internally by BertForTokenClassification when labels are provided)
            loss = outputs.loss

            # Perform a backward pass to calculate gradients
            loss.backward()

            optimizer.step()

            scheduler.step()

            total_loss += loss.item()

            if (step + 1) % 10 == 0 or step == 0:  # Print first step and every 10 steps
                print(f"  Epoch {epoch + 1}, Step {step + 1}, Loss: {loss.item():.4f}")


        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1} Complete, Average Loss: {avg_loss:.4f}")
    print(f"\nSaving model to {save_directory}...")
    import os
    os.makedirs(save_directory, exist_ok=True)
    model.save_pretrained(save_directory)
    print("Model saved.")


