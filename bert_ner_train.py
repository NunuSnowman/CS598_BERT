import os

import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizerFast, BertForTokenClassification, get_linear_schedule_with_warmup

import bert_common
# Assuming these imports are available from bert_common.py
from bert_common import ProcessedRecord, MAX_LENGTH, BATCH_SIZE, NUM_EPOCHS, \
    LEARNING_RATE, SAVE_DIRECTORY, \
    process_data_label, SAVE_MODEL_EVERY_N_EPOCH
def focal_loss(logits, labels, alpha=1, gamma=2, ignore_index=-100):
    ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction='none', ignore_index=ignore_index)
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    return focal_loss.mean()

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

    total_steps = len(train_dataloader) * bert_common.NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(bert_common.WARM_UP_RATIO * total_steps),  # 10% warmup
        num_training_steps=total_steps
    )
    # --- Training Loop ---
    print(f"\nTraining on device: {device}")
    model.train()  # Set the model to training mode
    for epoch in range(bert_common.NUM_EPOCHS):
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
            if bert_common.use_crossing_entropy_loss:
                loss = outputs.loss
            else:
                # loss = asymmetric_focal_loss(outputs.logits, inputs['labels'])
                loss = focal_loss(outputs.logits, inputs['labels'])

            loss.backward()

            optimizer.step()

            scheduler.step()

            total_loss += loss.item()

            if (step + 1) % 30 == 0 or step == 0:  # Print first step and every 10 steps
                print(f"  Epoch {epoch + 1}, Step {step + 1}, Loss: {loss.item():.4f}")


        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1} Complete, Average Loss: {avg_loss:.4f}")
        if SAVE_MODEL_EVERY_N_EPOCH != 0 and (epoch + 1) % SAVE_MODEL_EVERY_N_EPOCH == 0 and epoch+1 != bert_common.NUM_EPOCHS :
            epoch_save_directory = f"{save_directory}_epoch_{epoch + 1}"
            os.makedirs(epoch_save_directory, exist_ok=True)
            model.save_pretrained(epoch_save_directory)
            print(f"Model checkpoint saved to {epoch_save_directory}")

    print(f"\nSaving model to {save_directory}...")
    os.makedirs(save_directory, exist_ok=True)
    model.save_pretrained(save_directory)
    print("Model saved.")


