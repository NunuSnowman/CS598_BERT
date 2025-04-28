from bert_ner.bert_common import *
from bert_ner.bert_ner_v1 import process_data

device = torch.device("cpu")
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
model = BertForTokenClassification.from_pretrained(SAVE_DIRECTORY, num_labels=num_labels)

print("\n--- Testing (Evaluation) ---")
model.eval() # Set the model to evaluation mode

print("Processing test data...")

test_processed = process_data(test_data, tokenizer, label_map, MAX_LENGTH)

test_dataset = TensorDataset(
    test_processed['input_ids'],
    test_processed['attention_masks'],
    test_processed['labels'] # Now includes true labels
)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

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

        attention_mask = inputs['attention_mask']

        for i in range(predictions.shape[0]):
            seq_predictions = predictions[i][attention_mask[i] == 1].cpu().numpy()
            seq_true_labels = true_labels[i][attention_mask[i] == 1].cpu().numpy()

            # Extend the global lists
            all_predicted_labels.extend(seq_predictions)
            all_true_labels.extend(seq_true_labels)


try:
    from sklearn.metrics import classification_report
    label_names = [id_to_label[i] for i in range(num_labels)]

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