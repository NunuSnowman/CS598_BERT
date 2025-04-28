from bert_ner.bert_common import *
from bert_ner.bert_ner_train import process_data
import torch

def evaluate_model(data: LabeledData, tokenizer, model):
    device = torch.device("cpu")

    print("\n--- Testing (Evaluation) ---")
    model.eval() # Set the model to evaluation mode

    print("Processing test data...")

    # This part remains the same for getting overall predictions for the classification report
    test_processed = process_data(data, tokenizer, label_map, MAX_LENGTH)

    test_dataset = TensorDataset(
        test_processed['input_ids'],
        test_processed['attention_masks'],
        test_processed['labels'] # Now includes true labels
    )
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    all_true_labels = []
    all_predicted_labels = []
    all_input_ids = []

    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1]}
            true_labels = batch[2]

            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
            attention_mask = inputs['attention_mask']

            for i in range(predictions.shape[0]):
                seq_predictions = predictions[i][attention_mask[i] == 1].cpu().numpy()
                seq_true_labels = true_labels[i][attention_mask[i] == 1].cpu().numpy()
                seq_input_ids = inputs['input_ids'][i][attention_mask[i] == 1].cpu().numpy()

                all_predicted_labels.extend(seq_predictions)
                all_true_labels.extend(seq_true_labels)
                all_input_ids.extend(seq_input_ids)


    # --- Print Predicted Entities with Original Text (Sentence by Sentence) ---
    print("\n--- Predicted Entities ---")

    # Process sentence by sentence to correctly align predictions with original text and offsets
    token_index_in_flattened_list = 0 # To keep track of our position in the flattened prediction lists
    for sentence_index, (original_text, _) in enumerate(data):
        encoded_inputs = tokenizer(
            original_text,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
            return_offsets_mapping=True
        )
        offset_mapping = encoded_inputs['offset_mapping'][0]
        attention_mask = encoded_inputs['attention_mask'][0]
        input_ids = encoded_inputs['input_ids'][0] # Get input_ids for this sentence

        # Iterate through the tokens of the current sentence
        for token_idx in range(len(attention_mask)):
            # Only consider real tokens (not padding or special tokens indicated by attention_mask)
            if attention_mask[token_idx] == 1:
                # Get the predicted label for this specific token from the flattened list
                predicted_label_id = all_predicted_labels[token_index_in_flattened_list]
                predicted_label = id_to_label[predicted_label_id]

                # Get the original token ID to check for special tokens
                original_token_id = input_ids[token_idx].item()
                special_token_ids = set([tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id])

                # If the predicted label is not 'O' and it's not a special token, extract and print
                if predicted_label != 'O' and original_token_id not in special_token_ids:
                    char_start, char_end = offset_mapping[token_idx].tolist()

                    # Ensure offsets are valid before slicing
                    if char_start is not None and char_end is not None and char_end > char_start:
                        original_string = original_text[char_start:char_end]

                        # Simplify the predicted label (remove B- or I-)
                        simplified_predicted_label = predicted_label[2:] if predicted_label.startswith(('B-', 'I-')) else predicted_label

                        print(f"  Text: '{original_string}', Label: {simplified_predicted_label}")

                # Move to the next token in the flattened list
                token_index_in_flattened_list += 1


    try:
        from sklearn.metrics import classification_report

        # Simplified label processing logic for classification report (from previous turn)
        def simplify_label_for_report(label_id):
            original_label = id_to_label[label_id]
            if original_label == 'O':
                return 'O' # Keep 'O' for the report if needed, or filter later
            else:
                # Remove B- and I- prefixes
                return original_label[2:]

        simplified_true_labels_report = [simplify_label_for_report(label) for label in all_true_labels if id_to_label[label] != 'O']
        simplified_predicted_labels_report = [simplify_label_for_report(label) for label in all_predicted_labels if id_to_label[label] != 'O']

        report_target_names = sorted(list(set(simplified_true_labels_report + simplified_predicted_labels_report)))


        print("\nClassification Report:")
        print(classification_report(simplified_true_labels_report, simplified_predicted_labels_report,
                                    target_names=report_target_names,
                                    zero_division=0)) # Handle cases where a label has no true/predicted instances

    except ImportError:
        print("\nScikit-learn not found. Install it (`pip install scikit-learn`) to see evaluation metrics.")
        print("Skipping detailed classification report.")


if __name__ == "__main__":
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    model = BertForTokenClassification.from_pretrained(SAVE_DIRECTORY, num_labels=num_labels)
    evaluate_model(test_data, tokenizer, model)