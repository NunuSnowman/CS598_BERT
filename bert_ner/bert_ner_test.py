from bert_ner.bert_common import *
from bert_ner.bert_ner_train import process_data
import torch


def simplify_label(label_id):
    original_label = id_to_label[label_id]
    if original_label == 'O':
        return 'O'
    else:
        # Remove B- and I- prefixes
        return original_label[2:]


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
        input_ids = encoded_inputs['input_ids'][0]

        current_entity_text = ""
        current_entity_label = None
        entity_start_char = -1
        # Iterate through the tokens of the current sentence
        for token_idx in range(len(attention_mask)):
            # Only consider real tokens (not padding or special tokens indicated by attention_mask)
            if attention_mask[token_idx] == 1:
                # Get the predicted label for this specific token
                predicted_label_id = all_predicted_labels[token_index_in_flattened_list]
                simplified_predicted_label = simplify_label(predicted_label_id)

                # Get the original token ID to check for special tokens
                original_token_id = input_ids[token_idx].item()
                special_token_ids = set([tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id])

                # Get character offsets for the current token
                char_start, char_end = offset_mapping[token_idx].tolist()

                if simplified_predicted_label != 'O' and original_token_id not in special_token_ids:
                    # This token is part of a named entity
                    if current_entity_label is None:
                        # Start of a new entity
                        current_entity_text = original_text[char_start:char_end]
                        current_entity_label = simplified_predicted_label
                        entity_start_char = char_start
                    elif simplified_predicted_label == current_entity_label:
                        # Continuation of the current entity
                        # Append the text, considering potential spaces in the original text
                        # Find the gap between the end of the last token and the start of the current token
                        previous_token_end_char = offset_mapping[token_idx - 1].tolist()[1] if token_idx > 0 and offset_mapping[token_idx - 1].tolist()[1] is not None else char_start
                        gap_text = original_text[previous_token_end_char:char_start] if previous_token_end_char < char_start else ""
                        current_entity_text += gap_text + original_text[char_start:char_end]

                    else:
                        # End of previous entity, start of a new one with a different label
                        print(f"  Entity: '{current_entity_text}', Label: {current_entity_label}")
                        current_entity_text = original_text[char_start:char_end]
                        current_entity_label = simplified_predicted_label
                        entity_start_char = char_start
                else:
                    # This token is 'O' or a special token, which ends any ongoing entity
                    if current_entity_label is not None:
                        # End of an entity
                        print(f"  Entity: '{current_entity_text}', Label: {current_entity_label}")
                        current_entity_text = ""
                        current_entity_label = None
                        entity_start_char = -1

                # Move to the next token in the flattened list
                token_index_in_flattened_list += 1

        # End of sentence: check if there's an ongoing entity to print
        if current_entity_label is not None:
            print(f"  Entity: '{current_entity_text}', Label: {current_entity_label}")

    try:
        from sklearn.metrics import classification_report

        simplified_true_labels_report = [simplify_label(label) for label in all_true_labels if id_to_label[label] != 'O']
        simplified_predicted_labels_report = [simplify_label(label) for label in all_predicted_labels if id_to_label[label] != 'O']

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