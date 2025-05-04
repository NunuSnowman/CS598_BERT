import string

import bert_common
from bert_common import *
import torch
from sklearn.metrics import classification_report


def evaluation_label(label_id):
    if label_id == bert_common.IGNORE_INDEX:
        # Return a placeholder or None for ignored indices, or handle as needed
        # For evaluation metrics, we will filter these out, so the exact return value here
        # for -100 might not matter if filtered before report generation.
        # However, let's return None or a distinct string just in case.
        return None # Or a string like "IGNORED"
    original_label = bert_common.id_to_label[label_id]
    if original_label == 'O':
        return 'O'
    else:
        # Remove B- and I- prefixes
        return original_label[2:]

def print_classification_reports(filtered_true_labels_ids, filtered_predicted_labels_ids, filtered_input_ids, tokenizer):
    """
    Prints classification reports based on different token length criteria,
    excluding tokens with true labels equal to IGNORE_INDEX.

    Args:
        all_true_labels (list): List of true label IDs for all tokens (includes IGNORE_INDEX).
        all_predicted_labels (list): List of predicted label IDs for all tokens.
        all_input_ids (list): List of input IDs for all tokens.
        tokenizer: The BERT tokenizer with a decode method.
    """


    try:
        if filtered_true_labels_ids and filtered_predicted_labels_ids:
            # Convert filtered label IDs to simplified labels (e.g., 'O', 'NAME')
            simplified_filtered_true_labels = [evaluation_label(label) for label in filtered_true_labels_ids]
            simplified_filtered_predicted_labels = [evaluation_label(label) for label in filtered_predicted_labels_ids]
            filtered_token_texts = [tokenizer.decode([token_id], skip_special_tokens=False) for token_id in filtered_input_ids]


            # Criteria 1: Including all filtered tokens (excluding those with true label IGNORE_INDEX)
            print("\n--- Classification Report (Filtered Tokens - Excluding IGNORE_INDEX) ---")
            # Determine target names from the filtered simplified labels
            report_target_names = sorted(list(set(simplified_filtered_true_labels + simplified_filtered_predicted_labels)))
            print(classification_report(simplified_filtered_true_labels, simplified_filtered_predicted_labels,
                                        target_names=report_target_names,
                                        zero_division=0))

            # Criteria 2: Including only filtered tokens > 1 char
            filtered_true_labels_gt1 = []
            filtered_predicted_labels_gt1 = []
            for i in range(len(filtered_token_texts)):
                # Filter out special tokens and then check length (already handled IGNORE_INDEX true labels)
                # Check against original special token IDs just in case, though IGNORE_INDEX should cover most.
                if filtered_input_ids[i] not in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]:
                    # Remove '##' prefix for length check if it exists
                    token_text_for_length = filtered_token_texts[i].replace('##', '')
                    # Also strip potential punctuation for length check
                    token_text_for_length = token_text_for_length.strip(string.punctuation)
                    if len(token_text_for_length) > 1:
                        filtered_true_labels_gt1.append(simplified_filtered_true_labels[i])
                        filtered_predicted_labels_gt1.append(simplified_filtered_predicted_labels[i])


            if filtered_true_labels_gt1 and filtered_predicted_labels_gt1:
                print("\n--- Classification Report (Filtered Tokens > 1 Char) ---")
                report_target_names_gt1 = sorted(list(set(filtered_true_labels_gt1 + filtered_predicted_labels_gt1)))
                print(classification_report(filtered_true_labels_gt1, filtered_predicted_labels_gt1,
                                            target_names=report_target_names_gt1,
                                            zero_division=0))
            else:
                print("\nNo filtered tokens > 1 character found for evaluation.")


            # Criteria 3: Including only filtered tokens > 2 chars
            filtered_true_labels_gt2 = []
            filtered_predicted_labels_gt2 = []
            for i in range(len(filtered_token_texts)):
                # Filter out special tokens and then check length (already handled IGNORE_INDEX true labels)
                if filtered_input_ids[i] not in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]:
                    # Remove '##' prefix for length check if it exists
                    token_text_for_length = filtered_token_texts[i].replace('##', '')
                    # Also strip potential punctuation for length check
                    token_text_for_length = token_text_for_length.strip(string.punctuation)
                    if len(token_text_for_length) > 2:
                        filtered_true_labels_gt2.append(simplified_filtered_true_labels[i])
                        filtered_predicted_labels_gt2.append(simplified_filtered_predicted_labels[i])


            if filtered_true_labels_gt2 and filtered_predicted_labels_gt2:
                print("\n--- Classification Report (Filtered Tokens > 2 Chars) ---")
                report_target_names_gt2 = sorted(list(set(filtered_true_labels_gt2 + filtered_predicted_labels_gt2)))
                print(classification_report(filtered_true_labels_gt2, filtered_predicted_labels_gt2,
                                            target_names=report_target_names_gt2,
                                            zero_division=0))
            else:
                print("\nNo filtered tokens > 2 characters found for evaluation.")

        else:
            print("\nNo true or predicted labels available after filtering for classification report.")

    except ImportError:
        print("\nScikit-learn not found. Install it (`pip install scikit-learn`) to see evaluation metrics.")
        print("Skipping detailed classification report.")


def evaluate_model(data: [ProcessedRecord],
                   tokenizer: BertTokenizerFast,
                   model: BertForTokenClassification,):
    device = torch.device("cpu")

    print("\n--- Testing (Evaluation) ---")
    model.eval() # Set the model to evaluation mode

    print("Processing test data...")

    # process_data_label will now return labels with IGNORE_INDEX for tokens to skip
    test_processed = process_data_label(data, tokenizer, MAX_LENGTH)

    test_dataset = TensorDataset(
        test_processed['input_ids'],
        test_processed['attention_masks'],
        test_processed['labels'] # This now includes true labels with IGNORE_INDEX
    )
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    all_true_labels = []
    all_predicted_labels = []
    all_input_ids = [] # Keep input IDs for potential debugging or token text

    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1]}
            true_labels = batch[2] # Get the true labels (including IGNORE_INDEX)

            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
            attention_mask = inputs['attention_mask']

            # Collect all predictions and true labels for later filtering and reporting
            # Masking by attention_mask[i] == 1 is still correct here to exclude padding
            for i in range(predictions.shape[0]):
                seq_predictions = predictions[i][attention_mask[i] == 1].cpu().numpy()
                seq_true_labels = true_labels[i][attention_mask[i] == 1].cpu().numpy()
                seq_input_ids = inputs['input_ids'][i][attention_mask[i] == 1].cpu().numpy()


                all_predicted_labels.extend(seq_predictions)
                all_true_labels.extend(seq_true_labels)
                all_input_ids.extend(seq_input_ids)


    # --- Pass filtered lists to reporting functions ---
    # print_predictions(all_predicted_labels, data, tokenizer) # This function might need further modification to align with filtered data

    # Filter the collected labels and input_ids before passing to reporting functions
    # Define IGNORE_INDEX here as well for clarity or import it from bert_common

    filtered_all_true_labels = []
    filtered_all_predicted_labels = []
    filtered_all_input_ids = []

    for i in range(len(all_true_labels)):
        if all_true_labels[i] != bert_common.IGNORE_INDEX:
            filtered_all_true_labels.append(all_true_labels[i])
            filtered_all_predicted_labels.append(all_predicted_labels[i])
            filtered_all_input_ids.append(all_input_ids[i])

    print(f"origin data size = {len(all_true_labels)}, filtered data size = {len(filtered_all_true_labels)}")
    # Now pass the filtered lists to the reporting functions
    print_classification_reports(filtered_all_true_labels, filtered_all_predicted_labels, filtered_all_input_ids, tokenizer)

    # Note: print_debug_logs also needs to handle the filtering or receive filtered data.
    # For simplicity, let's update print_debug_logs to accept filtered data.
    print_debug_logs(filtered_all_input_ids, filtered_all_predicted_labels, filtered_all_true_labels, tokenizer)



def print_predictions(all_predicted_labels, data, tokenizer):
    # This function is more complex to update because it iterates based on the original data structure
    # and relies on aligning with a flattened list.
    # To correctly align with the filtered labels, this function would need significant changes
    # or a different approach to reconstruction.
    # For now, keeping the original print_predictions might lead to misalignment if used with filtered data.
    # Consider if this function is still necessary or how to adapt it to the new filtering.
    print("\nSkipping print_predictions as it needs significant updates to align with filtered data.")
    pass # Skipping this function for now


def print_debug_logs(filtered_input_ids, filtered_predicted_labels, filtered_true_labels, tokenizer):
    """
    Prints debug logs for tokens, excluding those with true labels equal to IGNORE_INDEX.

    Args:
        filtered_input_ids (list): List of input IDs for filtered tokens.
        filtered_predicted_labels (list): List of predicted label IDs for filtered tokens.
        filtered_true_labels (list): List of true label IDs for filtered tokens.
        tokenizer: The BERT tokenizer with a decode method.
    """
    if bert_common.bert_print_debug_log:
        # Define IGNORE_INDEX here or import it if needed, but it's already filtered out.
        # IGNORE_INDEX = -100
        count = 0
        for i in range(len(filtered_input_ids)):
            token_id = filtered_input_ids[i]
            true_label = evaluation_label(filtered_true_labels[i]) # evaluation_label now handles None for -100
            predicted_label = evaluation_label(filtered_predicted_labels[i])

            # Skip printing if the true label was originally IGNORE_INDEX (already filtered, but double check)
            if filtered_true_labels[i] == -100: # Use the raw ID for the check
                print("Warning, filtered data contains IGNORE_INDEX.")
                continue

            if count%20 == 0:
                print("\n")
            count += 1
            token_text = tokenizer.decode([token_id], skip_special_tokens=False)

            # Now true_label and predicted_label from evaluation_label won't be None if original was not -100
            if true_label != predicted_label:
                # Print token and label. Handle potential sub-word tokenization (e.g., ##ing)
                # You might want to add spaces or newlines to structure the output
                if token_text.startswith('##'):
                    print(f"{token_text}❌ {predicted_label}({true_label}) ")  # No space before sub-word token
                else:
                    print(f"{token_text}❌ {predicted_label}({true_label}) ",
                          end="")  # Add space before new word token
            elif true_label != "O": # Only print correctly predicted non-'O' labels
                print(f"{token_text}✅ ({true_label}) ")