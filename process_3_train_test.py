import os
import random
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast, BertForTokenClassification
import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Tuple, Dict

# Assuming these files are in the same directory or accessible in the path
from record_file_reader import read_json_list_as_processed_records
from record_label_maper import simplify_label_string, simplify_record_labels  # Import the label simplification function
from bert_common import MODEL_NAME, SAVE_DIRECTORY, MAX_LENGTH, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, ProcessedRecord, MaskInfo # Import necessary components
from bert_ner_train import train_model
from bert_ner_test import evaluate_model
# --- Define Simplified Label Mapping ---
# Create a new label map based on the simplified labels
simplified_label_map = {'O': 0, 'NAME': 1, 'LOCATION': 2, 'DATE': 3}
simplified_id_to_label = {v: k for k, v in simplified_label_map.items()}
simplified_num_labels = len(simplified_label_map)

# --- Reusable Methods ---

def load_and_split_data(jsonl_path: str, test_size: float = 0.2, random_state: int = 42) -> Tuple[List[ProcessedRecord], List[ProcessedRecord]]:
    print(f"Reading data from {jsonl_path}...")
    all_records: list[ProcessedRecord] = read_json_list_as_processed_records(jsonl_path)
    print(f"Successfully read {len(all_records)} records.")
    return train_test_split(all_records, test_size=test_size, random_state=random_state)

def initialize_model_and_tokenizer() -> Tuple[BertTokenizerFast, BertForTokenClassification]:
    print(f"\nInitializing tokenizer: {MODEL_NAME}")
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    print(f"Initializing model: {MODEL_NAME} with {simplified_num_labels} labels.")
    # Use the simplified_num_labels for the model
    model = BertForTokenClassification.from_pretrained(MODEL_NAME, num_labels=simplified_num_labels)
    return tokenizer, model



# --- Main Execution ---
if __name__ == "__main__":
    # Define the path to your input JSONL file
    # IMPORTANT: Replace with the actual path to your JSONL file
    jsonl_file_path =  './data/physionet_nurse/processed_data.jsonl'

    output_model_dir = './tmp/saved_models' # Using the SAVE_DIRECTORY from bert_common

    train_data, test_data = load_and_split_data(jsonl_file_path)
    train_data = simplify_record_labels(train_data)
    test_data = simplify_record_labels(test_data)

    tokenizer, model = initialize_model_and_tokenizer()

    train_model(train_data, tokenizer, model, save_directory=output_model_dir)

    # 4. Load the trained model for evaluation
    print(f"\nLoading trained model from {output_model_dir} for evaluation...")
    try:
        # Ensure the model is loaded with the correct number of simplified labels
        trained_model = BertForTokenClassification.from_pretrained(output_model_dir, num_labels=simplified_num_labels)
        print("Trained model loaded successfully.")
    except Exception as e:
        print(f"Error loading trained model from {output_model_dir}: {e}")
        exit() # Exit if model loading fails

    # 5. Evaluate the model
    evaluate_model(test_data, tokenizer, trained_model)

    print("\nScript finished.")
