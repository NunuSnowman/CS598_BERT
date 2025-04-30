from sklearn.model_selection import train_test_split
from typing import List, Tuple

from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast, BertForTokenClassification

import bert_common
from bert_common import SAVE_DIRECTORY, ProcessedRecord  # Import necessary components
from bert_ner_train import train_model
# Assuming these files are in the same directory or accessible in the path
from record_file_reader import read_json_list_as_processed_records
from record_label_maper import simplify_record_labels  # Import the label simplification function


# --- Reusable Methods ---

def load_and_split_data(jsonl_path: str, test_size: float = 0.5, random_state: int = 42) -> Tuple[
    List[ProcessedRecord], List[ProcessedRecord]]:
    print(f"Reading data from {jsonl_path}...")
    all_records: list[ProcessedRecord] = read_json_list_as_processed_records(jsonl_path)
    print(f"Successfully read {len(all_records)} records.")
    train_data, test_data = train_test_split(all_records, test_size=test_size, random_state=random_state)
    train_data = simplify_record_labels(train_data)
    test_data = simplify_record_labels(test_data)
    return train_data, test_data


def initialize_model_and_tokenizer() -> Tuple[BertTokenizerFast, BertForTokenClassification]:
    print(f"\nInitializing tokenizer: {bert_common.model_name}")
    tokenizer = BertTokenizerFast.from_pretrained(bert_common.model_name)
    print(f"Initializing model: {bert_common.model_name} with {bert_common.num_labels} labels.")
    # Use the simplified_num_labels for the model
    model = BertForTokenClassification.from_pretrained(bert_common.model_name, num_labels=bert_common.num_labels)
    return tokenizer, model


# --- Main Execution ---
if __name__ == "__main__":
    # Define the path to your input JSONL file
    # IMPORTANT: Replace with the actual path to your JSONL file
    jsonl_file_path = './data/physionet_nurse/processed_data.jsonl'

    train_data, test_data = load_and_split_data(jsonl_file_path)

    tokenizer, model = initialize_model_and_tokenizer()

    train_model(train_data, tokenizer, model, save_directory=SAVE_DIRECTORY)
