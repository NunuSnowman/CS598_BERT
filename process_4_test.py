from transformers import BertForTokenClassification

from bert_common import SAVE_DIRECTORY, num_labels
from bert_ner_test import evaluate_model
from process_3_train import load_and_split_data, initialize_model_and_tokenizer

if __name__ == "__main__":
    # Define the path to your input JSONL file
    # IMPORTANT: Replace with the actual path to your JSONL file
    jsonl_file_path =  './data/physionet_nurse/processed_data.jsonl'

    train_data, test_data = load_and_split_data(jsonl_file_path)

    tokenizer, model = initialize_model_and_tokenizer()

    trained_model = BertForTokenClassification.from_pretrained(SAVE_DIRECTORY, num_labels=num_labels)
    print("Trained model loaded successfully.")

    evaluate_model(test_data, tokenizer, trained_model)

    print("\nScript finished.")