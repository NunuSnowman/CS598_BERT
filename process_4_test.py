from transformers import BertForTokenClassification

import os
from bert_common import SAVE_DIRECTORY, num_labels, SAVE_MODEL_EVERY_N_EPOCH, NUM_EPOCHS
from bert_ner_test import evaluate_model
from process_3_train import load_and_split_data, initialize_model_and_tokenizer

if __name__ == "__main__":
    # Define the path to your input JSONL file
    # IMPORTANT: Replace with the actual path to your JSONL file
    jsonl_file_path =  './data/physionet_nurse/processed_data.jsonl'

    train_data, test_data = load_and_split_data(jsonl_file_path)

    tokenizer, model = initialize_model_and_tokenizer()

    # for epoch in range(SAVE_MODEL_EVERY_N_EPOCH, NUM_EPOCHS + 1, SAVE_MODEL_EVERY_N_EPOCH):
    #     checkpoint_path = f"tmp/saved_models_epoch_{epoch}"
    #     if os.path.exists(checkpoint_path):
    #         print(f"\nLoading and evaluating model from {checkpoint_path}")
    #         eval_model = BertForTokenClassification.from_pretrained(checkpoint_path, num_labels=num_labels)
    #         evaluate_model(test_data, tokenizer, eval_model)
    trained_model = BertForTokenClassification.from_pretrained(SAVE_DIRECTORY, num_labels=num_labels)
    print("Trained model loaded successfully.")

    evaluate_model(test_data, tokenizer, trained_model)

    print("\nScript finished.")