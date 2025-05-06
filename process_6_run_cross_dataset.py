from transformers import BertForTokenClassification

import bert_common
import process_2_validation_model
from bert_common import SAVE_DIRECTORY
from bert_ner_test import evaluate_model
from bert_ner_train import train_model
from process_3_train import load_and_split_data, initialize_model_and_tokenizer

def get_DOCRed():
    jsonl_file_path = './data/DocRED/processed_data.jsonl'
    _, data = load_and_split_data(jsonl_file_path, test_size=0.01)
    return data

def get_physioNet():
    jsonl_file_path = './data/physionet_nurse/processed_data.jsonl'
    _, data = load_and_split_data(jsonl_file_path)
    return data

def run():
    train_data = get_DOCRed()
    tokenizer, model = initialize_model_and_tokenizer()
    train_model(train_data, tokenizer, model, save_directory=SAVE_DIRECTORY)
    trained_model = BertForTokenClassification.from_pretrained(SAVE_DIRECTORY, num_labels=bert_common.num_labels)
    print("Trained model loaded successfully.")
    test_data = get_DOCRed()
    evaluate_model(test_data, tokenizer, trained_model)
    print("\nScript finished.")


if __name__ == "__main__":
    bert_common.bert_print_debug_log = True
    bert_common.SAVE_MODEL_EVERY_N_EPOCH = 0
    bert_common.set_classify_type(use_multi_class=True)
    print(f"Using model {bert_common.model_name}")
    run()
