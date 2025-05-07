import torch
from transformers import BertForTokenClassification

import bert_common
import bert_ner_train
from bert_common import SAVE_DIRECTORY
from bert_ner_test import evaluate_model
from bert_ner_train import train_model
from process_3_train import load_and_split_data, initialize_model_and_tokenizer



def run():
    jsonl_file_path = './data/physionet_nurse/processed_data.jsonl'
    train_data, test_data = load_and_split_data(jsonl_file_path)
    tokenizer, model = initialize_model_and_tokenizer()
    train_model(train_data, tokenizer, model, save_directory=SAVE_DIRECTORY)
    trained_model = BertForTokenClassification.from_pretrained(SAVE_DIRECTORY, num_labels=bert_common.num_labels)
    print("Trained model loaded successfully.")
    evaluate_model(test_data, tokenizer, trained_model)
    print("\nScript finished.")


if __name__ == "__main__":
    bert_common.bert_print_debug_log = False
    bert_common.SAVE_MODEL_EVERY_N_EPOCH = 0
    bert_common.use_crossing_entropy_loss = bert_ner_train.focal_loss

    print(f"Using focal_loss")
    bert_common.set_classify_type(use_multi_class=False)
    run()

    bert_common.use_crossing_entropy_loss = bert_ner_train.asymmetric_focal_loss
    print(f"Using asymmetric_focal_loss")
    run()