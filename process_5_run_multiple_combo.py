from transformers import BertForTokenClassification

import bert_common
from bert_common import SAVE_DIRECTORY
from bert_ner_test import evaluate_model
from bert_ner_train import train_model
from process_3_train import load_and_split_data, initialize_model_and_tokenizer


def run():
    bert_common.bert_print_debug_log = True
    bert_common.SAVE_MODEL_EVERY_N_EPOCH = 0
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

    for model_name in [
        'bert-base-uncased',
        "allenai/scibert_scivocab_cased",
        "dmis-lab/biobert-base-cased-v1.1",
        "emilyalsentzer/Bio_ClinicalBERT"
    ]:
        bert_common.model_name = model_name
        bert_common.set_classify_type(use_multi_class=True)
        print(f"Using model {bert_common.model_name}")
        run()
        bert_common.set_classify_type(use_multi_class=False)
        run()
