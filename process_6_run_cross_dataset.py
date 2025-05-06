from transformers import BertForTokenClassification

import bert_common
from bert_ner_test import evaluate_model
from bert_ner_train import train_model
from process_3_train import load_and_split_data, initialize_model_and_tokenizer



def get_DOCRed(test_size=0.01):
    jsonl_file_path = './data/DocRED/processed_data.jsonl'
    train_data, test_data = load_and_split_data(jsonl_file_path, test_size=test_size)
    return train_data, test_data

def get_physioNet(test_size=0.5):
    jsonl_file_path = './data/physionet_nurse/processed_data.jsonl'
    train_data, test_data = load_and_split_data(jsonl_file_path, test_size=test_size)
    return train_data, test_data

DR_MODEL_DIR = "./tmp/saved_models_DR"
PN_MODEL_DIR = "./tmp/saved_models_PN"
def train():
    docRED_train_data, _ = get_DOCRed(0.98)
    tokenizer, doc_model = initialize_model_and_tokenizer()
    train_model(docRED_train_data, tokenizer, doc_model, save_directory=DR_MODEL_DIR)

    physioNet_train_data, _ = get_physioNet()
    tokenizer, pn_model = initialize_model_and_tokenizer()
    train_model(physioNet_train_data, tokenizer, pn_model, save_directory=PN_MODEL_DIR)

def evaluate(model_dir):
    _, docRED_test_data = get_DOCRed(0.02)
    _, physioNET_test_data = get_physioNet()
    tokenizer, _ = initialize_model_and_tokenizer()
    trained_model = BertForTokenClassification.from_pretrained(model_dir, num_labels=bert_common.num_labels)
    print(f"Trained model loaded successfully.{model_dir}")
    print("Using DOCRED test data")
    evaluate_model(docRED_test_data, tokenizer, trained_model)

    trained_model = BertForTokenClassification.from_pretrained(model_dir, num_labels=bert_common.num_labels)
    print(f"Trained model loaded successfully.{model_dir}")
    print("Using PhysioNET test data")
    evaluate_model(physioNET_test_data, tokenizer, trained_model)



if __name__ == "__main__":
    bert_common.bert_print_debug_log = False
    bert_common.SAVE_MODEL_EVERY_N_EPOCH = 0
    bert_common.set_classify_type(use_multi_class=False)
    print(f"Using model {bert_common.model_name}")
    train()
    evaluate(DR_MODEL_DIR)
    evaluate(PN_MODEL_DIR)
