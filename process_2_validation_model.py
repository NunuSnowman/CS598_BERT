from typing import List

from transformers import BertTokenizerFast, BertForTokenClassification

import bert_common
from bert_common import create_processed_record, set_classify_type
from bert_ner_test import evaluate_model
from bert_ner_train import train_model
from common import ProcessedRecord
from record_label_maper import simplify_record_labels

# --- Example Usage of the Helper ---

# Creating the train_data using the helper
# Assuming create_processed_record is a helper function that structures the data
# Example structure (replace with your actual definition if different):
# from typing import List, Tuple, Dict
#
# class ProcessedRecord:
#     def __init__(self, text: str, entities: List[Tuple[int, int, str]]):
#         self.text = text
#         self.entities = entities
#
# def create_processed_record(text: str, entities: List[Tuple[int, int, str]]) -> ProcessedRecord:
#     return ProcessedRecord(text, entities)

# Creating the train_data using the helper
train_data: List[ProcessedRecord] = [
    create_processed_record("Dr. John Doe will see the patient.", [(4, 12, "NAME")]), # "John Doe" - Correct
    create_processed_record("The appointment is with Jane Smith tomorrow.", [(28, 38, "NAME")]), # "Jane Smith" - Correct
    create_processed_record("Report for Patient ID 12345.", []), # No entities - Correct
    create_processed_record("Contact Dr. Emily White at the clinic.", [(12, 23, "NAME")]), # "Emily White" - Correct
    create_processed_record("This is a normal sentence.", []), # Correct
    create_processed_record("Please refer to the notes from Dr. Brown.", [(33, 38, "NAME")]), # "Brown" - Correct
    create_processed_record("Mr. David Lee is the attending physician.", [(4, 13, "NAME")]), # "David Lee" - Correct
    create_processed_record("No specific name mentioned here.", []), # Correct
    create_processed_record("The nurse, Mary Johnson, provided the care.", [(11, 23, "NAME")]), # "Mary Johnson" - Correct
    create_processed_record("Patient admitted by Dr. Robert Green.", [(24, 36, "NAME")]), # "Robert Green" - Correct
    create_processed_record("Consultation with Dr. Alice Williams.", [(21, 36, "NAME")]), # "Alice Williams" - Correct
    create_processed_record("Patient transferred to Dr. Michael Brown's care.", [(25, 38, "NAME")]), # "Michael Brown" - Correct
    create_processed_record("Examined by Dr. Sarah Davis.", [(16, 27, "NAME")]), # "Sarah Davis" - Correct

    create_processed_record("The meeting is in London.", [(19, 25, "LOCATION")]), # "London" - Correct
    create_processed_record("We will arrive on Monday.", [(18, 24, "DATE")]), # "Monday" - Correct
    create_processed_record("Visited Paris, France last year.", [(8, 13, "LOCATION"), (15, 21, "LOCATION")]),
    create_processed_record("The deadline is December 31, 2024.", [(17, 33, "DATE")]), # "December 31, 2024" - Correct
    create_processed_record("He lives in New York City.", [(12, 25, "LOCATION")]), # "New York City" - Correct
    create_processed_record("The event is scheduled for next Tuesday.", [(28, 39, "DATE")]), # "next Tuesday" - Correct
    create_processed_record("Patient report from London, UK dated January 5, 2023.", [(20, 26, "LOCATION"), (28, 31, "LOCATION"), (38, 51, "DATE")]),
    create_processed_record("Travel to Rome on the 15th.", [(9, 13, "LOCATION"), (21, 25, "DATE")]), # FIX: "15th" is chars 21-25
    create_processed_record("Appointment is on Friday afternoon.", [(18, 24, "DATE")]), # "Friday" - Correct
    create_processed_record("Location: Seattle. Date: 2025-04-27.", [(10, 17, "LOCATION"), (24, 34, "DATE")]),
    create_processed_record("Dr. Chen is relocating to Boston next month.", [(28, 34, "LOCATION"), (35, 45, "DATE")]),
    create_processed_record("Meeting with Mr. Adams in Berlin on Wednesday.", [(20, 25, "NAME"), (30, 36, "LOCATION"), (40, 49, "DATE")]),
]

# Creating the test_data using the helper
test_data: List[ProcessedRecord] = [
    create_processed_record("The patient name is Michael Clark.", [(24, 35, "NAME")]), # "Michael Clark" - Correct
    create_processed_record("Follow up with Dr. Susan Davis.", [(20, 31, "NAME")]), # "Susan Davis" - Correct
    create_processed_record("General health information.", []),
    create_processed_record("Check the records for Alice.", [(24, 29, "NAME")]),
    # "Williams", "Jones" - Correct
    create_processed_record("Meeting with Dr. Williams and Dr. Jones.", [(19, 28, "NAME"), (37, 42, "NAME")]),
    create_processed_record("Report filed by Dr. Evans.", [(21, 26, "NAME")]),
    create_processed_record("The primary contact is Mr. Thomas Green.", [(28, 39, "NAME")]),
    create_processed_record("Visited Washington D.C. last week.", [(9, 24, "LOCATION"), (25, 34, "DATE")]),
    create_processed_record("Conference held in Tokyo on March 10.", [(21, 26, "LOCATION"), (30, 39, "DATE")]),
    create_processed_record("Dr. Wilson will be in Chicago until Friday.", [(4, 10, "NAME"), (25, 32, "LOCATION"), (39, 45, "DATE")]),
    create_processed_record("The report is due by 2024-12-01.", [(21, 31, "DATE")]), # "2024-12-01" - Correct
    create_processed_record("Sent details to Mr. White in Sydney.", [(24, 29, "NAME"), (33, 39, "LOCATION")]),
    create_processed_record("The event is scheduled for tomorrow in London.", [(28, 36, "DATE"), (40, 46, "LOCATION")]),
    create_processed_record("Patient arrived from New York on July 4th.", [(22, 30, "LOCATION"), (34, 43, "DATE")]),
]
train_data = simplify_record_labels(train_data)
test_data = simplify_record_labels(test_data)


def run():
    global train_data
    global test_data
    tokenizer = BertTokenizerFast.from_pretrained(bert_common.model_name)
    model = BertForTokenClassification.from_pretrained(bert_common.model_name, num_labels=bert_common.num_labels)
    model_path_prefix = "tmp/saved_test_model"
    train = simplify_record_labels(train_data)
    test = simplify_record_labels(test_data)
    train_model(train, tokenizer, model, save_directory=model_path_prefix)
    tokenizer = BertTokenizerFast.from_pretrained(bert_common.model_name)
    print(f"\nLoading and evaluating model from {model_path_prefix}")
    model = BertForTokenClassification.from_pretrained("tmp/saved_test_model", num_labels=bert_common.num_labels)
    evaluate_model(test, tokenizer, model)


if __name__ == "__main__":
    bert_common.bert_print_debug_log = True
    bert_common.SAVE_MODEL_EVERY_N_EPOCH = 0
    set_classify_type(use_multi_class=True)
    run()
    set_classify_type(use_multi_class=False)
    run()