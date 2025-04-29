from typing import List

from transformers import BertTokenizerFast, BertForTokenClassification

from bert_common import MODEL_NAME, num_labels, create_processed_record, SAVE_MODEL_EVERY_N_EPOCH, NUM_EPOCHS
from bert_ner_test import evaluate_model
from bert_ner_train import train_model
from common import ProcessedRecord
import os
# --- Example Usage of the Helper ---

# Creating the train_data using the helper
train_data: List[ProcessedRecord] = [
    create_processed_record("Dr. John Doe will see the patient.", [(4, 12, "NAME")]), # "John Doe"
    create_processed_record("The appointment is with Jane Smith tomorrow.", [(28, 38, "NAME")]), # "Jane Smith"
    create_processed_record("Report for Patient ID 12345.", []), # No entities
    create_processed_record("Contact Dr. Emily White at the clinic.", [(12, 23, "NAME")]), # "Emily White"
    create_processed_record("This is a normal sentence.", []),
    create_processed_record("Please refer to the notes from Dr. Brown.", [(33, 38, "NAME")]), # "Brown"
    create_processed_record("Mr. David Lee is the attending physician.", [(4, 13, "NAME")]), # "David Lee"
    create_processed_record("No specific name mentioned here.", []),
    create_processed_record("The nurse, Mary Johnson, provided the care.", [(11, 23, "NAME")]), # "Mary Johnson"
    create_processed_record("Patient admitted by Dr. Robert Green.", [(24, 36, "NAME")]), # "Robert Green"
    create_processed_record("Consultation with Dr. Alice Williams.", [(21, 36, "NAME")]), # "Alice Williams"
    create_processed_record("Patient transferred to Dr. Michael Brown's care.", [(25, 38, "NAME")]), # "Michael Brown"
    create_processed_record("Examined by Dr. Sarah Davis.", [(16, 27, "NAME")]), # "Sarah Davis"

    # Adding LOCATION and DATE examples
    create_processed_record("The meeting is in London.", [(19, 25, "LOCATION")]), # "London"
    create_processed_record("We will arrive on Monday.", [(18, 24, "DATE")]), # "Monday"
    create_processed_record("Visited Paris, France last year.", [(8, 13, "LOCATION"), (15, 21, "LOCATION")]), # "Paris", "France"
    create_processed_record("The deadline is December 31, 2024.", [(17, 33, "DATE")]), # "December 31, 2024"
    create_processed_record("He lives in New York City.", [(12, 25, "LOCATION")]), # "New York City"
    create_processed_record("The event is scheduled for next Tuesday.", [(28, 39, "DATE")]), # "next Tuesday"
    create_processed_record("Patient report from London, UK dated January 5, 2023.", [(20, 26, "LOCATION"), (28, 31, "LOCATION"), (38, 51, "DATE")]), # "London", "UK", "January 5, 2023"
    create_processed_record("Travel to Rome on the 15th.", [(9, 13, "LOCATION"), (21, 27, "DATE")]), # "Rome", "15th"
    create_processed_record("Appointment is on Friday afternoon.", [(18, 24, "DATE")]), # "Friday"
    create_processed_record("Location: Seattle. Date: 2025-04-27.", [(10, 17, "LOCATION"), (24, 34, "DATE")]), # "Seattle", "2025-04-27"
    create_processed_record("Dr. Chen is relocating to Boston next month.", [(28, 34, "LOCATION"), (35, 45, "DATE")]), # "Boston", "next month"
    create_processed_record("Meeting with Mr. Adams in Berlin on Wednesday.", [(20, 25, "NAME"), (30, 36, "LOCATION"), (40, 49, "DATE")]), # "Adams", "Berlin", "Wednesday"
]

# Creating the test_data using the helper
test_data: List[ProcessedRecord] = [
    create_processed_record("The patient name is Michael Clark.", [(24, 35, "NAME")]), # "Michael Clark"
    create_processed_record("Follow up with Dr. Susan Davis.", [(20, 31, "NAME")]), # "Susan Davis"
    create_processed_record("General health information.", []), # No entity
    create_processed_record("Check the records for Alice.", [(24, 29, "NAME")]), # "Alice"
    create_processed_record("Meeting with Dr. Williams and Dr. Jones.", [(19, 28, "NAME"), (37, 42, "NAME")]), # "Williams", "Jones"
    create_processed_record("Report filed by Dr. Evans.", [(21, 26, "NAME")]), # "Evans"
    create_processed_record("The primary contact is Mr. Thomas Green.", [(28, 39, "NAME")]), # "Thomas Green"

    # Test cases with LOCATION and DATE
    create_processed_record("Visited Washington D.C. last week.", [(9, 24, "LOCATION"), (25, 34, "DATE")]), # "Washington D.C.", "last week"
    create_processed_record("Conference held in Tokyo on March 10.", [(21, 26, "LOCATION"), (30, 39, "DATE")]), # "Tokyo", "March 10"
    create_processed_record("Dr. Wilson will be in Chicago until Friday.", [(4, 10, "NAME"), (25, 32, "LOCATION"), (39, 45, "DATE")]), # "Wilson", "Chicago", "Friday"
    create_processed_record("The report is due by 2024-12-01.", [(21, 31, "DATE")]), # "2024-12-01"
    create_processed_record("Sent details to Mr. White in Sydney.", [(24, 29, "NAME"), (33, 39, "LOCATION")]), # "White", "Sydney"
    create_processed_record("The event is scheduled for tomorrow in London.", [(28, 35, "DATE"), (39, 45, "LOCATION")]), # "tomorrow", "London"
    create_processed_record("Patient arrived from New York on July 4th.", [(22, 30, "LOCATION"), (34, 43, "DATE")]), # "New York", "July 4th"
]


if __name__ == "__main__":
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    model = BertForTokenClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)
    model_path_prefix = "tmp/saved_test_model"
    train_model(train_data, tokenizer, model, save_directory=model_path_prefix)

    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    for epoch in range(SAVE_MODEL_EVERY_N_EPOCH, NUM_EPOCHS + 1, SAVE_MODEL_EVERY_N_EPOCH):
        checkpoint_path = f"tmp/saved_test_model_epoch_{epoch}"
        if os.path.exists(checkpoint_path):
            print(f"\nLoading and evaluating model from {checkpoint_path}")
            eval_model = BertForTokenClassification.from_pretrained(checkpoint_path, num_labels=num_labels)
            evaluate_model(test_data, tokenizer, eval_model)


    print(f"\nLoading and evaluating model from {model_path_prefix}")
    model = BertForTokenClassification.from_pretrained("tmp/saved_test_model", num_labels=num_labels)
    evaluate_model(test_data, tokenizer, model)