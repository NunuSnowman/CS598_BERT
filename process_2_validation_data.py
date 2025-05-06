import json
import os
import dataclasses
from typing import List
import re  # Import re for whitespace normalization

from common import ProcessedRecord, MaskInfo
from record_file_reader import read_json_list_as_processed_records_iterator, read_json_list_as_processed_records
from record_validator import validate_mapping


class RecordValidator:
    def run_validation(self, data_folder_path: str):
        """
        Reads all JSONL files in the specified folder, validates each record,
        and prints validation failures.

        Args:
            data_folder_path: The path to the folder containing JSONL files.
        """
        print(f"Starting validation for JSONL files in: {data_folder_path}")

        jsonl_files = [f for f in os.listdir(data_folder_path) if f.endswith('.jsonl')]
        total_records_validated = 0
        failed_validations = 0

        for file_name in jsonl_files:
            file_path = os.path.join(data_folder_path, file_name)
            print(f"\nValidating file: {file_name}")
            record_list = read_json_list_as_processed_records(file_path)
            for record in record_list:
                processed_record: ProcessedRecord = record
                total_records_validated += 1
                # Validate the record using the imported validate_mapping function
                if not validate_mapping(processed_record):
                    failed_validations += 1
                    # print(f"Validation Failed in {file_name}, Record #{line_num}")
                    # Optional: Print more details about the failed record
                    print(f"  Original Text: {processed_record.text_record}")
                    print(f"  Masked Text: {processed_record.res_record}")
                    print(f"  Masks: {processed_record.mask_info}")
                    print("-" * 20)  # Separator

        print("\nValidation process finished.")
        print(f"Total records validated: {total_records_validated}")
        print(f"Total validation failures: {failed_validations}")


if __name__ == "__main__":
    # Define the path to the folder containing your JSONL files
    # You will need to change this path to where your processed_data.jsonl file(s) are located

    validator = RecordValidator()
    validator.run_validation("data/physionet_nurse")
    validator.run_validation("data/DocRED")
