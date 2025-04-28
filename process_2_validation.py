import json
import os
import dataclasses
from typing import List
import re # Import re for whitespace normalization

from common import ProcessedRecord, MaskInfo
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

            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    total_records_validated += 1
                    try:
                        record_dict = json.loads(line)

                        # Convert dictionary to ProcessedRecord object
                        # Need to manually convert masks list items to MaskInfo objects
                        masks_list_of_dicts = record_dict.get('mask_info', [])
                        # Ensure masks is a list before attempting list comprehension
                        if not isinstance(masks_list_of_dicts, list):
                            print(f"Validation Error: 'mask_info' field is not a list on line {line_num} in {file_name}. Skipping record.")
                            failed_validations += 1
                            continue # Skip to next line

                        masks_list_of_objects = [MaskInfo(**mask_dict) for mask_dict in masks_list_of_dicts]

                        processed_record = ProcessedRecord(
                            res_record=record_dict.get('res_record', ''),
                            text_record=record_dict.get('text_record', ''),
                            mask_info=masks_list_of_objects
                        )

                        # Validate the record using the imported validate_mapping function
                        if not validate_mapping(processed_record):
                            failed_validations += 1
                            print(f"Validation Failed in {file_name}, Record #{line_num}")
                            # Optional: Print more details about the failed record
                            print(f"  Original Text: {processed_record.text_record}")
                            print(f"  Masked Text: {processed_record.res_record}")
                            print(f"  Masks: {processed_record.mask_info}")
                            print("-" * 20) # Separator

                    except json.JSONDecodeError:
                        failed_validations += 1
                        print(f"Error: Could not decode JSON on line {line_num} in {file_name}. Skipping record.")
                    except KeyError as e:
                        failed_validations += 1
                        print(f"Error: Missing key {e} on line {line_num} in {file_name}. Skipping record.")
                    except TypeError as e:
                        failed_validations += 1
                        print(f"Error: Type error when creating MaskInfo or ProcessedRecord on line {line_num} in {file_name}: {e}. Skipping record.")
                    except Exception as e:
                        failed_validations += 1
                        print(f"An unexpected error occurred processing line {line_num} in {file_name}: {e}")


        print("\nValidation process finished.")
        print(f"Total records validated: {total_records_validated}")
        print(f"Total validation failures: {failed_validations}")


if __name__ == "__main__":
    # Define the path to the folder containing your JSONL files
    # You will need to change this path to where your processed_data.jsonl file(s) are located
    data_folder_to_validate = "data/physionet_nurse" # Example path

    validator = RecordValidator()
    validator.run_validation(data_folder_to_validate)
