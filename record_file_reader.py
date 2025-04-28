import json
import dataclasses
from typing import List, Dict, Any

# Assuming common.py is accessible and contains the ProcessedRecord and MaskInfo definitions
from common import ProcessedRecord, MaskInfo

def read_json_list_as_processed_records(file_path: str) -> List[ProcessedRecord]:
    return list(read_json_list_as_processed_records_iterator(file_path))

def read_json_list_as_processed_records_iterator(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            record_dict = json.loads(line)

            masks_list_of_dicts = record_dict.get('mask_info', [])
            if not isinstance(masks_list_of_dicts, list):
                print(f"Warning: 'mask_info' on line {line_num} is not a list. Skipping.")
                continue

            masks_list_of_objects = [MaskInfo(**mask_dict) for mask_dict in masks_list_of_dicts]

            # Create the ProcessedRecord object
            processed_record = ProcessedRecord(
                res_record=record_dict.get('res_record', ''),
                text_record=record_dict.get('text_record', ''),
                mask_info=masks_list_of_objects
            )

            # Yield the processed record instead of appending to a list
            yield processed_record