import re
from typing import List

from common import ProcessedRecord


def simplify_record_labels(records: List[ProcessedRecord]) -> List[ProcessedRecord]:
    print(f"Simplifying labels for {len(records)} records...")
    for record in records:
        for mask_info_item in record.mask_info:
            original_label = mask_info_item.label
            simplified_label = simplify_label_string(original_label)
            mask_info_item.label = simplified_label
    return records

def simplify_label_string(label: str) -> str:
    """
    Simplifies a given label string into one of the categories:
    "O", "NAME", "LOCATION", or "DATE" based on predefined rules.

    Args:
        label: The input label string (e.g., "B-NAME", "I-LOCATION", "O", "08-28", "Address").

    Returns:
        A simplified label string ("O", "NAME", "LOCATION", or "DATE").
    """
    # Convert label to uppercase for case-insensitive matching
    label_upper = label.upper()

    # Rule 1: Check for DATE format (digits and dashes)
    # This regex checks if the entire string consists only of digits and dashes
    if re.fullmatch(r'[\d-]+', label):
        return "DATE"

    # Rule 2: Check for LOCATION keywords
    # Add more keywords to this set to extend LOCATION mapping
    location_keywords = {"ADDRESS", "LOCATION", "CITY"}
    for keyword in location_keywords:
        if keyword in label_upper:
            return "LOCATION"

    # Rule 3: Check for NAME keywords
    # Add more keywords to this set to extend NAME mapping
    name_keywords = {"NAME", "HOSPITAL"} # Added Hospital as per request
    for keyword in name_keywords:
        if keyword in label_upper:
            return "NAME"

    # Default Rule: If none of the above match, return "O"
    return "O"

#
# # --- Example Usage ---
# print(f"'B-NAME' -> {simplify_label_string('B-NAME')}")
# print(f"'I-LOCATION' -> {simplify_label_string('I-LOCATION')}")
# print(f"'O' -> {simplify_label_string('O')}")
# print(f"'08-28' -> {simplify_label_string('08-28')}")
# print(f"'Address' -> {simplify_label_string('Address')}")
# print(f"'City' -> {simplify_label_string('City')}")
# print(f"'Hospital as x' -> {simplify_label_string('Hospital')}")
# print(f"'PatientID' -> {simplify_label_string('PatientID')}") # Example that should return O
# print(f"'2023-12-31' -> {simplify_label_string('2023-12-31')}")
# print(f"'AnotherLabel' -> {simplify_label_string('AnotherLabel')}")
