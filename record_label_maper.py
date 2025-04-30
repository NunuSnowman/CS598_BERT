import re
from typing import List, Dict, Tuple, Union

from common import ProcessedRecord
LABEL_MEMBERSHIP : Dict[str, List[Union[str, List[str]]]]= {
    'simple':
        [
            [
                'NAME',
                [
                    'NAME',
                    'DOCTOR',
                    'PATIENT',
                    'USERNAME',
                    'HCPNAME',
                    'RELATIVEPROXYNAME',
                    'PTNAME',
                    'PTNAMEINITIAL',
                    'KEYVALUE',
                ]
            ], ['PROFESSION', ['PROFESSION']],
            [
                'LOCATION',
                [
                    'LOCATION', 'HOSPITAL', 'ORGANIZATION', 'URL', 'STREET',
                    'STATE', 'CITY', 'COUNTRY', 'ZIP', 'LOCATION-OTHER',
                    'PROTECTED_ENTITY', 'PROTECTED ENTITY', 'NATIONALITY'
                ]
            ], ['AGE', ['AGE', 'AGE_>_89', 'AGE > 89']],
            ['DATE', ['DATE', 'DATEYEAR']],
            [
                'ID',
                [
                    'BIOID', 'DEVICE', 'HEALTHPLAN', 'IDNUM', 'MEDICALRECORD',
                    'ID', 'IDENTIFIER', 'OTHER'
                ]
            ],
            [
                'CONTACT',
                ['EMAIL', 'FAX', 'PHONE', 'CONTACT', 'IPADDR', 'IPADDRESS']
            ], ['O', ['O']]
        ],
    'hipaa':
        [
            [
                'NAME',
                [
                    'NAME',
                    'PATIENT',
                    'USERNAME',
                    'RELATIVEPROXYNAME',
                    'PTNAME',
                    'PTNAMEINITIAL',
                    'KEYVALUE',
                ]
            ],
            [
                'LOCATION',
                [
                    'LOCATION', 'ORGANIZATION', 'HOSPITAL', 'STREET', 'CITY',
                    'ZIP', 'URL', 'PROTECTED_ENTITY', 'PROTECTED ENTITY',
                    'LOCATION-OTHER'
                ]
            ],
            ['AGE', ['AGE', 'AGE_>_89', 'AGE > 89']],
            ['DATE', ['DATE', 'DATEYEAR']],
            [
                'ID',
                [
                    'BIOID', 'DEVICE', 'HEALTHPLAN', 'IDNUM', 'MEDICALRECORD',
                    'ID', 'IDENTIFIER', 'OTHER'
                ]
            ],
            [
                'CONTACT',
                [
                    'EMAIL',
                    'FAX',
                    'PHONE',
                    'CONTACT',
                    # it is unclear whether these are HIPAA in i2b2 paper
                    'IPADDR',
                    'IPADDRESS'
                ]
            ],
            [
                'O',
                [
                    'DOCTOR', 'HCPNAME', 'PROFESSION', 'STATE', 'COUNTRY',
                    'NATIONALITY', 'O'
                ]
            ]
        ],
    'binary':
        [
            [
                'PHI',
                [
                    'NAME',
                    'PATIENT',
                    'USERNAME',
                    'RELATIVEPROXYNAME',
                    'PTNAME',
                    'PTNAMEINITIAL',
                    'DOCTOR',
                    'KEYVALUE',
                    'HCPNAME',
                    'LOCATION',
                    'ORGANIZATION',
                    'HOSPITAL',
                    'PROTECTED_ENTITY',
                    'PROTECTED ENTITY',
                    'LOCATION-OTHER',
                    'STREET',
                    'CITY',
                    'ZIP',
                    'STATE',
                    'COUNTRY',
                    'NATIONALITY',
                    # two URLs in i2b2 which aren't URLs but web service names
                    'URL',
                    'AGE',
                    'AGE_>_89',
                    'AGE > 89',
                    'DATE',
                    'DATEYEAR',
                    'BIOID',
                    'DEVICE',
                    'HEALTHPLAN',
                    'IDNUM',
                    'MEDICALRECORD',
                    'ID',
                    'IDENTIFIER',
                    'OTHER',
                    'EMAIL',
                    'FAX',
                    'PHONE',
                    'CONTACT',
                    'PROFESSION',
                    'IPADDR',
                    'IPADDRESS'
                ]
            ],
            ['O', ['O']]
        ]
}

def simplify_record_labels(records: List[ProcessedRecord]) -> List[ProcessedRecord]:
    print(f"Simplifying labels for {len(records)} records...")
    for record in records:
        for mask_info_item in record.mask_info:
            original_label = mask_info_item.label
            simplified_label = simplify_label_string(original_label)
            mask_info_item.label = simplified_label
    return records

def simplify_label_string(label: str, mapping_group: str = "simple") -> str:
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

    if label_upper == "O":
        return "O"

    def is_valid_date_string(s: str) -> bool:
        pattern = r'^(?:\d{1,4}-){0,2}\d{1,4}$'
        if re.match(pattern, s):
            digits_only = re.sub(r'\D', '', s)
            return 3 <= len(digits_only) <= 8
        return False

    if is_valid_date_string(label_upper):
        return "DATE"

    # Rule 2: Check for LOCATION keywords
    # Add more keywords to this set to extend LOCATION mapping
    for high_level, low_level_list in LABEL_MEMBERSHIP[mapping_group]:
        for sub_label in low_level_list:
            if sub_label in label_upper:
                return high_level

    # Default Rule: If none of the above match, return "O"
    return "O"

if __name__ == "__main__":
    print(f"'B-NAME' -> {simplify_label_string('B-NAME')}")
    print(f"'I-LOCATION' -> {simplify_label_string('I-LOCATION')}")
    print(f"'O' -> {simplify_label_string('O')}")
    print(f"'08-28' -> {simplify_label_string('08-28')}")
    print(f"'Address' -> {simplify_label_string('Address')}")
    print(f"'City' -> {simplify_label_string('City')}")
    print(f"'Hospital as x' -> {simplify_label_string('Hospital')}")
    print(f"'PatientID' -> {simplify_label_string('PatientID')}") # Example that should return O
    print(f"'2023-12-31' -> {simplify_label_string('2023-12-31')}")
    print(f"'AnotherLabel' -> {simplify_label_string('AnotherLabel')}")
