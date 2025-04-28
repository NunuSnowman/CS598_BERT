import dataclasses
from typing import List

@dataclasses.dataclass
class MaskInfo:
    """Represents a single masked segment found and its corresponding text."""
    start: int          # Start index in the original text record (inclusive)
    end: int            # End index in the original text record (exclusive)
    label: str          # The content inside the mask, e.g., "First Name"
    text: str           # The matched text from the original record
    masked_text: str    # The original mask string from the res record, e.g., "[**First Name**]"

@dataclasses.dataclass
class ProcessedRecord:
    """Represents a single processed record with original text, masked text, and mask details."""
    res_record: str         # The masked text of the record
    text_record: str        # The original text of the record
    mask_info: List[MaskInfo] # A list of MaskResult objects detailing the masks applied