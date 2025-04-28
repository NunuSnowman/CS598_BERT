import dataclasses


@dataclasses.dataclass
class MaskResult:
    """Represents a single masked segment found and its corresponding text."""
    label: str          # The content inside the mask, e.g., "First Name"
    text: str           # The matched text from the original record
    start: int          # Start index in the original text record (inclusive)
    end: int            # End index in the original text record (exclusive)
    masked_text: str    # The original mask string from the res record, e.g., "[**First Name**]"

