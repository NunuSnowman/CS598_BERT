import json
import re

# Regex to find the masked tokens like [**xxx**]
MASK_TOKEN_PATTERN = re.compile(r'\[\*\*.*?\]')

def verify_single_record_mapping(original_text: str, masked_text: str, mapping_dict: dict) -> bool:
    """
    Verifies if the mapping for a single record (line) can correctly reconstruct the original text.

    Args:
        original_text: The original text string for one record.
        masked_text: The masked text string for the same record (containing [**xxx**] tokens).
        mapping_dict: The corresponding mapping dictionary for this record.

    Returns:
        Returns True if the mapping can correctly reconstruct the original text from the masked text,
        otherwise returns False. This function does not print error messages, it only returns the
        verification result.
    """
    # --- Verification Step 1: Check if all tokens in masked_text are in the mapping ---
    tokens_in_masked = set(MASK_TOKEN_PATTERN.findall(masked_text))
    tokens_in_mapping_keys = set(mapping_dict.keys())

    if tokens_in_masked - tokens_in_mapping_keys:
        # Tokens found in masked text but not present as keys in the mapping
        return False
    #
    # # --- Verification Step 2: Check if all keys in mapping are present as tokens in masked_text ---
    # if tokens_in_mapping_keys - tokens_in_masked:
    #     # Tokens found as keys in the mapping but not present in the masked text
    #     return False

    # --- Verification Step 3: Attempt full reconstruction using the mapping and compare with original ---
    reconstructed_text = masked_text
    # Replace each masked token with its original value from the mapping
    for mask_token, original_value in mapping_dict.items():
        # Use replace() instead of re.sub() for exact string matching of tokens
        reconstructed_text = reconstructed_text.replace(mask_token, str(original_value)) # Ensure value is string for replacement

    # Compare the reconstructed line with the original line
    if reconstructed_text != original_text:
        # Reconstruction does not match the original text
        return False

    # If all checks pass
    return True

# Note: JSON parsing of the mapping string into a dictionary should happen BEFORE calling this function,
# as this function expects a dictionary as input.