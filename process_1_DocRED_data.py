import json
import os
import re

# --- Configuration ---
# Set these paths according to your environment
in_path = "downloaded/DocRED/data" # Path to the directory containing DocRED data files.
out_path = "data/DocRED" # Path to the directory to save processed data.
data_file_name = "train_distant.json" # Name of the DocRED data file to process.
output_file_name = "processed_data.jsonl" # Name of the output JSONL file.
# --- End Configuration ---

# Construct the full path to the input data file and output file
data_file = os.path.join(in_path, data_file_name)
output_file = os.path.join(out_path, output_file_name)

# Ensure the output directory exists
os.makedirs(out_path, exist_ok=True)

def pseudo_de_identify_docred_single_file(data_file, output_file):
    """
    Reads a DocRED data file, masks entities (PER, LOC, TIME) with item-specific IDs
    based on vertex spans and sentence IDs, calculates correct offsets,
     and saves the original text, masked text, and mask details to a single JSONL file.

    Handles the DocRED format where 'pos' is a list of [start, end+1] pairs
    and 'sent_id' is a separate key for the vertex mention.

    Args:
        data_file (str): Path to the input DocRED JSON file.
        output_file (str): Path to the output JSONL file.
    """
    print(f"Attempting to load data from: {data_file}")
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {data_file}. Please ensure the file exists at this path.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {data_file}. Ensure it's a valid JSON file.")
        return

    processed_records = []

    print(f"Processing {len(data)} records...")

    for item_idx, item in enumerate(data):
        sents = item.get('sents', [])
        vertex_set = item.get('vertexSet', [])

        # --- Step 1: Reconstruct Original Text and Calculate Token Offsets ---
        # This part remains the same, it correctly builds text and token offsets
        original_item_text = ""
        token_char_offsets = [] # List of lists: token_char_offsets[sent_idx][token_idx] = (start_char, end_char)
        current_offset = 0

        for sent_idx, sentence in enumerate(sents):
            sentence_token_offsets = []
            for token_idx, token in enumerate(sentence):
                # Add space before token if it's not the very first token in the item
                if current_offset > 0:
                    original_item_text += ' '
                    current_offset += 1

                token_start = current_offset
                original_item_text += token
                token_end = current_offset + len(token)
                sentence_token_offsets.append((token_start, token_end))
                current_offset = token_end

            token_char_offsets.append(sentence_token_offsets)

        # --- Step 2: Prepare Entity Masking Information Map (Corrected) ---
        # Map (sent_idx, start_token_idx, end_token_idx) -> {masked_token, original_name, entity_type, mention_id}
        entity_mask_map = {}
        mention_counter = 1 # Counter for generating unique IDs for each mention

        for vertex_list in vertex_set:
            for vertex in vertex_list:
                entity_type = vertex.get('type')
                original_name = vertex.get('name')

                # --- Correctly get sent_id and pos list for this mention ---
                mention_sent_id = vertex.get('sent_id')
                mention_pos_list = vertex.get('pos') # This should be the list like [[start, end+1]]

                # We only mask PER, LOC, TIME and need valid sent_id and pos list
                mapping = {
                    'PER': 'NAME',
                    'LOC': 'LOCATION',
                    'TIME': 'DATE'
                }
                if entity_type in ['PER', 'LOC', 'TIME']:
                    # Validate sent_id
                    if not isinstance(mention_sent_id, int) or mention_sent_id < 0 or mention_sent_id >= len(sents):
                        print(f"Warning: Skipping vertex '{original_name}' with invalid 'sent_id': {mention_sent_id}")
                        continue

                    # Validate pos list
                    if not isinstance(mention_pos_list, list):
                        print(f"Warning: Skipping vertex '{original_name}' (sent_id: {mention_sent_id}) with non-list 'pos' value: {mention_pos_list}")
                        continue

                    # Iterate through the spans *within* the 'pos' list for this mention

                        # If the span format is correct, extract start and end token indices
                    start_token_idx = mention_pos_list[0] # DocRED format: [start_token, end_token + 1]
                    end_token_idx = mention_pos_list[1] - 1 # Adjust to be inclusive end token index

                    # Validate token indices against sentence length
                    if start_token_idx < 0 or end_token_idx >= len(sents[mention_sent_id]) or start_token_idx > end_token_idx:
                        print(f"Warning: Skipping entity '{original_name}' (type: {entity_type}) in sent {mention_sent_id} with invalid token indices: [{start_token_idx}, {end_token_idx+1}]")
                        continue


                    # Key based on the mention's full position (sent_idx, start, end)
                    mention_key = (mention_sent_id, start_token_idx, end_token_idx)

                    # Generate a unique masked token for THIS mention
                    masked_token_str = f"[**{mapping[entity_type.upper()]} {mention_counter}**]"

                    # Store masking info mapped to the mention's span
                    # Check for potential duplicate keys if the same span is listed multiple times for the same vertex (shouldn't happen in standard data but good safety)
                    if mention_key in entity_mask_map:
                        # This case means the exact same token span in the exact same sentence
                        # is listed multiple times for the same entity vertex mention.
                        # We'll just overwrite, but it indicates unusual data.
                        print(f"Warning: Duplicate mention key found for entity '{original_name}' at {mention_key}. Overwriting.")


                    entity_mask_map[mention_key] = {
                        "masked_token_str": masked_token_str,
                        "original_name": original_name,
                        "entity_type": entity_type,
                        "mention_id": mention_counter # Store the counter for potential use in label
                    }
                    mention_counter += 1 # Increment for the next valid mention span


        # --- Step 3: Perform Masking and Generate Mask Details ---
        # This part uses the entity_mask_map and token_char_offsets, and remains similar
        masked_sents = []
        masks = []

        for sent_idx, sentence in enumerate(sents):
            masked_sentence = []
            token_idx_in_sentence = 0
            while token_idx_in_sentence < len(sentence):
                # Check if the current token is the start of ANY mention span in our map
                current_span_key = None
                mask_info = None
                span_end_token_idx = token_idx_in_sentence - 1 # Initialize end index before checking

                # Iterate through map keys to find a matching span starting here
                # We check all spans for efficiency, though ideally we'd only check spans in the current sentence
                # The key lookup `entity_mask_map.get((sent_idx, token_idx_in_sentence, end_candidate), None)` could be more direct if we knew the end,
                # but iterating keys allows finding any valid span starting here.
                # A more optimized approach might group entity_mask_map by (sent_idx, start_token_idx)
                found_span_key = None
                for span_key in entity_mask_map:
                    s_idx, start_t_idx, end_t_idx = span_key
                    # Check if this span belongs to the current sentence and starts at the current token index
                    if s_idx == sent_idx and start_t_idx == token_idx_in_sentence:
                        # Found a mention span starting here
                        found_span_key = span_key
                        mask_info = entity_mask_map[found_span_key]
                        span_end_token_idx = end_t_idx # This is the inclusive end token index
                        # If there are overlapping spans starting at the same token,
                        # this will pick one based on dictionary iteration order.
                        # For simplicity, we'll just take the first one found.
                        break # Found a span starting here, no need to check others for this token_idx_in_sentence


                if found_span_key is not None:
                    # We are at the start of a mention span
                    char_start = token_char_offsets[sent_idx][token_idx_in_sentence][0]
                    char_end = token_char_offsets[sent_idx][span_end_token_idx][1] # End of the last token in the span

                    # Add the masked token to the masked sentence
                    masked_sentence.append(mask_info["masked_token_str"])

                    # Create the mask detail
                    mask_detail = {
                        "label": f"{mask_info['entity_type'].upper()} {mask_info['mention_id']}", # Use entity type and assigned mention ID
                        "text": mask_info["original_name"],   # Original entity name
                        "start": char_start,                 # Character start offset
                        "end": char_end,                     # Character end offset
                        "masked_text": mask_info["masked_token_str"] # The generated mask string
                    }
                    masks.append(mask_detail)

                    # Advance the token_idx_in_sentence past the end of the masked span
                    token_idx_in_sentence = span_end_token_idx + 1

                else:
                    # If the current token is NOT the start of a mention span, keep the original token
                    masked_sentence.append(sentence[token_idx_in_sentence])
                    token_idx_in_sentence += 1 # Move to the next token

            masked_sents.append(' '.join(masked_sentence))


        masked_item_text = ' '.join(masked_sents)

        # --- Construct the output JSON object for this record ---
        record_json = {
            "res_record": masked_item_text,
            "text_record": original_item_text,
            "mask_info": masks
        }
        processed_records.append(record_json)

        # Optional: Print progress
        if (item_idx + 1) % 1000 == 0:
            print(f"Processed {item_idx + 1}/{len(data)} records.")

    # --- Saving the processed data to a single JSONL file ---
    print(f"\nSaving processed data to: {output_file}")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for record in processed_records:
                json.dump(record, f, ensure_ascii=False)
                f.write('\n')
        print(f"\nProcessing complete. Processed data saved to: {output_file}")
    except IOError as e:
        print(f"Error: Could not write to output file {output_file}. Reason: {e}")


if __name__ == "__main__":
    # --- Execute the function ---
    pseudo_de_identify_docred_single_file(data_file, output_file)