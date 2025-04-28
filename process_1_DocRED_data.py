import json
import os
import re

# --- Configuration ---
# Set these paths according to your environment
in_path = "downloaded/DocRED/data" # Path to the directory containing DocRED data files.
out_path = "data/DocREDv2" # Path to the directory to save processed data.
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
    Reads a DocRED data file, masks person names with item-specific IDs,
    and saves the original text, masked text, and mask details to a single JSONL file.

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

        # --- Prepare Original Text ---
        original_item_text = ' '.join([' '.join(s) for s in sents])

        # --- Logic for item-specific name mapping and masking ---
        item_name_mapping = {} # Mapping for names in THIS item: Masked Token -> Original Name
        used_names_in_item = {} # Mapping for names in THIS item: Original Name -> Masked Token
        item_name_counter = 1 # Counter for generating IDs within THIS item

        all_words_in_item = [word for sentence in sents for word in sentence]

        for vertex_list in vertex_set:
            for vertex in vertex_list:
                if vertex.get('type') == 'PER':
                    name = vertex.get('name')
                    # Check if the name is a single word and appears in the sentences
                    # and has not been assigned an ID yet for this item
                    if name and ' ' not in name and name in all_words_in_item and name not in used_names_in_item:
                        name_id = f"{item_name_counter}"
                        masked_token = f"[***NAME {name_id}***]"
                        item_name_mapping[masked_token] = name
                        used_names_in_item[name] = masked_token
                        item_name_counter += 1

        # --- Masking and Generating Mask Details ---
        masked_sents = []
        masks = []
        current_offset = 0 # Keep track of the current index in the original text

        for sentence in sents:
            masked_sentence = []
            # Join words with a space, handling potential empty sentences or words
            sentence_text = ' '.join(word for word in sentence)
            word_offset = 0 # Keep track of index within the current sentence_text

            for i, word in enumerate(sentence):
                # Find the exact word in the sentence_text starting from the current word_offset
                original_word_start_in_sentence = sentence_text.find(word, word_offset)

                if original_word_start_in_sentence != -1:
                    original_word_end_in_sentence = original_word_start_in_sentence + len(word)

                    if word in used_names_in_item:
                        masked_token = used_names_in_item[word]
                        # Create the mask detail corresponding to the MaskResult structure
                        mask_detail = {
                            "label": word,          # The content inside the mask (original name)
                            "text": word,           # The matched text from the original record (original name)
                            "start": current_offset + original_word_start_in_sentence, # Start index in the original text record
                            "end": current_offset + original_word_end_in_sentence,     # End index in the original text record
                            "masked_text": masked_token # The original mask string
                        }
                        masks.append(mask_detail)
                        masked_sentence.append(masked_token)
                    else:
                        masked_sentence.append(word)

                    # Update word_offset for the next search within the sentence_text
                    word_offset = original_word_end_in_sentence + (1 if i < len(sentence) - 1 else 0) # +1 for space if not the last word
                else:
                    # If the word was not found, append it as is and move offset past it
                    masked_sentence.append(word)
                    word_offset += len(word) + (1 if i < len(sentence) - 1 else 0)


            masked_sents.append(' '.join(masked_sentence))
            # Update current_offset for the next sentence
            current_offset += len(sentence_text) + (1 if sents.index(sentence) < len(sents) - 1 else 0) # +1 for space between sentences if not the last sentence


        masked_item_text = ' '.join(masked_sents)

        # --- Construct the output JSON object for this record ---
        record_json = {
            "res_record": masked_item_text,
            "text_record": original_item_text,
            "masks": masks
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


# --- Execute the function ---
pseudo_de_identify_docred_single_file(data_file, output_file)