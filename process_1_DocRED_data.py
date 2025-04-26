import json
import os
import argparse

# Set up argument parser for input and output paths
parser = argparse.ArgumentParser(description='Pseudo de-identify DocRED data by masking person names.')
parser.add_argument('--in_path', type=str, default="downloaded/DocRED/data", help='Path to the directory containing DocRED data files.')
# Using the requested output path
parser.add_argument('--out_path', type=str, default="data/DocREDv2", help='Path to the directory to save processed data.')
parser.add_argument('--data_file_name', type=str, default="train_distant.json", help='Name of the DocRED data file to process (e.g., train_distant.json, dev.json).')

args = parser.parse_args()

in_path = args.in_path
out_path = args.out_path # Use the potentially overridden out_path from args
data_file_name = args.data_file_name

# Construct the full path to the input data file
data_file = os.path.join(in_path, data_file_name)

# Ensure the output directory exists
os.makedirs(out_path, exist_ok=True)

def pseudo_de_identify_docred(data_file, out_path):
    """
    Reads a DocRED data file, masks person names with item-specific IDs,
    saves the original text, masked text, and name mapping per item.

    Args:
        data_file (str): Path to the input DocRED JSON file.
        out_path (str): Path to the directory to save the output files.
    """
    print(f"Loading data from: {data_file}")
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {data_file}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {data_file}. Ensure it's a valid JSON file.")
        return

    # Lists to store data for each record (item)
    original_texts = [] # New list for original texts
    masked_texts = []
    item_name_mappings = []

    print(f"Processing {len(data)} records...")

    # Process each record (item) in the dataset
    for item_idx, item in enumerate(data):
        # DocRED data structure: item['sents'] is a list of sentences,
        # each sentence is a list of words.
        sents = item.get('sents', [])
        # item['vertexSet'] contains lists of entities (vertices)
        vertex_set = item.get('vertexSet', [])

        # --- Prepare Original Text ---
        # Join the original sentences into a single string for this item
        # sents is a list of lists of words, e.g., [['Word1', 'Word2'], ['Word3', 'Word4']]
        # First join words in each sentence, then join sentences with a space
        original_item_text = ' '.join([' '.join(s) for s in sents])
        original_texts.append(original_item_text)


        # --- Logic for item-specific name mapping and masking ---
        item_name_mapping = {} # Mapping for names in THIS item: Masked Token -> Original Name
        used_names_in_item = {} # Mapping for names in THIS item: Original Name -> Masked Token
        item_name_counter = 1 # Counter for generating IDs within THIS item

        # First pass: identify all unique person names in this item and assign masked tokens
        # We use the original words from sents to find matches within vertex names
        all_words_in_item = [word for sentence in sents for word in sentence]

        for vertex_list in vertex_set:
            for vertex in vertex_list:
                # Check if the entity type is 'PER' (Person)
                # Note: Vertex names might span multiple words or match substrings.
                # This simple approach matches exact word tokens found in sentences
                # if they are also listed as a vertex name. More complex matching
                # might be needed for full de-identification but this matches the original script's logic.
                if vertex.get('type') == 'PER':
                    name = vertex.get('name')
                    # Check if the name is a single word and appears in the sentences
                    # and has not been assigned an ID yet for this item
                    if name and ' ' not in name and name in all_words_in_item and name not in used_names_in_item:
                        # Assign a new ID unique to this item
                        name_id = f"{item_name_counter}"
                        # Create the masked token string
                        masked_token = f"[***NAME {name_id}***]"
                        # Store mapping from masked token to original name
                        item_name_mapping[masked_token] = name
                        # Store mapping from original name to masked token for quick lookup during masking
                        used_names_in_item[name] = masked_token
                        item_name_counter += 1


        # Second pass: mask the names in the sentences using item-specific masked tokens
        masked_sents = []
        for sentence in sents:
            # Create a copy of the sentence (list of words) to modify
            masked_sentence = sentence[:]
            for i, word in enumerate(sentence):
                # Check if the word is one of the person names identified in this item
                if word in used_names_in_item:
                    # Get the pre-generated masked token for this name
                    masked_token = used_names_in_item[word]
                    # Replace the word with the masked token
                    masked_sentence[i] = masked_token
            # Join the words in the masked sentence back into a string
            masked_sents.append(' '.join(masked_sentence))

        # Join all masked sentences for this item into a single string
        masked_texts.append(' '.join(masked_sents))

        # Store the name mapping specific to this item
        item_name_mappings.append(item_name_mapping)

        # Optional: Print progress
        if (item_idx + 1) % 1000 == 0:
            print(f"Processed {item_idx + 1}/{len(data)} records.")


    # --- Saving the processed data ---

    # Save original texts, one record per line (NEW FILE)
    original_text_file = os.path.join(out_path, 'original_data.txt')
    print(f"\nSaving original data to: {original_text_file}")
    with open(original_text_file, 'w', encoding='utf-8') as f:
        for line in original_texts:
            f.write(line + '\n')

    # Save masked texts, one record per line (Existing file, potentially renamed for clarity later if needed)
    masked_text_file = os.path.join(out_path, 'masked_data.txt') # Keeping original name for now
    print(f"Saving masked data to: {masked_text_file}")
    with open(masked_text_file, 'w', encoding='utf-8') as f:
        for line in masked_texts:
            f.write(line + '\n')

    # Save name mappings, one JSON object per record per line (JSON Lines format)
    name_mapping_file = os.path.join(out_path, 'mapping.jsonl') # Keeping original name for now
    print(f"Saving name mappings to: {name_mapping_file}")
    with open(name_mapping_file, 'w', encoding='utf-8') as f:
        for mapping in item_name_mappings:
            json.dump(mapping, f, ensure_ascii=False) # ensure_ascii=False to handle non-ASCII names
            f.write('\n')

    print("\nProcessing complete.")
    print(f"Original data saved to: {original_text_file}")
    print(f"Masked data saved to: {masked_text_file}")
    print(f"Name mapping saved to: {name_mapping_file}")

# --- Execute the function ---
pseudo_de_identify_docred(data_file, out_path)