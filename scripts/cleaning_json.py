import json
import re

# 1. Config
INPUT_FILE = "synthethic_data.txt"
OUTPUT_FILE = "llama3_multilingual_data.jsonl"


def extract_json_objects(text):
    """
    GPT-4 sometimes puts text before/after the JSON.
    This finds the list [...] or individual objects {...}
    """
    json_objects = []

    # Try to find a JSON list block first (starts with [ and ends with ])
    # DOTALL allows . to match newlines
    list_match = re.search(r'\[\s*\{.*?\}\s*\]', text, re.DOTALL)

    if list_match:
        try:
            # Parse the full list found in the text
            data = json.loads(list_match.group(0))
            if isinstance(data, list):
                json_objects.extend(data)
        except json.JSONDecodeError:
            print("Found a list block but failed to parse it. Trying individual objects...")

    # If no list found or parsing failed, try finding individual JSON objects
    if not json_objects:
        # Regex to find separate JSON objects { ... }
        # This is a bit complex to handle nested braces, but works for standard GPT outputs
        matches = re.findall(r'\{[^{}]*\}', text)
        # Note: The simple regex above fails on nested JSON strings.
        # Since our "input_json" is a stringified JSON, we need a smarter parser.

        # BETTER APPROACH: Iterative JSON decoder
        decoder = json.JSONDecoder()
        pos = 0
        while True:
            # Skip whitespace to find the next '{' or '['
            match = re.search(r'[\{\[]', text[pos:])
            if not match:
                break
            start_index = pos + match.start()

            try:
                obj, end_index = decoder.raw_decode(text[start_index:])
                if isinstance(obj, list):
                    json_objects.extend(obj)
                elif isinstance(obj, dict):
                    json_objects.append(obj)
                pos = start_index + end_index
            except json.JSONDecodeError:
                pos = start_index + 1  # Skip bad char and try again

    return json_objects


# 2. Process
print(f"Reading {INPUT_FILE}...")
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    raw_text = f.read()

data_objects = extract_json_objects(raw_text)

print(f"Found {len(data_objects)} valid JSON objects.")

# 3. Write to JSONL
# We need strict format for the Colab script:
# {"input_json": "...", "lang_en": "...", "lang_tr": "..."}
valid_count = 0

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for item in data_objects:
        # Validation: Check if required keys exist
        if "input_json" in item and "lang_en" in item and "lang_tr" in item:
            # Dump as a single line
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            valid_count += 1
        else:
            print(f"Skipping invalid item: {item.keys()}")

print(f"Successfully converted {valid_count} items to {OUTPUT_FILE}")
print("You can now upload this file to Colab.")