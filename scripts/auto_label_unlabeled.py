import re
import os
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

# 1. Define your custom labels (MUST match the order used during fine-tuning)
label_list = [
    "O",
    "B-Product", "I-Product",
    "B-LOC", "I-LOC",
    "B-PRICE", "I-PRICE"
]
id2label = {i: label for i, label in enumerate(label_list)}
label2id = {label: i for i, label in enumerate(label_list)}

# 2. Load YOUR Fine-tuned Model and Tokenizer
model_path = "./models/your_finetuned_model"

try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path, id2label=id2label, label2id=label2id)
    ner_pipeline = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        grouped_entities=True,
        aggregation_strategy="simple",
        device=0 if torch.cuda.is_available() else -1
    )
    print(f"NER pipeline initialized successfully. Using {'GPU' if torch.cuda.is_available() else 'CPU'}.")
except Exception as e:
    print(f"Error loading fine-tuned model or initializing pipeline: {e}")
    print("Falling back to a default pipeline or check your model path/GPU setup.")
    tokenizer = AutoTokenizer.from_pretrained("Davlan/xlm-roberta-base-ner-hrl")
    model = AutoModelForTokenClassification.from_pretrained("Davlan/xlm-roberta-base-ner-hrl")
    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=True, aggregation_strategy="simple")
    print("Initialized with default 'Davlan/xlm-roberta-base-ner-hrl' model as fallback.")


def read_unlabeled_conll_generator(filepath):
    """
    Read unlabeled CoNLL text file as a generator, yielding sentences (list of tokens).
    """
    current_sentence = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_sentence:
                    yield current_sentence
                    current_sentence = []
            else:
                parts = line.split(' ')
                token = parts[0]
                current_sentence.append(token)
        if current_sentence:
            yield current_sentence

def ner_label_sentence(sentence, ner_pipeline):
    """
    Label tokens in a sentence using the NER pipeline.
    Returns a list of (token, label) tuples in B-/I-/O scheme.
    """
    if not sentence:
        return []

    text = " ".join(sentence)
    labels = ["O"] * len(sentence)

    try:
        ner_results = ner_pipeline(text)
    except Exception as e:
        print(f"NER pipeline failed for text: '{text[:100]}...'. Error: {e}")
        return [(token, "O") for token in sentence]

    tokens_char_offsets = []
    current_char_idx = 0
    for token in sentence:
        start_idx = text.find(token, current_char_idx)
        if start_idx == -1:
            tokens_char_offsets.append((None, None))
            current_char_idx += len(token) + 1
            continue
        end_idx = start_idx + len(token)
        tokens_char_offsets.append((start_idx, end_idx))
        current_char_idx = end_idx + 1

    for entity in ner_results:
        custom_label = entity['entity_group']
        ent_start = entity.get('start')
        ent_end = entity.get('end')

        if custom_label not in [lbl.replace('B-', '').replace('I-', '') for lbl in label_list if lbl != 'O']:
            continue

        if custom_label == "O":
            continue

        inside_tokens_indices = []
        for token_idx, (tok_start, tok_end) in enumerate(tokens_char_offsets):
            if tok_start is None or tok_end is None:
                continue

            if max(tok_start, ent_start) < min(tok_end, ent_end):
                inside_tokens_indices.append(token_idx)

        for i, token_idx in enumerate(sorted(list(set(inside_tokens_indices)))):
            if token_idx < len(labels):
                prefix = "B-" if i == 0 or \
                         (token_idx > 0 and labels[token_idx-1] not in [f"B-{custom_label}", f"I-{custom_label}"]) \
                         else "I-"
                if (prefix + custom_label) in label_list:
                    labels[token_idx] = prefix + custom_label
                else:
                    labels[token_idx] = "O"

    # --- RULE-BASED OVERRIDES/FALLBACKS (Specific and Temporary) ---
    combined_text = " ".join(sentence)
    
    # Updated Product Patterns (Brand and Model specific)
    product_patterns = [
        r"Imitation Volcano Humidifier with LED Light",
        r"Baby Carrier",
        r"Smart Usb Ultrasonic Car And Home Air Humidifier With Colorful Led Light Original Highquality",
        r"Baby Head Helmet Cotton Walk Safety Hat Breathable Headgear Toddler Antifall Pad",
        r"Green Lion Air Mist Humidifier",
        r"Acer Nitro 5",
        r"Lenovo ThinkPad X1 Extreme Gen 2",
        r"DELL G3",
        r"HP VICTUS 16",
        r"ASUS VIVOBOOK",
        r"HP ENVY X360",
        r"HP SPECTER",
        r"MICROSOFT LAPTOP FIVE5",
        r"MAC BOOK AIR",
        r"HP OMEN 016",
        r"DELL 14 FULL HD",
        # Adding more specific patterns from the latest examples
        r"GAMINGLAPTOP DELL G3", # Specific for the given format
        r"EUROPE STANDARD HP VICTUS 16",
        r"EUROPE STANDARD ASUS VIVOBOOK",
        r"EUROPE STANDARD HP ENVY X360",
        r"EUROPE STANDARD HP SPECTER",
        r"EUROPE STANDARD DELL 14 FULL HD",
        r"GAMING LAPTOP HP OMEN 016",
        r"EUROPE STANDARD MICROSOFT LAPTOP FIVE5",
        r"EUROPE STANDARD MAC BOOK AIR",
        r"LENOVO thinkpad p50", # Shorter version for the product name
        r"HP PAVILION X360", # Shorter version for the product name
        r"LENOVO LOQ", # Shorter version for the product name
        r"RAZER BLADE 18", # Specific Razer product
        r"Mattress PROTECTORPOLYESTER MICROFIBER "
        r"9th generation workstation laptop USA STANDARD LENOVO thinkpad p50 156 4k screen CORE I7 8th gen 12 cpu 16GB 𝗗𝗗𝗥4 512 GB SSD 4GB NIVIDIA QUADRO T2000" # Full string from example
    ]
    
    for pattern in product_patterns:
        for match in re.finditer(pattern, combined_text, re.IGNORECASE):
            match_start_char, match_end_char = match.span()
            matched_tokens_indices = []
            for token_idx, (tok_start, tok_end) in enumerate(tokens_char_offsets):
                if tok_start is not None and tok_end is not None:
                    if max(tok_start, match_start_char) < min(tok_end, match_end_char):
                        matched_tokens_indices.append(token_idx)
            
            for i, token_idx in enumerate(sorted(list(set(matched_tokens_indices)))):
                if token_idx < len(labels):
                    if labels[token_idx] == "O" or not labels[token_idx].endswith("-Product"):
                        prefix = "B-" if i == 0 else "I-"
                        labels[token_idx] = prefix + "Product"

    # Specific Location Patterns (retained)
    location_patterns = [
        r"መገናኛ_መሰረት_ደፋር_ሞል_ሁለተኛ_ፎቅ",
        r"መገናኛ ማራቶን የ ገበያ ማእከል በ ዋናው መግቢያ መሬት ላይ ወይንም ግራውንድ ፍሎር"
    ]

    for pattern in location_patterns:
        for match in re.finditer(pattern, combined_text):
            match_start_char, match_end_char = match.span()
            matched_tokens_indices = []
            for token_idx, (tok_start, tok_end) in enumerate(tokens_char_offsets):
                if tok_start is not None and tok_end is not None:
                    if max(tok_start, match_start_char) < min(tok_end, match_end_char):
                        matched_tokens_indices.append(token_idx)
            
            for i, token_idx in enumerate(sorted(list(set(matched_tokens_indices)))):
                if token_idx < len(labels):
                    if labels[token_idx] == "O" or not labels[token_idx].endswith("-LOC"):
                        prefix = "B-" if i == 0 else "I-"
                        labels[token_idx] = prefix + "LOC"

    # Price regex (existing, retained)
    for i in range(len(sentence)):
        if labels[i] == "O":
            if i + 1 < len(sentence) and \
               re.match(r"^\d+(,\d+)*(\.\d+)?$", sentence[i]) and \
               sentence[i+1].lower() == "ብር":
                labels[i] = "B-PRICE"
                labels[i+1] = "I-PRICE"
            elif re.match(r"^\d+(,\d+)*(\.\d+)?ብር$", sentence[i]):
                labels[i] = "B-PRICE"
            
    return list(zip(sentence, labels))

def write_labeled_conll_chunked(labeled_sentences_generator, output_path, chunk_size=1000):
    """
    Write labeled sentences to file in CoNLL format, writing in chunks
    to avoid holding everything in memory.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        current_chunk = []
        for sent in labeled_sentences_generator:
            for token, label in sent:
                current_chunk.append(f"{token} {label}\n")
            current_chunk.append("\n")
            
            if len(current_chunk) >= chunk_size:
                f.writelines(current_chunk)
                current_chunk = []
        if current_chunk:
            f.writelines(current_chunk)

def main():
    print("Initializing NER pipeline with fine-tuned model...")

    unlabeled_filepath = "data/clean/unlabeled_conll.txt"
    if not os.path.exists(unlabeled_filepath):
        print(f"Error: Input file not found at {unlabeled_filepath}")
        print("Please ensure 'data/clean/unlabeled_conll.txt' exists and is correctly formatted (one token per line, blank lines for sentence breaks).")
        return

    unlabeled_sentences_generator = read_unlabeled_conll_generator(unlabeled_filepath)

    output_filepath = "auto_labeled_conll.txt"
    sentence_count = 0

    def process_and_yield_labeled_sentences():
        nonlocal sentence_count
        for sent in unlabeled_sentences_generator:
            sentence_count += 1
            if sentence_count % 1000 == 0:
                print(f"Processing sentence {sentence_count}...")
            try:
                labeled = ner_label_sentence(sent, ner_pipeline)
                yield labeled
            except Exception as e:
                print(f"Error processing sentence {sentence_count} (first 5 tokens): {sent[:5]}...")
                print(f"Error details: {e}")
                yield [(token, "O") for token in sent]

    print(f"Starting auto-labeling process for large file...")
    write_labeled_conll_chunked(process_and_yield_labeled_sentences(), output_filepath, chunk_size=5000)

    print(f"Auto-labeling completed for {sentence_count} sentences. Output saved to {output_filepath}")

if __name__ == "__main__":
    main()