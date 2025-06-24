from transformers import AutoTokenizer

# Load tokenizer
model_name = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Label map for inverse lookups (if needed)
id2label = {0: "O", 1: "B-Product", 2: "I-Product", 3: "B-PRICE", 4: "I-PRICE", 5: "B-LOC", 6: "I-LOC"}

# Function to align labels with wordpieces
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding='max_length',
        max_length=128,
        return_offsets_mapping=True  # required for alignment
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        aligned_labels = []
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(-100)
            elif word_idx != previous_word_idx:
                aligned_labels.append(label[word_idx])
            else:
                # For subword tokens: either repeat or mask (-100)
                aligned_labels.append(label[word_idx])  # or -100
            previous_word_idx = word_idx

        labels.append(aligned_labels)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs
