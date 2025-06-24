from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from datasets import Dataset
import torch

# Label map (same as in training)
label2id = {
    "O": 0,
    "B-Product": 1,
    "I-Product": 2,
    "B-PRICE": 3,
    "I-PRICE": 4,
    "B-LOC": 5,
    "I-LOC": 6
}

# Parse your CoNLL again
def parse_conll(file_path):
    sentences = []
    tokens, tags = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    sentences.append({"tokens": tokens, "ner_tags": tags})
                    tokens, tags = [], []
            else:
                if len(line.split()) == 2:
                    token, tag = line.split()
                    tokens.append(token)
                    tags.append(label2id.get(tag, 0))
    return sentences

# Load and tokenize
dataset = parse_conll("data/clean/labeled_conll.txt")
hf_dataset = Dataset.from_list(dataset)

tokenizer = AutoTokenizer.from_pretrained("Davlan/bert-tiny-amharic-ner")

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding="max_length",
        max_length=128,
        return_offsets_mapping=True
    )
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx])
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_dataset = hf_dataset.map(tokenize_and_align_labels, batched=True)

# Split
eval_dataset = tokenized_dataset.train_test_split(test_size=0.2)["test"]

# Load pre-trained model
model = AutoModelForTokenClassification.from_pretrained("Davlan/bert-tiny-amharic-ner")

# Evaluation args
training_args = TrainingArguments(
    output_dir="./outputs/eval-tinybert",
    per_device_eval_batch_size=8,
    logging_dir="./logs",
    do_eval=True,
    report_to="none"
)

# Evaluate
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer
)

metrics = trainer.evaluate()
print("Evaluation Metrics:\n", metrics)
