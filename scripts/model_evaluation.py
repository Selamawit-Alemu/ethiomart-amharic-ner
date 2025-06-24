from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from datasets import Dataset, load_from_disk
import evaluate
import numpy as np
import json
import pickle

# Label mappings
label2id = {
    "O": 0,
    "B-Product": 1,
    "I-Product": 2,
    "B-PRICE": 3,
    "I-PRICE": 4,
    "B-LOC": 5,
    "I-LOC": 6
}
id2label = {v: k for k, v in label2id.items()}

# --- IMPORTANT: You need to have 'tokenized_dataset' defined here ---
# This part is missing in your provided code snippet.
# 'tokenized_dataset' should be a Hugging Face Dataset object,
# typically obtained after tokenizing your raw data.
# For demonstration purposes, let's assume you have a dummy one or load it:

# Example of how you might get tokenized_dataset (replace with your actual data loading/tokenization)
# If you have a pickled tokenized_dataset from a previous step, load it:
# with open("path/to/your/tokenized_dataset.pkl", "rb") as f:
#     tokenized_dataset = pickle.load(f)

# Or create a dummy one for the sake of making the script runnable for demonstration
# In a real scenario, this would come from your data processing pipeline.
raw_data = [
    {"tokens": ["This", "is", "a", "test", "product", "for", "10", "at", "Addis"],
     "ner_tags": [0, 0, 0, 0, 1, 0, 3, 0, 5]},
    {"tokens": ["Another", "item", "price", "5", "in", "Gonder"],
     "ner_tags": [0, 1, 0, 3, 0, 5]}
]

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True
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
                label_ids.append(label[word_idx]) # or -100 for subsequent tokens of a word
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Load tokenizer early for dummy dataset creation
model_path = "models/xlmr-ner-amharic"
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Convert dummy raw data to Dataset and then tokenize
dummy_dataset = Dataset.from_list(raw_data)
tokenized_dataset = dummy_dataset.map(tokenize_and_align_labels, batched=True)

# Split and save datasets to disk *before* attempting to load them
train_test = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset = train_test['train']
eval_dataset = train_test['test']

# Save datasets to disk
train_dataset.save_to_disk("train_dataset")
eval_dataset.save_to_disk("eval_dataset")

# Now, load the eval_dataset from disk
# eval_dataset = load_from_disk("eval_dataset") # This line is now redundant if you directly use the 'eval_dataset' created above

# --- End of 'tokenized_dataset' handling ---


# Load tokenizer and model
# model_path = "models/xlmr-ner-amharic" # Already defined above
# tokenizer = AutoTokenizer.from_pretrained(model_path) # Already loaded above
model = AutoModelForTokenClassification.from_pretrained(model_path, num_labels=len(label2id), id2label=id2label, label2id=label2id)


# Load metric
metric = evaluate.load("seqeval")

# Compute metrics function to use with Trainer
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = [
        [id2label[label] if label != -100 else "O" for label in label_seq]
        for label_seq in labels
    ]
    true_preds = [
        [id2label[pred] for pred in pred_seq]
        for pred_seq in predictions
    ]

    results = metric.compute(predictions=true_preds, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# Create Trainer
training_args = TrainingArguments(
    output_dir="./outputs/ner-xlmr",
    per_device_eval_batch_size=8,
    do_train=False,
    do_eval=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=eval_dataset, # Use the 'eval_dataset' created directly
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Run evaluation
metrics = trainer.evaluate()
print(json.dumps(metrics, indent=4))



from transformers import Trainer
import numpy as np
import evaluate

# Assuming you have these already loaded
# model: your fine-tuned model
# eval_dataset: your evaluation dataset
# id2label: mapping from label ids to label names

trainer = Trainer(model=model)

# Get predictions and label ids from the model on eval dataset
outputs = trainer.predict(eval_dataset)

logits = outputs.predictions  # shape: (num_samples, max_seq_length, num_labels)
labels = outputs.label_ids    # shape: (num_samples, max_seq_length)

# Convert logits to predicted class indices
predictions = np.argmax(logits, axis=2)

# Load metric
metric = evaluate.load("seqeval")

# Convert label ids and predictions to label names, ignoring special tokens (-100)
true_labels = [
    [id2label[label] for label in label_seq if label != -100]
    for label_seq in labels
]

predicted_labels = [
    [id2label[pred] for pred, label in zip(pred_seq, label_seq) if label != -100]
    for pred_seq, label_seq in zip(predictions, labels)
]

# Compute metrics
results = metric.compute(predictions=predicted_labels, references=true_labels)

print(results)
