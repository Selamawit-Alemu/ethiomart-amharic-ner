from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from datasets import Dataset
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from transformers import XLMRobertaModel, XLMRobertaPreTrainedModel

# Label mapping
label2id = {
    "O": 0,
    "B-Product": 1,
    "I-Product": 2,
    "B-PRICE": 3,
    "I-PRICE": 4,
    "B-LOC": 5,
    "I-LOC": 6
}

# Parse CoNLL data
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

# Load dataset
dataset = parse_conll("data/clean/labeled_conll.txt")
hf_dataset = Dataset.from_list(dataset)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

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

# Split into train and eval
train_test = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset = train_test['train']
eval_dataset = train_test['test']

# Define custom model (optional if using class weights)
class WeightedXLMRForTokenClassification(XLMRobertaPreTrainedModel):
    def __init__(self, config, class_weights=None):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.class_weights = class_weights
        self.roberta = XLMRobertaModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs[0])
        logits = self.classifier(sequence_output)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights, ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}

# Standard model for now
model = AutoModelForTokenClassification.from_pretrained(
    "xlm-roberta-base",
    num_labels=len(label2id),
    id2label={v: k for k, v in label2id.items()},
    label2id=label2id
)

# TrainingArguments with safe defaults for older versions
training_args = TrainingArguments(
    output_dir="./outputs/ner-xlmr",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    save_steps=500,
    logging_steps=50,
    save_total_limit=1,
    logging_strategy="epoch",
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

trainer.train()


trainer.save_model("models/final-xlmr-ner-amharic", safe_serialization=False)
