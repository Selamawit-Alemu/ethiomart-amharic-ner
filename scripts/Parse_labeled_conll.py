from pathlib import Path

label2id = {
    "O": 0,
    "B-Product": 1,
    "I-Product": 2,
    "B-PRICE": 3,
    "I-PRICE": 4,
    "B-LOC": 5,
    "I-LOC": 6
}

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
                splits = line.split()
                if len(splits) == 2:
                    token, tag = splits
                    tokens.append(token)
                    tags.append(label2id.get(tag, 0))  # default to 'O'
    
    return sentences

# Example usage
data_path = "data/clean/labeled_conll.txt"
dataset = parse_conll(data_path)
print(f"Loaded {len(dataset)} labeled messages")
print(dataset[0])
