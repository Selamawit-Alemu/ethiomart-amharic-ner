import pandas as pd
import json

# Load your preprocessed data (adjust path if needed)
df = pd.read_csv('data/clean/processed_telegram_data.csv', encoding='utf-8')

# Use the cleaned message column (replace 'Cleaned_Message' if named differently)
messages = df['Cleaned_Message'].dropna()

# Export to Label Studio JSONL format
with open('label_studio_tasks.jsonl', 'w', encoding='utf-8') as f:
    for msg in messages:
        task = {"text": msg}
        f.write(json.dumps(task, ensure_ascii=False) + '\n')

