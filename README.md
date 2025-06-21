
# EthioMart Amharic NER System

This project builds a Named Entity Recognition (NER) pipeline for extracting product, price, and location information from Amharic e-commerce messages posted on Telegram channels.

---

## 📌 Project Goals

- Scrape real-time messages from Ethiopian e-commerce Telegram channels
- Preprocess and normalize Amharic text data
- Manually label a subset of data in CoNLL format for NER training
- Fine-tune transformer models for Amharic entity extraction

---

## 📁 Folder Structure

    ├── data/
    │ ├── raw/ # Raw input files (e.g., Excel with channels)
    │ ├── clean/ # Preprocessed & labeled data
    ├── scripts/ # Data ingestion and preprocessing scripts
    ├── Notebooks/ # EDA and preprocessing notebooks
    ├── outputs/ # Model outputs, logs
    ├── photos/ # Downloaded media (images from Telegram)
    ├── README.md
    ├── requirements.txt


---

## 🛠️ Setup Instructions

1. **Clone the repo & install dependencies**
   ```bash
   git clone https://github.com/<your-username>/ethiomart-amharic-ner.git
   cd ethiomart-amharic-ner
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt


Configure your Telegram API credentials

Create a .env file in the root directory:

TG_API_ID=your_api_id
TG_API_HASH=your_api_hash
phone=+2519xxxxxxx

🚀 How to Run the System

🧲 Step 1: Scrape Telegram Channels

Edit data/raw/5_channels_to_crawl.xlsx to include 5+ target channels (one per line). Then run:

python scripts/telegram_scrapper.py

This will:

Log in to your Telegram account

Download up to 10,000 messages per channel

Save messages and metadata to telegram_data.csv

Download media to /photos

🧹 Step 2: Preprocess the Data

python scripts/preprocess_telegram_data.py

This will:

Clean Amharic/English text (punctuation, emojis, formatting)

Tokenize each message

Export structured data to processed_telegram_data.csv

Prepare unlabeled_conll.txt for manual labeling

🏷️ Step 3: Label in CoNLL Format

Manually label a subset (30–50 messages) using entity tags:

B-Product, I-Product

B-PRICE, I-PRICE

B-LOC, I-LOC

O for non-entities

Save your labels to data/clean/labeled_conll.txt.

✅ Completed Milestones

✅ Connected to 5 Telegram channels: @ZemenExpress, @sinayelj, @Leyueqa, @ethio_brand_collection, @nevacomputer

✅ Implemented async message ingestion with media download

✅ Tokenized Amharic-English text and handled Unicode

✅ Prepared unlabeled data in CoNLL format

✅ Labeled at least 30 messages manually

🧠 Next Steps (Task 3)

Fine-tune a transformer model (AfroXLMR or XLM-R) on labeled data

Use sklearn, transformers, and seqeval for training & evaluation

Apply SHAP/LIME for interpretability