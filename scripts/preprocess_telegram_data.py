import pandas as pd

# Load the raw scraped data
df = pd.read_csv("telegram_data.csv")

# Display basic info
print("Rows:", len(df))
print("Columns:", df.columns.tolist())
print(df.head(3))
