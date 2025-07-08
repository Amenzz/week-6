# src/clean_and_save.py
import pandas as pd
import re
import os

# Load the filtered complaint dataset
df = pd.read_csv(r"C:\Users\Amenzz\Desktop\week-6\data\filtered_complaints.csv")

# Define the text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\bi am writing to file a complaint\b', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Drop rows with empty narratives
df = df[df['Consumer complaint narrative'].notnull()].copy()

# Apply cleaning
df["cleaned_narrative"] = df["Consumer complaint narrative"].apply(clean_text)

# Save updated file
os.makedirs("data", exist_ok=True)
df.to_csv("data/filtered_complaints.csv", index=False)

print("✅ Updated filtered_complaints.csv with 'cleaned_narrative' column.")
