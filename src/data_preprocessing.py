# data_preprocessing.py

import pandas as pd
import re
import html
from tqdm import tqdm

# Чтение исходного CSV
with open("data/raw_data.csv", "r", encoding="latin-1") as file:
    lines = file.readlines()

# Создание DataFrame
df = pd.DataFrame({'text': lines})
df['text'] = df['text'].str.strip()

# --- Очистка текста ---
def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = html.unescape(text).lower()
    text = re.sub(r'http\S+|www\.\S+', ' <URL> ', text)  # ссылки
    text = re.sub(r'@\w+', ' <USER> ', text)             # упоминания
    re.sub(r'\brt\b', ' ', text)                          # retweet
    text = re.sub(r"[^a-z0-9а-яё\s\.\,\!\?\:\;\-\']+", ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

texts = df['text'].astype(str).tolist()
cleaned = [clean_text(t) for t in tqdm(texts, desc="Cleaning texts")]

df_clean = pd.DataFrame({"text": cleaned})
df_clean.to_csv("data/cleaned.csv", index=False, encoding="utf-8")

print(f"Сохранено: data/cleaned.csv (строк: {len(df_clean)})")