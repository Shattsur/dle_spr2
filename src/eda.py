# eda.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Загрузка очищенного датасета
df_clean = pd.read_csv("data/cleaned.csv")

# --- Мини-EDA ---
df_clean["len_chars"] = df_clean["text"].apply(len)
df_clean["len_tokens"] = df_clean["text"].apply(lambda x: len(x.split()))

print("\n=== Мини-EDA ===")
print("Всего текстов:", len(df_clean))
print("Средняя длина (символы):", df_clean["len_chars"].mean())
print("Медиана длины (символы):", df_clean["len_chars"].median())
print("Средняя длина (токены):", df_clean["len_tokens"].mean())
print("Медиана длины (токены):", df_clean["len_tokens"].median())
print("95-й перцентиль (токены):", np.percentile(df_clean["len_tokens"], 95))

# --- Гистограмма ---
plt.figure(figsize=(8,4))
plt.hist(df_clean["len_tokens"], bins=50, edgecolor="black")
plt.title("Распределение длины текстов (в токенах)")
plt.xlabel("Количество токенов")
plt.ylabel("Частота")
plt.show()