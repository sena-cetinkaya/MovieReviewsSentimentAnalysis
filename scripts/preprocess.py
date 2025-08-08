import pandas as pd
import re

df = pd.read_csv("data/turkish_movie_sentiment_dataset.csv")  # Dosya adını senin dosyana göre değiştir

def clean_text(text):
    text = str(text)
    text = re.sub(r'\n|\r', ' ', text)  # Satır sonlarını sil
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Linkleri kaldır
    text = re.sub(r'[^a-zA-ZğüşöçıİĞÜŞÖÇ0-9\s]', '', text)  # Noktalama ve özel karakterleri temizle
    text = re.sub(r'\s+', ' ', text)  # Çoklu boşlukları teke indir
    return text.strip().lower()  # Küçük harfe çevir ve baş/son boşlukları sil

df_clean = df.copy()

# Yorumları temizle
df_clean['comment'] = df_clean['comment'].apply(clean_text)

# Puanları ondalık hale getir (5,0 → 5.0)
df_clean['point'] = df_clean['point'].str.replace(',', '.').astype(float)

# NaN olan satırları sil
df_clean.dropna(subset=['comment', 'point'], inplace=True)

def label_sentiment(score):
    if score >= 4.0:
        return "positive"
    elif score <= 2.5:
        return "negative"
    else:
        return "neutral"

df_clean['sentiment'] = df_clean['point'].apply(label_sentiment)

df_clean.to_csv("cleaned_turkish_movie_sentiment_dataset.csv", index=False)
