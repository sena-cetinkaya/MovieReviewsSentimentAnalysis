import pandas as pd
import re

df = pd.read_csv("data/turkish_movie_sentiment_dataset.csv") 

def clean_text(text):
    text = str(text)
    text = re.sub(r'\n|\r', ' ', text) 
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  
    text = re.sub(r'[^a-zA-ZğüşöçıİĞÜŞÖÇ0-9\s]', '', text)  
    text = re.sub(r'\s+', ' ', text) 
    return text.strip().lower() 

df_clean = df.copy()

df_clean['comment'] = df_clean['comment'].apply(clean_text)

df_clean['point'] = df_clean['point'].str.replace(',', '.').astype(float)

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

