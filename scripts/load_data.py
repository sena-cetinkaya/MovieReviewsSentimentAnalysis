import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sqlmodel import Session
from app.schemas import Sentiment
from app.db import engine

df = pd.read_csv("../data/cleaned_turkish_movie_sentiment_dataset.csv")

with Session(engine) as session:
    for _, row in df.iterrows():
        sentiment = Sentiment(text=row["comment"], prediction=row["sentiment"])
        session.add(sentiment)
        session.commit()
print("The record was successfully uploaded to the database.")