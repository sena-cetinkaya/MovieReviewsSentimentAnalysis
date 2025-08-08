import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

"""with open("data/lstm_sentiment_model.h5", "rb") as f:
    model = pickle.load(f)"""

model = load_model("data/lstm_sentiment_model.h5")

with open("data/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("data/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

def predict_sentiment(text: str) -> str:
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=100)
    prediction = model.predict(padded)
    predicted_index = prediction.argmax(axis=1)[0]
    return label_encoder.inverse_transform([predicted_index])[0]