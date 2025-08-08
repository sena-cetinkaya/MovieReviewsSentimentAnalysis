import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import pickle

# 1. Temizlenmiş CSV dosyasını oku
df = pd.read_csv("C:/Users/LENOVO/PycharmProjects/movie_reviews_sentiment_analysis/data/cleaned_turkish_movie_sentiment_dataset.csv")

# 2. Yorum ve etiket sütunlarını ayır
texts = df['comment'].astype(str).tolist()
labels = df['sentiment'].tolist()

# 3. Etiketleri sayısallaştır (pozitif, negatif, nötr)
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_categorical = to_categorical(labels_encoded)

# 4. Tokenizer oluştur ve eğit
vocab_size = 10000  # Maksimum kelime sayısı
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

# 5. Metinleri dizilere çevir
sequences = tokenizer.texts_to_sequences(texts)

# 6. Dizi uzunluklarını eşitle (padding)
max_length = 100
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

# 7. Eğitim ve test setine ayır
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels_categorical, test_size=0.2, random_state=42)

# 8. LSTM modeli oluştur
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=64, input_length=max_length))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))  # 3 sınıf: pozitif, negatif, nötr

# 9. Modeli derle
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 10. Modeli eğit
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# 11. Modeli kaydet
model.save("lstm_sentiment_model.h5")

# 12. Tokenizer'ı kaydet
with open("../data/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# 13. Label encoder'ı kaydet
with open("../data/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("Model, tokenizer ve label encoder başarıyla kaydedildi.")
