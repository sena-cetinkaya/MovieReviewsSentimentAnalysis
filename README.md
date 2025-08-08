# 🎬 Movie Reviews Sentiment Analysis (FastAPI + LSTM)

Bu proje, **Türkçe film yorumlarını** analiz ederek, yorumun **olumlu**, **olumsuz** veya **nötr** olduğunu tahmin eden bir **duygu analizi (Sentiment Analysis)** API’sidir. Backend, **FastAPI** ile geliştirilmiş olup, derin öğrenme tabanlı **LSTM modeli** ile metin sınıflandırması yapılmaktadır.

---

## 🚀 Özellikler

- LSTM tabanlı derin öğrenme modeli
- Türkçe metin ön işleme (temizleme, tokenizasyon, etiketleme)
- Eğitilmiş model, tokenizer ve label encoder ile anında tahmin
- PostgreSQL veri tabanı entegrasyonu (SQLModel)
- Docker ve çoklu ortam desteği (dev/test)
- API testleri (`pytest` + `FastAPI TestClient`)

---

## 🗂️ Proje Yapısı

movie_reviews_sentiment_analysis/

│

├── app/

│ ├── db.py # DB bağlantısı ve başlatma

│ ├── main.py # FastAPI giriş noktası

│ ├── model_loader.py # Eğitilmiş model yükleme ve tahmin fonksiyonu

│ ├── routes.py # API endpoint'leri

│ ├── schemas.py # SQLModel tabloları ve şema tanımları

│

├── config/

│ ├── dev/.env.dev

│ └── test/.env.test

│

├── data/

│ ├── turkish_movie_sentiment_dataset.csv

│ ├── cleaned_turkish_movie_sentiment_dataset.csv

│ ├── lstm_sentiment_model.h5

│ ├── tokenizer.pkl

│ └── label_encoder.pkl

│

├── scripts/ # Sadece bir kez çalıştırılan yardımcı betikler

│ ├── preprocess.py

│ ├── create_lstm_model.py

│ └── load_data.py

│

├── test/

│ └── test_predict.py

│

├── docker-compose.dev.yml

├── docker-compose.test.yml

├── Dockerfile

└── requirements.txt

---

## ⚙️ Kurulum

1️⃣ Ortam değişkenlerini ayarla 

`config/dev/.env.dev` dosyasını düzenleyin:

```
DATABASE_URL=postgresql://kul_adi:sifre@db:5432/movies_dev
```

2️⃣ Docker ile çalıştırma (Geliştirme ortamı)

```
docker-compose -f docker-compose.dev.yml up --build
API, http://localhost:8000 adresinde çalışacaktır.
```

📡 API Kullanımı

Tahmin Endpoint’i

```
POST /predict
Content-Type: application/json
```
```
{
  "text": "Film harikaydı, bayıldım!"
}
```
✅ Örnek cevap:

```
{
  "prediction": "positive"
}
```

🧪 Test Çalıştırma

```
docker-compose -f docker-compose.test.yml up --build --exit-code-from test
```
veya
```
pytest
```

📦 Gereksinimler

requirements.txt dosyasında listelenmiştir:

```
fastapi
uvicorn
sqlmodel
tensorflow
scikit-learn
pandas
python-dotenv
pytest
```

📃 Lisans: MIT Lisansı

👩‍💻 Geliştirici: Sena Çetinkaya

🌐 GitHub: [https://github.com/sena-cetinkaya](https://github.com/sena-cetinkaya)
