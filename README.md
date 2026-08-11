# ✈️ Askeri Uçak Sınıflandırıcı

EfficientNet-B0 tabanlı derin öğrenme modeliyle askeri uçakları sınıflandıran modern web uygulaması.

## 🗂 Proje Yapısı

```
Uçak Sınıflandırma/
├── backend/
│   ├── main.py                    # FastAPI uygulaması
│   ├── model.py                   # Model yükleme & tahmin
│   ├── requirements.txt           # Python bağımlılıkları
│   └── efficientnet_b0_ucak.pth  # ← Buraya koy!
└── frontend/
    ├── index.html
    ├── style.css
    └── app.js
```

---

## 🚀 Kurulum & Çalıştırma

### 1. Model dosyasını kopyala

Colab'dan indirdiğin `efficientnet_b0_ucak.pth` dosyasını `backend/` klasörüne koy:

```
backend/efficientnet_b0_ucak.pth
```

### 2. Python bağımlılıklarını yükle

```bash
cd backend
pip install -r requirements.txt
```

### 3. Backend'i başlat

```bash
cd backend
uvicorn main:app --reload
```

Sunucu `http://localhost:8000` adresinde çalışmaya başlar.

### 4. Frontend'i aç

`frontend/index.html` dosyasını tarayıcıda aç.

> **İpucu:** VS Code kullanıyorsan "Live Server" eklentisiyle kolayca açabilirsin.

---

## 🎯 Özellikler

- **Drag & Drop** görsel yükleme
- **8 askeri uçak sınıfı** tanıma
- Animasyonlu **güven skoru halkası**
- Tüm sınıflar için **olasılık çubukları**
- **Dark mode** + **Glassmorphism** tasarım
- FastAPI **Swagger UI** → `http://localhost:8000/docs`

## ✈️ Tanınan Sınıflar

| Sınıf | Tam Ad |
|-------|--------|
| An-12 | Antonov An-12 |
| C-130 | Lockheed C-130 Hercules |
| C-47 | Douglas C-47 Skytrain |
| Eurofighter_Typhoon | Eurofighter Typhoon |
| F-16A-B | General Dynamics F-16 Fighting Falcon |
| Hawk_T1 | BAE Systems Hawk T1 |
| Spitfire | Supermarine Spitfire |
| Tornado | Panavia Tornado |
