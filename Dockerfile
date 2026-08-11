FROM python:3.10-slim

WORKDIR /app

# Sistem bağımlılıkları (Pillow için)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Önce requirements.txt kopyala (cache için)
COPY backend/requirements.txt ./requirements.txt

# PyTorch CPU-only sürümü yükle (GPU yok, image boyutunu küçültür ~800MB kazanç)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Geri kalan bağımlılıklar
RUN pip install --no-cache-dir fastapi uvicorn[standard] pillow python-multipart

# Uygulama dosyalarını kopyala
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Cloud Run PORT env variable'ını kullan (varsayılan 8080)
EXPOSE 8080

# Başlatma komutu — Cloud Run $PORT ortam değişkenini inject eder
CMD exec uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8080}