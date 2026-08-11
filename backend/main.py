from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
import sys
import os
import uvicorn

# Docker veya lokal ortam için path ayarla
sys.path.insert(0, str(Path(__file__).parent))
from model import load_model, predict, MODEL_PATH

app = FastAPI(
    title="Askeri Uçak Sınıflandırıcı API",
    description="EfficientNet-B0 tabanlı askeri uçak sınıflandırma servisi",
    version="1.0.0"
)

# CORS ayarları - frontend'den istek kabul et
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model ve class_names
model = None
class_names = None

@app.on_event("startup")
async def startup_event():
    """Uygulama başlarken modeli yükle."""
    global model, class_names
    if not MODEL_PATH.exists():
        print(f"⚠️  UYARI: Model dosyası bulunamadı: {MODEL_PATH}")
        print("   Lütfen 'efficientnet_b0_ucak.pth' dosyasını backend/ klasörüne kopyalayın.")
    else:
        print("✅ Model yükleniyor...")
        model, class_names = load_model()
        print(f"✅ Model başarıyla yüklendi! Sınıflar: {class_names}")


@app.get("/")
async def root():
    """Ana sayfa - frontend index.html'i serve eder."""
    # Docker içi path dene, yoksa lokal path dene
    candidates = [
        Path("/app/frontend/index.html"),
        Path(__file__).parent.parent / "frontend" / "index.html",
    ]
    for path in candidates:
        if path.exists():
            return FileResponse(str(path))
    return {"message": "Askeri Uçak Sınıflandırıcı API çalışıyor!", "docs": "/docs"}


@app.get("/health")
async def health_check():
    """Sağlık kontrolü endpoint'i."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
        "model_exists": MODEL_PATH.exists()
    }


@app.post("/predict")
async def predict_aircraft(file: UploadFile = File(...)):
    """
    Yüklenen görseldeki uçağı sınıflandırır.
    
    - **file**: PNG, JPG veya JPEG formatında görsel dosyası
    
    Yanıt olarak tahmin edilen sınıf, güven skoru ve tüm sınıfların olasılıkları döner.
    """
    # Model yüklü mü kontrol et
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model henüz yüklenmedi. 'efficientnet_b0_ucak.pth' dosyasının backend/ klasöründe olduğundan emin olun."
        )

    # Dosya tipi kontrolü
    allowed_types = ["image/jpeg", "image/jpg", "image/png", "image/webp"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Desteklenmeyen dosya formatı: {file.content_type}. Lütfen JPG, PNG veya WebP yükleyin."
        )

    # Dosyayı oku
    image_bytes = await file.read()

    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Boş dosya yüklendi.")

    # Tahmin yap
    try:
        result = predict(model, image_bytes, class_names)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Tahmin sırasında hata oluştu: {str(e)}"
        )


# Frontend statik dosyaları serve et (hem lokal hem Docker için)
frontend_candidates = [
    Path("/app/frontend"),
    Path(__file__).parent.parent / "frontend",
]
for frontend_dir in frontend_candidates:
    if frontend_dir.exists():
        app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")
        break


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
