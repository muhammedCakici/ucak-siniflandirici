import torch
import torchvision
from torchvision import transforms
from PIL import Image
import io
from pathlib import Path

CLASS_NAMES = [
    'An-12',
    'C-130',
    'C-47',
    'Eurofighter_Typhoon',
    'F-16A-B',
    'Hawk_T1',
    'Spitfire',
    'Tornado'
]

# Sınıf görünen adları (UI için daha okunabilir)
CLASS_DISPLAY_NAMES = {
    'An-12': 'Antonov An-12',
    'C-130': 'Lockheed C-130 Hercules',
    'C-47': 'Douglas C-47 Skytrain',
    'Eurofighter_Typhoon': 'Eurofighter Typhoon',
    'F-16A-B': 'General Dynamics F-16 Fighting Falcon',
    'Hawk_T1': 'BAE Systems Hawk T1',
    'Spitfire': 'Supermarine Spitfire',
    'Tornado': 'Panavia Tornado'
}

MODEL_PATH = Path(__file__).parent / "efficientnet_b0_ucak.pth"

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model() -> torch.nn.Module:
    """EfficientNet-B0 modelini yükler ve sınıflandırıcıyı ayarlar."""
    model = torchvision.models.efficientnet_b0(weights=None)

    # Kaydedilen dosyadaki class_names listesine göre output boyutu belirlenir
    # Model kaydedilirken class_names de kaydedilmişti; fallback olarak CLASS_NAMES kullanıyoruz
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2),
        torch.nn.Linear(in_features=1280, out_features=len(CLASS_NAMES), bias=True)
    )

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        # Eğer model dosyasında class_names varsa onu kullan
        saved_class_names = checkpoint.get("class_names", CLASS_NAMES)
    else:
        # Sadece state_dict kaydedilmişse
        model.load_state_dict(checkpoint)
        saved_class_names = CLASS_NAMES

    model.to(device)
    model.eval()
    return model, saved_class_names


def predict(model: torch.nn.Module, image_bytes: bytes, class_names: list) -> dict:
    """Görsel baytlarını alır, model tahmini yapar ve sonuçları döndürür."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = TRANSFORM(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    # Tüm sınıf olasılıklarını hazırla
    all_probs = [
        {
            "class_name": name,
            "display_name": CLASS_DISPLAY_NAMES.get(name, name),
            "probability": round(prob.item() * 100, 2)
        }
        for name, prob in zip(class_names, probs)
    ]

    # Olasılığa göre büyükten küçüğe sırala
    all_probs.sort(key=lambda x: x["probability"], reverse=True)

    top = all_probs[0]

    return {
        "predicted_class": top["class_name"],
        "display_name": top["display_name"],
        "confidence": top["probability"],
        "all_predictions": all_probs
    }
