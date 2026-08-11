import torch
import torchvision
from torchvision import transforms
from PIL import Image
import gradio as gr
from pathlib import Path

# Hugging Face ZeroGPU desteği
try:
    import spaces
    HAS_SPACES = True
except ImportError:
    HAS_SPACES = False
    # Fallback decorator if running locally
    class spaces:
        @staticmethod
        def GPU(func):
            return func

# Sınıf İsimleri
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
if not MODEL_PATH.exists():
    MODEL_PATH = Path(__file__).parent / "backend" / "efficientnet_b0_ucak.pth"

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_model():
    model = torchvision.models.efficientnet_b0(weights=None)
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2),
        torch.nn.Linear(in_features=1280, out_features=len(CLASS_NAMES), bias=True)
    )
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model

model = load_model()

@spaces.GPU
def predict(img):
    if img is None:
        return None
    
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    tensor = TRANSFORM(img.convert("RGB")).unsqueeze(0).to(device)

    with torch.inference_mode():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    results = {}
    for name, prob in zip(CLASS_NAMES, probs):
        display_name = CLASS_DISPLAY_NAMES.get(name, name)
        results[display_name] = float(prob)
        
    return results

title = "✈️ Askeri Uçak Sınıflandırıcı (ZeroGPU)"
description = "Görsel yükleyin. EfficientNet-B0 yapay zeka modeli ZeroGPU desteğiyle uçak türünü tahmin etsin."

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Uçak Görseli Yükleyin"),
    outputs=gr.Label(num_top_classes=3, label="Tahmin Sonuçları"),
    title=title,
    description=description
)

if __name__ == "__main__":
    demo.launch()
