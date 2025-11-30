"""Inference utilities for InceptionV4 classifier (Benigno/Maligno/Pré-Maligno)."""

from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image
import timm


CLASS_NAMES_PT = ["Benigno", "Maligno", "Pré-Maligno"]
# Ordem original de treino era ['Benignos','Malignos','Pre-Malignos'] -> convertemos para labels amigáveis

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_TRANSFORM = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def _weights_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    candidates = [
        root / "Treinamento Modelos" / "inceptionV4" / "best_inceptionv4.pth",
        root / "Treinamento Modelos" / "inceptionV4" / "outputs" / "best_inceptionv4.pth",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Arquivo de pesos 'best_inceptionv4.pth' não encontrado.")


@lru_cache(maxsize=1)
def _load_model():
    weights_path = _weights_path()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = timm.create_model('inception_v4', pretrained=False, num_classes=len(CLASS_NAMES_PT))
    state = torch.load(weights_path, map_location=device)
    # early_stopper salvou somente state_dict
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model, device


def classify_image_bytes(image_bytes: bytes) -> dict:
    """Run inference on the provided image bytes and returns prediction + probabilities."""
    model, device = _load_model()

    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    tensor = _TRANSFORM(img).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    top_idx = int(probs.argmax())
    result = {
        "prediction": CLASS_NAMES_PT[top_idx],
        "probabilities": {
            CLASS_NAMES_PT[i]: float(round(prob, 4)) for i, prob in enumerate(probs)
        }
    }
    return result
