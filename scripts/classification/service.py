from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path

import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import timm
import numpy as np


CLASS_NAMES_PT = ["Benigno", "Maligno", "Pré-Maligno"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

SPECIALIST_PAIRS = {
    ("Benigno", "Maligno"): "benignos_vs_malignos",
    ("Maligno", "Benigno"): "benignos_vs_malignos",
    ("Maligno", "Pré-Maligno"): "malignos_vs_premalignos",
    ("Pré-Maligno", "Maligno"): "malignos_vs_premalignos",
    ("Benigno", "Pré-Maligno"): "premalignos_vs_benignos",
    ("Pré-Maligno", "Benigno"): "premalignos_vs_benignos",
}


def _get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _weights_path_general() -> Path:
    root = Path(__file__).resolve().parents[2]
    candidates = [
        root / "Treinamento Modelos" / "resnetrs50" / "best_model.pth",
        root / "Treinamento Modelos" / "resnetrs50" / "outputs" / "best_model.pth",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Arquivo de pesos 'best_model.pth' não encontrado em Treinamento Modelos/resnetrs50.")


def _weights_path_specialist(pair_name: str) -> Path | None:
    root = Path(__file__).resolve().parents[2]
    path = root / "Treinamento Modelos" / "especialistas" / "resnetrs50" / pair_name / "best_model.pth"
    if path.exists():
        return path
    return None


@lru_cache(maxsize=1)
def _load_general_model():
    weights_path = _weights_path_general()
    device = _get_device()
    model = timm.create_model('resnetrs50', pretrained=False, num_classes=len(CLASS_NAMES_PT))
    state = torch.load(weights_path, map_location=device)

    new_state = {}
    for k, v in state.items():
        if k == 'fc.1.weight':
            new_state['fc.weight'] = v
        elif k == 'fc.1.bias':
            new_state['fc.bias'] = v
        else:
            new_state[k] = v

    model.load_state_dict(new_state)
    model.eval()
    model.to(device)
    return model, device


@lru_cache(maxsize=3)
def _load_specialist_model(pair_name: str):
    weights_path = _weights_path_specialist(pair_name)
    if weights_path is None:
        return None, None

    device = _get_device()
    model = timm.create_model('resnetrs50', pretrained=False, num_classes=2)
    state = torch.load(weights_path, map_location=device)

    new_state = {}
    for k, v in state.items():
        if k == 'fc.1.weight':
            new_state['fc.weight'] = v
        elif k == 'fc.1.bias':
            new_state['fc.bias'] = v
        else:
            new_state[k] = v

    model.load_state_dict(new_state)
    model.eval()
    model.to(device)
    return model, device


def _get_top_2_classes(probs: np.ndarray) -> tuple[str, str]:
    top_2_indices = np.argsort(probs)[-2:][::-1]
    return CLASS_NAMES_PT[top_2_indices[0]], CLASS_NAMES_PT[top_2_indices[1]]


def classify_image_bytes(image_bytes: bytes) -> dict:
    device = _get_device()

    general_model, _ = _load_general_model()
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    tensor = _TRANSFORM(img).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = general_model(tensor)
        general_probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    top_class_1, top_class_2 = _get_top_2_classes(general_probs)
    specialist_key = SPECIALIST_PAIRS.get((top_class_1, top_class_2))
    # specialist_key = None  # Desabilitar especialistas para teste com modelo geral apenas
    specialist_model, _ = _load_specialist_model(specialist_key) if specialist_key else (None, None)

    if specialist_model is None:
        top_idx = int(general_probs.argmax())
        return {
            "prediction": CLASS_NAMES_PT[top_idx],
            "probabilities": {
                CLASS_NAMES_PT[i]: float(round(prob, 4)) for i, prob in enumerate(general_probs)
            },
            "specialist_used": False
        }

    with torch.inference_mode():
        specialist_logits = specialist_model(tensor)
        specialist_probs = torch.softmax(specialist_logits, dim=1)[0].cpu().numpy()

    specialist_classes = sorted([top_class_1, top_class_2])
    final_prediction = specialist_classes[int(specialist_probs.argmax())]

    all_probs = {}
    specialist_idx = 0
    for class_name in CLASS_NAMES_PT:
        if class_name in specialist_classes:
            all_probs[class_name] = float(round(specialist_probs[specialist_idx], 4))
            specialist_idx += 1
        else:
            all_probs[class_name] = 0.0

    return {
        "prediction": final_prediction,
        "probabilities": all_probs,
        "specialist_used": True
    }
