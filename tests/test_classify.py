import os
from scripts.classification.service import classify_image_bytes

image_path = os.path.join("tests", "pre_maligno.jpg")

if not image_path:
    print(f"Arquivo não encontrado: {image_path}")
    exit(1)

print(f"📷 Testando: {os.path.basename(image_path)}")

with open(image_path, "rb") as f:
    image_bytes = f.read()

try:
    result = classify_image_bytes(image_bytes)
    print("\nClassificação bem-sucedida!\n")
    print(f"Predição: {result['prediction']}")
    print(f"Probabilidades: {result['probabilities']}")
    print(f"Especialista usado: {result['specialist_used']}")
except Exception as e:
    print(f"\nErro: {e}")
    import traceback
    traceback.print_exc()