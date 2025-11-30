from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def segment_lesion(image_path: Path, output_dir: Path) -> None:
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Falha ao carregar a imagem: {image_path}")
        return

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    a_channel = lab[:, :, 1]
    blurred = cv2.GaussianBlur(a_channel, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    isolated_lesion = cv2.bitwise_and(img, img, mask=mask)

    output_dir.mkdir(parents=True, exist_ok=True)
    mask_path = output_dir / f"{image_path.stem}_mask.jpg"
    lesion_path = output_dir / f"{image_path.stem}_isolated.jpg"

    cv2.imwrite(str(mask_path), mask)
    cv2.imwrite(str(lesion_path), isolated_lesion)

    print(f"Segmentacao concluida para {image_path.name}.")


def main() -> None:
    folder_input = input("Informe o caminho da pasta com imagens: ").strip().strip('"')
    if not folder_input:
        print("Nenhum caminho informado. Encerrando.")
        return

    folder = Path(folder_input).expanduser().resolve()
    if not folder.exists() or not folder.is_dir():
        print(f"Caminho invalido: {folder}")
        return

    image_files = [
        file_path
        for file_path in folder.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not image_files:
        print("Nenhuma imagem encontrada na pasta informada.")
        return

    output_dir = folder / "output_segmentation"
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in image_files:
        segment_lesion(image_path, output_dir)

    print(f"Processamento finalizado. Resultados em {output_dir}")


if __name__ == "__main__":
    main()
