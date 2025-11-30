from __future__ import annotations

import random
import shutil
from pathlib import Path

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


def list_images(folder: Path) -> list[Path]:
    return [
        file_path
        for file_path in folder.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def resolve_destination_collision(destination: Path) -> Path:
    stem = destination.stem
    suffix = destination.suffix
    counter = 1

    while True:
        candidate = destination.with_name(f"{stem}_{counter}{suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def main() -> None:
    folder_raw = input("Informe o caminho da pasta com imagens: ").strip().strip('"')
    if not folder_raw:
        print("Nenhum caminho informado. Encerrando.")
        return

    folder = Path(folder_raw).expanduser().resolve()
    if not folder.exists() or not folder.is_dir():
        print(f"Caminho invalido: {folder}")
        return

    images = list_images(folder)
    total_images = len(images)
    print(f"Total de imagens encontradas: {total_images}")

    if total_images == 0:
        print("Nenhuma imagem encontrada para processamento.")
        return

    while True:
        desired_raw = input(
            "Quantas imagens a pasta deve possuir apos a execucao? "
        ).strip()
        try:
            desired_total = int(desired_raw)
        except ValueError:
            print("Valor invalido. Informe um numero inteiro.")
            continue

        if desired_total < 0:
            print("O valor nao pode ser negativo.")
            continue
        if desired_total > total_images:
            print(
                f"Valor informado ({desired_total}) maior que a quantidade atual ({total_images})."
            )
            continue
        break

    to_move = total_images - desired_total
    if to_move == 0:
        print("Nenhuma imagem sera movida para o trash.")
        return

    trash_dir = folder / "trash"
    trash_dir.mkdir(exist_ok=True)

    for image_path in random.sample(images, to_move):
        destination = trash_dir / image_path.name
        if destination.exists():
            destination = resolve_destination_collision(destination)
        shutil.move(str(image_path), str(destination))

    print(f"{to_move} imagem(ns) movidas para {trash_dir}")


if __name__ == "__main__":
    main()
