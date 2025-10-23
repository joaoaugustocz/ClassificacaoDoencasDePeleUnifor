import cv2
import numpy as np


def _segment_mask_from_bgr(img_bgr: np.ndarray, roi_mask: np.ndarray | None = None) -> np.ndarray:
    """Gera uma máscara binária (uint8 0/255) da lesão a partir de uma imagem BGR.

    Estratégia simples inspirada no segment_lesion.py existente:
    - Converte para LAB e usa canal 'a'.
    - Desfoque gaussiano + Otsu para binarizar.
    - Abertura e fechamento morfológicos para limpar ruído.
    - Seleciona o maior contorno como região da lesão.
    Quando `roi_mask` é fornecida (alpha circular do front), neutralizamos o exterior
    para o Otsu não escolher o disco inteiro.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    a_channel = lab[:, :, 1]

    if roi_mask is not None:
        roi = roi_mask.astype(bool)
        if np.any(roi):
            median_val = int(np.median(a_channel[roi]))
            a_channel = a_channel.copy()
            a_channel[~roi] = median_val

    blurred = cv2.GaussianBlur(a_channel, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(img_bgr.shape[:2], dtype="uint8")
    if contours:
        c = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [c], -1, (255), thickness=cv2.FILLED)

    if roi_mask is not None:
        mask = cv2.bitwise_and(mask, roi_mask)
    return mask


def segment_lesion_bytes(image_bytes: bytes) -> bytes:
    """Recebe bytes de imagem e retorna PNG RGBA com a lesão isolada.

    - Decodifica a imagem (preservando alpha se houver)
    - Calcula máscara restringindo ao ROI (alpha) quando presente
    - Retorna PNG com fundo transparente (alpha = máscara)
    """
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Falha ao decodificar imagem recebida")

    roi_mask = None
    if img.ndim == 3 and img.shape[2] == 4:
        # PNG com alpha (provável recorte circular do front)
        b, g, r, a = cv2.split(img)
        img_bgr = cv2.merge((b, g, r))
        roi_mask = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY)[1]
    elif img.ndim == 3 and img.shape[2] == 3:
        img_bgr = img
    else:  # grayscale
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    mask = _segment_mask_from_bgr(img_bgr, roi_mask)

    # Constrói BGRA usando a máscara como alpha
    b, g, r = cv2.split(img_bgr)
    rgba = cv2.merge((b, g, r, mask))

    ok, buf = cv2.imencode('.png', rgba)
    if not ok:
        raise RuntimeError("Falha ao codificar PNG")
    return buf.tobytes()

