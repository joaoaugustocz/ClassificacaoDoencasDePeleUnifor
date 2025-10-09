# Pré-processamento Geral

Scripts genéricos para qualquer dataset de imagens dermatológicas.

## Scripts

### `circle_to_square.py`
Detecta ROI circular e converte para quadrado.

**Quando usar**: Imagens com fundo circular transparente ou vinheta escura

**Entrada**: Qualquer pasta com imagens
**Saída**: `<pasta>/roiSquare/`

```bash
python scripts/preprocessing/circle_to_square.py
```

---

### `color_constancy_clahe.py`
Normalização de cor e contraste.

**Técnicas**:
- Gray-World: Balanceia canais RGB
- CLAHE: Equalização de histograma adaptativo (Lab ou HSV)

**Entrada**: Qualquer pasta com imagens
**Saída**: `<pasta>/colorFixed_Lab/` ou `colorFixed_HSV/`

```bash
python scripts/preprocessing/color_constancy_clahe.py
```