# Data Augmentation

Scripts para expansão de datasets através de transformações.

## Scripts

### `augment_skin_images.py`
Gera novas imagens aplicando transformações aleatórias.

**Transformações aplicadas** (3-5 por imagem):
- Rotação (±30°)
- Flips (horizontal, vertical, ambos)
- Translação e zoom (±6%, escala 0.92-1.08x)
- Brilho e contraste
- Blur/Sharpen
- Ruído gaussiano
- Salt-and-pepper noise

**Uso**:
1. Informe a pasta com imagens originais
2. Informe quantas imagens deseja no total (originais + geradas)

**Saída**: `<pasta>/imgOutput/`

```bash
python scripts/augmentation/augment_skin_images.py
```

**Exemplo**:
- Originais: 100 imagens
- Desejado: 500 imagens
- Geradas: 400 augmentations