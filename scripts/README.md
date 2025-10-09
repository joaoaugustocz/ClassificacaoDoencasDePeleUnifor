# Scripts de Processamento de Imagens

Organização dos scripts Python do projeto.

## 📁 Estrutura

### `preprocessing/` - Pré-processamento Geral
Scripts genéricos para qualquer dataset de imagens dermatológicas:

- **`circle_to_square.py`** - Extrai ROI circular e converte para quadrado
- **`color_constancy_clahe.py`** - Normalização de cor (Gray-World + CLAHE)

**Uso típico**: Datasets com vinheta circular ou iluminação irregular

---

### `augmentation/` - Data Augmentation
Scripts para expansão de datasets:

- **`augment_skin_images.py`** - Gera augmentations (rotação, flip, zoom, ruído, etc)

**Uso típico**: Aumentar número de imagens de treinamento

---

### `ham10000_pipeline/` - Pipeline HAM10000
Scripts específicos para o dataset HAM10000:

- **`apply_masks.py`** - Aplica máscaras de segmentação + correção de vinheta
- **`resize_for_model.py`** - Redimensiona para tamanhos padrão de modelos (224×224 ou 299×299)

**Uso típico**: Pipeline completo do HAM10000 para classificação

---

## 🔄 Fluxo de Trabalho Recomendado (HAM10000)

```
1. apply_masks.py          → Aplica máscaras + corrige vinheta
   ├─ Entrada: HAM10000/images + HAM10000/masks
   └─ Saída: HAM10000/masked_images_cropped_vignette_corrected/

2. resize_for_model.py     → Redimensiona para modelo
   ├─ Entrada: HAM10000/masked_images_cropped_vignette_corrected/
   └─ Saída: HAM10000/model_ready_224x224_padded/

3. [Treinar modelo]        → Usar imagens prontas
```

---

## 📝 Como Executar

Todos os scripts são interativos. Execute a partir da **raiz do projeto**:

```bash
# Pré-processamento geral
python scripts/preprocessing/circle_to_square.py
python scripts/preprocessing/color_constancy_clahe.py

# Augmentation
python scripts/augmentation/augment_skin_images.py

# Pipeline HAM10000
python scripts/ham10000_pipeline/apply_masks.py
python scripts/ham10000_pipeline/resize_for_model.py
```