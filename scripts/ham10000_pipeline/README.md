# Pipeline HAM10000

Scripts específicos para processar o dataset HAM10000 para classificação de lesões de pele.

## Pipeline Completo

```
HAM10000/images + HAM10000/masks
         ↓
   apply_masks.py (modo 4)
         ↓
HAM10000/masked_images_cropped_vignette_corrected/
         ↓
   resize_for_model.py
         ↓
HAM10000/model_ready_224x224_padded/
         ↓
   [Treinar modelo de classificação]
```

---

## Scripts

### 1. `apply_masks.py`
Aplica máscaras de segmentação com opções de processamento.

**Modos disponíveis**:
1. Isolado - Lesão sobre fundo neutro
2. Recortado - Crop ao bounding box
3. Isolado + Vinheta corrigida
4. Recortado + Vinheta corrigida ⭐ **RECOMENDADO**
5. Todos - Gera todas as versões

**Correção de vinheta**: Remove iluminação não-uniforme usando blur gaussiano adaptativo no canal L (Lab)

**Entrada**:
- `HAM10000/images/` - Imagens originais
- `HAM10000/masks/` - Máscaras de segmentação

**Saída** (depende do modo):
- `HAM10000/masked_images_isolated/`
- `HAM10000/masked_images_cropped/`
- `HAM10000/masked_images_isolated_vignette_corrected/`
- `HAM10000/masked_images_cropped_vignette_corrected/` ⭐

```bash
python scripts/ham10000_pipeline/apply_masks.py
```

---

### 2. `resize_for_model.py`
Redimensiona imagens para tamanhos padrão de modelos.

**Tamanhos disponíveis**:
- **224×224** - ResNet, EfficientNet, VGG, DenseNet
- **299×299** - Inception v3, Xception

**Métodos**:
1. **Aspect ratio + padding** ⭐ **RECOMENDADO**
   - Preserva proporção da lesão
   - Adiciona barras cinzas se necessário

2. **Stretch**
   - Estica para preencher
   - Pode distorcer levemente

**Entrada**: `HAM10000/masked_images_cropped_vignette_corrected/`

**Saída**: `HAM10000/model_ready_<SIZE>_<METHOD>/`
- Exemplo: `model_ready_224x224_padded/`

```bash
python scripts/ham10000_pipeline/resize_for_model.py
```

---

## Configuração Recomendada

Para melhor performance em classificação:

1. **apply_masks.py**: Modo 4 (Recortado + Vinheta corrigida)
2. **resize_for_model.py**: 224×224 com padding

Isso produz:
- ✅ Lesão isolada e centralizada
- ✅ Iluminação uniforme (vinheta corrigida)
- ✅ Tamanho padronizado sem distorção
- ✅ Pronto para ResNet/EfficientNet