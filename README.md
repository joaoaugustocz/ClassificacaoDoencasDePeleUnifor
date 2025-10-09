# Classificação de Doenças de Pele - UNIFOR

Sistema de processamento e classificação de imagens dermatológicas para análise de lesões de pele usando deep learning.

## 📋 Sobre o Projeto

Este projeto implementa um pipeline completo de pré-processamento de imagens médicas para classificação de lesões de pele, com foco no dataset HAM10000. Inclui ferramentas para:

- Aplicação de máscaras de segmentação
- Correção de vinheta (iluminação não-uniforme)
- Normalização de cor e contraste
- Data augmentation
- Preparação de imagens para modelos de deep learning

## 📁 Estrutura do Projeto

```
ClassificacaoDoencasDePeleUnifor/
├── scripts/                          # Scripts organizados por categoria
│   ├── preprocessing/                # Pré-processamento geral
│   │   ├── circle_to_square.py      # Extração de ROI circular
│   │   └── color_constancy_clahe.py # Normalização de cor
│   ├── augmentation/                 # Data augmentation
│   │   └── augment_skin_images.py   # Geração de augmentations
│   └── ham10000_pipeline/            # Pipeline HAM10000
│       ├── apply_masks.py           # Aplicação de máscaras + correção vinheta
│       └── resize_for_model.py      # Redimensionamento para modelos
├── HAM10000/                         # Dataset HAM10000
│   ├── images/                       # Imagens originais
│   ├── masks/                        # Máscaras de segmentação
│   └── model_ready_*/                # Imagens prontas para treino
└── requirements.txt                  # Dependências Python
```

## 🚀 Pipeline Recomendado (HAM10000)

```bash
# 1. Aplicar máscaras + correção de vinheta
python scripts/ham10000_pipeline/apply_masks.py
# → Escolha modo 4 (Recortado + Vinheta corrigida)

# 2. Redimensionar para modelo
python scripts/ham10000_pipeline/resize_for_model.py
# → Escolha 224×224 com padding

# 3. Treinar modelo
# Usar imagens de: HAM10000/model_ready_224x224_padded/
```

## Instalação e Configuração

1.  **Clone o repositório:**
    ```bash
    git clone <URL_DO_REPOSITORIO>
    cd <NOME_DA_PASTA_DO_PROJETO>
    ```

2.  **Crie e ative um ambiente virtual (Recomendado):**
    ```bash
    # Criar o ambiente
    python -m venv venv

    # Ativar no Windows
    .\venv\Scripts\activate

    # Ativar no macOS/Linux
    source venv/bin/activate
    ```

3.  **Instale as dependências:**
    O arquivo `requirements.txt` contém todas as bibliotecas necessárias. Instale-as com o seguinte comando:
    ```bash
    pip install -r requirements.txt
    ```

## 📚 Documentação dos Scripts

Cada categoria de scripts possui documentação detalhada:

- **[scripts/README.md](scripts/README.md)** - Visão geral e fluxo de trabalho
- **[scripts/preprocessing/](scripts/preprocessing/)** - Scripts de pré-processamento geral
- **[scripts/augmentation/](scripts/augmentation/)** - Scripts de data augmentation
- **[scripts/ham10000_pipeline/](scripts/ham10000_pipeline/)** - Pipeline completo HAM10000

## 🔧 Scripts Disponíveis

### Pipeline HAM10000 (Principal)
```bash
# Aplicar máscaras + correção de vinheta
python scripts/ham10000_pipeline/apply_masks.py

# Redimensionar para modelo
python scripts/ham10000_pipeline/resize_for_model.py
```

### Pré-processamento Geral
```bash
# Extrair ROI circular
python scripts/preprocessing/circle_to_square.py

# Normalização de cor (Gray-World + CLAHE)
python scripts/preprocessing/color_constancy_clahe.py
```

### Data Augmentation
```bash
# Gerar augmentations
python scripts/augmentation/augment_skin_images.py
```

Todos os scripts são interativos e guiarão você através das opções disponíveis.