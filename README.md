# Projeto de Processamento de Imagens de Pele

Este projeto contém uma coleção de scripts Python para pré-processamento e aumento de dados (data augmentation) de imagens de lesões de pele, útil para tarefas de visão computacional e aprendizado de máquina.

## Funcionalidades

- **Aumento de Dados (`augment_skin_images.py`):** Gera novas imagens com variações de rotação, zoom, brilho, contraste, ruído, etc., para expandir um conjunto de dados.
- **Extração de ROI (`circle_to_square.py`):** Detecta a região de interesse circular principal em imagens de dermatoscopia (especialmente aquelas com fundo transparente), recorta-a em um formato quadrado e a salva.
- **Correção de Cor (`color_constancy_clahe.py`):** Padroniza a coloração das imagens aplicando o algoritmo de constância de cor "Gray-World" e melhora o contraste local com CLAHE.

## Scripts

- **`augment_skin_images.py`**: Pede uma pasta de entrada e o número total de imagens desejado. Ele então gera novas imagens aumentadas até atingir esse total e as salva na subpasta `imgOutput`.
- **`circle_to_square.py`**: Processa imagens para detectar a área circular principal, recorta-a como um quadrado e a salva na subpasta `roiSquare`. É ideal para imagens PNG circulares com canal alfa.
- **`color_constancy_clahe.py`**: Aplica o balanceamento de branco (Gray-World) e, em seguida, o algoritmo CLAHE (no espaço de cor Lab ou HSV, à escolha do usuário). As imagens processadas são salvas em `colorFixed_Lab` ou `colorFixed_HSV`.

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

## Como Usar

Cada script é interativo e solicitará o caminho para a pasta com as imagens que você deseja processar.

1.  Coloque suas imagens em uma pasta.
2.  Execute o script desejado a partir do seu terminal. Por exemplo:

    ```bash
    python color_constancy_clahe.py
    ```
3.  Siga as instruções no terminal (informe o caminho da pasta e outras opções, se solicitado).
4.  As imagens processadas serão salvas em uma nova subpasta criada dentro do diretório que você forneceu.
