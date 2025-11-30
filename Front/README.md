# Interface de Captura, Segmentação e Classificação

Frontend responsivo (abas, foco em uso mobile) integrado ao backend Flask que faz segmentação automática e classificação (Benigno, Maligno ou Pré-Maligno) usando o InceptionV4.

## Como executar

1. Na raiz do projeto, suba o backend (que também serve o Front) com o orquestrador:

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python startApp.py
```

O script sobe o Flask em `http://127.0.0.1:5000/` (também disponível pelo IP local) e, se o binário `cloudflared` estiver instalado, publica um túnel HTTPS. Siga o link exibido no console para abrir no celular.

## Fluxo de uso

- Aba Captura: abra a câmera e toque em **Capturar** (ou importe uma imagem). O recorte circular é aplicado automaticamente.
- Aba Resultado: compare a imagem original recortada (esquerda) com a segmentada (direita), escolha “Não ficou bom — usar original” ou “Ficou bom — usar segmentada”.
- Botões adicionais:
  - **Classificar**: envia a imagem escolhida para `/classify` (InceptionV4) e mostra o rótulo e as probabilidades.
  - **Baixar selecionado**: salva localmente.
  - **Salvar na pasta**: usa File System Access API (quando suportado) para gravar em uma pasta escolhida.

APIs servidas pelo backend Flask (todas `POST` com campo `image`):

- `/segment` – aplica a segmentação e retorna PNG RGBA.
- `/save` – grava em `imagens_salvas/` na raiz.
- `/classify` – classifica usando os últimos pesos do InceptionV4.
