# Interface de Captura e Segmentação

Este front-end foi reorganizado para mobile (abas) e integra com uma API Python para segmentação de lesões.

## Executar

1) Inicie a API (na raiz do projeto):

```
python -m venv venv
venv\\Scripts\\activate  # Windows
pip install -r requirements.txt
python scripts\\server.py
```

2) Sirva a pasta `Front/` (para liberar a câmera):

```
cd Front
python -m http.server 5500
```

Abra `http://127.0.0.1:5500/` no navegador.

## Fluxo de uso

- Aba Captura: abra a câmera e toque em “Capturar” (botão grande centralizado) ou importe uma imagem.
- Aba Resultado: compare a original recortada (à esquerda) com a segmentada (à direita).
- Escolha “Ficou bom — usar segmentada” ou “Não ficou bom — usar original”.
- Use “Baixar selecionado” para download local ou “Salvar na pasta” (se o navegador suportar File System Access API).

A chamada para a API ocorre em `POST http://127.0.0.1:5000/segment` enviando o arquivo no campo `image`.
