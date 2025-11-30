from flask import Flask, request, Response, jsonify
from flask_cors import CORS
from pathlib import Path
from datetime import datetime
import secrets

from autoSegmentation.service import segment_lesion_bytes
from classification.service import classify_image_bytes


def create_app() -> Flask:
    root = Path(__file__).resolve().parents[1]
    front_dir = root / 'Front'

    app = Flask(__name__, static_folder=str(front_dir), static_url_path='')
    CORS(app)  # libera para http://localhost e file-based dev

    @app.get('/')
    def index():
        return app.send_static_file('index.html')

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/segment")
    def segment():
        if 'image' not in request.files:
            return jsonify({"error": "Envie um arquivo em 'image' (multipart/form-data)"}), 400
        f = request.files['image']
        data = f.read()
        try:
            png_bytes = segment_lesion_bytes(data)
        except Exception as e:
            return jsonify({"error": str(e)}), 400
        return Response(png_bytes, mimetype='image/png')

    @app.post("/save")
    def save():
        """Salva a imagem enviada em uma pasta local do projeto.
        Campo esperado: 'image' (arquivo), opcional 'kind' em {original,segmented}.
        """
        if 'image' not in request.files:
            return jsonify({"error": "Envie um arquivo em 'image' (multipart/form-data)"}), 400
        f = request.files['image']
        kind = request.form.get('kind', 'chosen')

        # Pasta no raiz do projeto: ../imagens_salvas
        out_dir = root / 'imagens_salvas'
        out_dir.mkdir(exist_ok=True)

        now = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        rand = secrets.token_hex(3)
        ext = Path(f.filename).suffix or '.png'
        name = f"{kind}_{now}_{rand}{ext}"
        out_path = out_dir / name
        f.save(out_path)
        return jsonify({"saved": True, "path": str(out_path)})

    @app.post("/classify")
    def classify():
        if 'image' not in request.files:
            return jsonify({"error": "Envie um arquivo em 'image' (multipart/form-data)"}), 400
        data = request.files['image'].read()
        try:
            result = classify_image_bytes(data)
        except FileNotFoundError as e:
            return jsonify({"error": str(e)}), 500
        except Exception as e:
            return jsonify({"error": str(e)}), 400
        return jsonify(result)

    return app


if __name__ == "__main__":
    app = create_app()
    # Porta padrão 5000 para alinhar com a chamada do front
    app.run(host="0.0.0.0", port=5000, debug=True)
