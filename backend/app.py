from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS  # type: ignore
import os
from predict import predict

app = Flask(__name__)

# Configure CORS: if FRONTEND_ORIGIN is set, restrict to that origin (production).
frontend_origin = os.getenv("FRONTEND_ORIGIN")
if frontend_origin:
    CORS(app, origins=[frontend_origin])
else:
    # Liberal CORS for development convenience
    CORS(app)

@app.route("/api/analyze", methods=["POST"])
def analyze_api():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    f = request.files["image"]
    save_path = os.path.join(os.path.dirname(__file__), "uploaded_image.jpg")
    f.save(save_path)

    try:
        result = predict(save_path)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify(result)


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


# Optional: serve a built frontend from ../frontend/build when enabled.
# Enable by setting environment variable `SERVE_FRONTEND=1` and placing
# your frontend build output in `frontend/build` relative to the repo root.
if os.getenv("SERVE_FRONTEND") == "1":
    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def serve_frontend(path):
        build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend", "build"))
        index_path = os.path.join(build_dir, "index.html")
        if path and os.path.exists(os.path.join(build_dir, path)):
            return send_from_directory(build_dir, path)
        if os.path.exists(index_path):
            return send_from_directory(build_dir, "index.html")
        return jsonify({"error": "Frontend build not found"}), 404


if __name__ == "__main__":
    # Respect environment variables for production readiness
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "5000"))
    debug_env = os.getenv("FLASK_DEBUG", "0")
    debug = True if debug_env == "1" or os.getenv("FLASK_ENV") == "development" else False
    app.run(host=host, port=port, debug=debug)
