from flask import Flask, send_from_directory, jsonify, request
from flask_cors import CORS
import os
import io
import time
from contextlib import redirect_stdout
from master import run

app = Flask(__name__, static_folder='../', static_url_path='/')
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/')
def serve_frontend():
    return send_from_directory(app.static_folder, 'frontend.html')

@app.route('/<path:path>')
def serve_static_files(path):
    return send_from_directory(app.static_folder, path)

@app.route("/health")
def health():
    return jsonify({"ok": True, "status": "Server running", "time": time.time()})


@app.route("/run", methods=["POST"])
def run_endpoint():
    start = time.time()
    try:
        payload = request.get_json(force=True)
        user_input = payload.get("input", "").strip()

        if not user_input:
            return jsonify({
                "ok": False,
                "error": "No input provided. Please type your workout issue before running analysis."
            }), 400

        print(f"🧠 Received input: {user_input}")

        # call master.py's run() → returns dict
        run_result = run(user_input)
        final = run_result["result"]
        printed = run_result["printed"]

        elapsed = int((time.time() - start) * 1000)
        print(f"✅ Analysis completed in {elapsed} ms")

        return jsonify({
            "ok": True,
            "result": final,
            "printed": printed,
            "elapsed_ms": elapsed
        })

    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        print(f"❌ Server error: {err_msg}")
        return jsonify({
            "ok": False,
            "error": str(e),
            "trace": err_msg
        }), 500


if __name__ == '__main__':
  print("Starting server on http://0.0.0.0:5000")
  app.run(host="0.0.0.0", port=5000, debug=True)
