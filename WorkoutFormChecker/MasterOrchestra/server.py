#!/usr/bin/env python3
"""
server.py

Flask server to run MasterOrchestra/master.py using the importable run(user_input) entrypoint.
- POST /run accepts JSON:
    { "input": "text to analyze", "timeout": seconds (optional, default 30) }
  or:
    { "mode": "import"|"subprocess", "args": [...], "timeout": seconds }

Behavior:
- Prefer import-and-call (fast): import the master module and call run(user_input).
- Runs the call in a ThreadPoolExecutor to enforce a timeout.
- If import/call fails, attempts a subprocess fallback.
- Returns JSON: { ok: bool, result: ..., output: printed_stdout, stderr: ..., elapsed_ms: N, error?: ... }
"""
import os
import sys
import time
import json
import subprocess
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from importlib import util as importlib_util
from contextlib import redirect_stdout
import io

from flask import Flask, request, jsonify, Response
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Path to master.py (adjust if you place server.py elsewhere)
MASTER_REL_PATH = os.path.join("WorkoutFormChecker", "MasterOrchestra", "master.py")
MASTER_PATH = os.path.join(os.getcwd(), MASTER_REL_PATH)

EXECUTOR = ThreadPoolExecutor(max_workers=2)
DEFAULT_TIMEOUT = 30  # seconds

def import_and_run_master(user_input):
    """
    Import the master.py module from MASTER_PATH and call its run(user_input) function.
    Returns a tuple: (ok, result_obj_or_str, printed_stdout, elapsed_ms)
    """
    start = time.time()
    if not os.path.exists(MASTER_PATH):
        raise FileNotFoundError(f"master.py not found at {MASTER_PATH}")

    spec = importlib_util.spec_from_file_location("master_module", MASTER_PATH)
    module = importlib_util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:
        # propagate import errors
        raise

    if not hasattr(module, "run"):
        raise AttributeError("Imported master module has no run(user_input) entrypoint")

    buf = io.StringIO()
    result_value = None
    try:
        with redirect_stdout(buf):
            # call run; assuming run returns a JSON-serializable Python object
            result_value = module.run(user_input)
    except Exception:
        # propagate execution errors (caller will handle fallback or report)
        raise

    printed = buf.getvalue() or ""
    elapsed = int((time.time() - start) * 1000)
    return True, result_value, printed, elapsed

def run_master_subprocess(args=None, timeout=DEFAULT_TIMEOUT):
    """
    Run master.py as a subprocess and capture stdout/stderr.
    Returns: (ok_bool, stdout, stderr, returncode, elapsed_ms)
    """
    start = time.time()
    if not os.path.exists(MASTER_PATH):
        raise FileNotFoundError(f"master.py not found at {MASTER_PATH}")

    cmd = [sys.executable, MASTER_PATH]
    if args:
        cmd += list(args)
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
        elapsed = int((time.time() - start) * 1000)
        ok = proc.returncode == 0
        return ok, proc.stdout, proc.stderr, proc.returncode, elapsed
    except subprocess.TimeoutExpired:
        raise TimeoutError("Subprocess timed out")

def safe_jsonify(obj):
    """
    Try to jsonify obj; if it fails due to non-serializable values, fall back to JSON with default=str.
    Returns a Flask Response with application/json.
    """
    try:
        return jsonify(obj)
    except Exception:
        try:
            text = json.dumps(obj, default=str)
            return Response(text, mimetype="application/json")
        except Exception:
            # Last resort: return string
            return jsonify({"ok": False, "error": "Unable to serialize result", "raw": str(obj)})

@app.route("/run", methods=["POST"])
def run_endpoint():
    """
    POST /run
    Accepts JSON body and returns JSON response.
    Preferred JSON shapes:
      { "input": "describe the issue", "timeout": 30 }
    Or:
      { "mode": "subprocess", "args": ["..."], "timeout": 30 }
    """
    body = request.get_json(silent=True) or {}
    mode = body.get("mode", "import")
    timeout = int(body.get("timeout", DEFAULT_TIMEOUT))

    # Determine user_input from possible fields
    user_input = None
    if "input" in body and body["input"]:
        user_input = body["input"]
    else:
        # If args provided, join them to form an input string
        args = body.get("args") or []
        if args:
            user_input = " ".join(args)
        else:
            # fallback default test prompt
            user_input = "I did squats today and my right knee hurts when standing up"

    if mode == "subprocess":
        # Run subprocess directly
        try:
            ok, out, err, code, elapsed = run_master_subprocess(args=[user_input], timeout=timeout)
            resp = {"ok": ok, "output": out.strip(), "stderr": err.strip(), "returncode": code, "elapsed_ms": elapsed}
            return safe_jsonify(resp)
        except Exception as e:
            tb = traceback.format_exc()
            return safe_jsonify({"ok": False, "error": str(e), "trace": tb}), 500

    # Default: import + call run(user_input)
    future = EXECUTOR.submit(import_and_run_master, user_input)
    try:
        ok, result_obj, printed, elapsed = future.result(timeout=timeout + 1)
        # Build response
        resp = {"ok": True, "result": result_obj, "output": printed.strip(), "elapsed_ms": elapsed}
        return safe_jsonify(resp)
    except TimeoutError:
        future.cancel()
        return safe_jsonify({"ok": False, "error": "Execution timed out", "elapsed_ms": timeout * 1000}), 504
    except Exception as e:
        tb = traceback.format_exc()
        # Attempt subprocess fallback
        try:
            ok2, out2, err2, code2, elapsed2 = run_master_subprocess(args=[user_input], timeout=timeout)
            return safe_jsonify({
                "ok": ok2,
                "output": out2.strip(),
                "stderr": err2.strip(),
                "returncode": code2,
                "fallback": "subprocess",
                "elapsed_ms": elapsed2,
                "import_error": str(e),
                "import_trace": tb
            })
        except Exception as e2:
            tb2 = traceback.format_exc()
            return safe_jsonify({"ok": False, "error": str(e2), "trace": tb2, "import_error": str(e), "import_trace": tb}), 500

if __name__ == "__main__":
    print("Starting server on http://0.0.0.0:5000 (development mode)")
    app.run(host="0.0.0.0", port=5000, debug=True)
