"""
Flask blueprint exposing:
  POST /api/optimize   → run optimiser on JSON payload
  GET  /api/health     → simple liveness probe
"""

from flask import Blueprint, request, jsonify
from model.optimize import VideoOptimiser

bp = Blueprint("video_opt_api", __name__)
optim = VideoOptimiser()                 # load once at import

@bp.route("/api/optimize", methods=["POST"])
def optimize():
    if not request.is_json:
        return jsonify({"error": "JSON body required"}), 400
    try:
        result = optim.suggest(request.get_json(), top_n=5)
        return jsonify(result)
    except Exception as e:
        import traceback, sys
        traceback.print_exc(file=sys.stderr)
        return jsonify({"error": str(e)}), 500

@bp.route("/api/health")
def health():
    return {"status": "ok"}
