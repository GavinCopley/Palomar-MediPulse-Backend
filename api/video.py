"""
Flask blueprint exposing:
  POST /api/optimize   → run optimiser on JSON payload
  GET  /api/health     → simple liveness probe
"""

from flask import Blueprint, request, jsonify
import traceback, sys
from model.optimize import VideoOptimiser, _empty_tip_block

bp = Blueprint("video_opt_api", __name__)
optim = VideoOptimiser()                 # load once at import

@bp.route("/api/optimize", methods=["POST"])
def optimize():
    if not request.is_json:
        return jsonify({"error": "JSON body required"}), 400
    try:
        # Get prediction
        result = optim.suggest(request.get_json(), top_n=5)
        
        # Ensure tips are properly structured
        if "gemini_tips" not in result or not isinstance(result["gemini_tips"], dict):
            result["gemini_tips"] = _empty_tip_block()
        elif "error" in result["gemini_tips"]:
            print("Gemini error:", result["gemini_tips"]["error"])
            result["gemini_tips"] = _empty_tip_block()
            
        return jsonify(result)
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        return jsonify({
            "error": str(e),
            "gemini_tips": _empty_tip_block()
        }), 500

@bp.route("/api/health")
def health():
    return {"status": "ok"}
