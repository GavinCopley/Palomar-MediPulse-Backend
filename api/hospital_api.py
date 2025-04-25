# api/hospital_api.py
from flask import Blueprint, request, jsonify
from model.hospital_recommender import HospitalRecommender

bp = Blueprint("hospital_api", __name__, url_prefix="/api")

recommender = HospitalRecommender()        # lazy-loads the model

@bp.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    try:
        disease = data["disease"]
        lat     = float(data["lat"])
        lon     = float(data["lon"])
    except (KeyError, ValueError) as exc:
        return jsonify(error=f"Missing or invalid field → {exc}"), 400

    radius = float(data.get("radius", 50))
    limit  = int(data.get("limit", 3))

    try:
        results = recommender.recommend(
            disease=disease,
            user_lat=lat,
            user_lon=lon,
            radius_miles=radius,
            top_n=limit
        )
    except Exception as exc:
        return jsonify(error=str(exc)), 500

    if not results:
        return jsonify(error="No hospital matched your criteria."), 200

    return jsonify(recommended_hospitals=results), 200
