# api/hospital_api.py

import pandas as pd
from flask import Blueprint, request, jsonify

bp = Blueprint('hospital_api', __name__)

def load_hospitals_df(disease):
    """
    Load your hospital CSV and filter by the requested performance measure (disease).
    Adjust the file path and column names as needed.
    """
    df = pd.read_csv('path/to/your/hospital_data.csv')  # e.g. 'data/hospitals.csv'
    # Assume column 'Performance Measure' holds the disease/measure name
    df = df[df['Performance Measure'] == disease].copy()
    # Ensure latitude/longitude columns are named appropriately
    df.rename(columns={'Latitude': 'latitude', 'Longitude': 'longitude'}, inplace=True)
    return df

class HospitalRecommender:
    """Recommends hospitals based on weighted scoring of distance, quality, experience, and safety."""

    def __init__(self, distance_weight=2.0, quality_weight=1.0, experience_weight=1.0, safety_weight=1.0):
        # Weigh distance more heavily by default
        self.weights = {
            'distance': distance_weight,
            'quality': quality_weight,
            'experience': experience_weight,
            'safety': safety_weight
        }

    def recommend(self, hospitals_df, user_lat, user_lon, radius, limit):
        # Copy and compute haversine distance
        df = hospitals_df.copy()
        df['distance_mi'] = df.apply(
            lambda row: self._haversine(user_lon, user_lat, row['longitude'], row['latitude']), axis=1
        )
        # Filter by max radius
        df = df[df['distance_mi'] <= radius]

        # Fill missing component scores with zero
        df['score_distance']   = df['score_distance'].fillna(0)
        df['score_quality']    = df['score_quality'].fillna(0)
        df['score_experience'] = df['score_experience'].fillna(0)
        df['score_safety']     = df['score_safety'].fillna(0)

        # Compute weighted combined score
        total_weight = sum(self.weights.values())
        df['predicted_score'] = (
            df['score_distance']   * self.weights['distance'] +
            df['score_quality']    * self.weights['quality'] +
            df['score_experience'] * self.weights['experience'] +
            df['score_safety']     * self.weights['safety']
        ) / total_weight

        # Sort by score descending and take top N
        df = df.sort_values(by='predicted_score', ascending=False).head(limit)
        return df.to_dict(orient='records')

    def _haversine(self, lon1, lat1, lon2, lat2):
        # Haversine formula to compute distance in miles
        from math import radians, cos, sin, asin, sqrt
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        return 3959 * c  # Earth radius in miles

@bp.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Unpack payload
    disease = data.get('disease')
    lat     = data.get('lat')
    lon     = data.get('lon')
    radius  = data.get('radius', 50)
    limit   = data.get('limit', 5)

    # Load and filter your hospital data
    hospitals_df = load_hospitals_df(disease)

    # Instantiate recommender with higher distance weight
    recommender = HospitalRecommender(
        distance_weight=2.0,
        quality_weight=1.0,
        experience_weight=1.0,
        safety_weight=1.0
    )

    # Get recommendations
    results = recommender.recommend(hospitals_df, lat, lon, radius, limit)

    # Return JSON response
    return jsonify({'recommended_hospitals': results})
