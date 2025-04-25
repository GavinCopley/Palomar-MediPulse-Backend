from flask import Blueprint, request, jsonify
import pandas as pd
import math
import os

# Create a Blueprint
hospital_api = Blueprint('hospital_api', __name__, url_prefix='/api')

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the CSV file
csv_path = os.path.join(current_dir, 'hospitals.csv')

# Load hospital data from CSV
try:
    df = pd.read_csv(csv_path)
    print(f"Successfully loaded hospitals.csv from {csv_path}")
    print(f"Found {len(df)} hospital records")
except FileNotFoundError:
    print(f"Warning: hospitals.csv not found at {csv_path}")
    # Create an empty DataFrame with the expected columns
    df = pd.DataFrame(columns=['name', 'location', 'specialties', 'insurance', 'treatments', 'rating'])

@hospital_api.route('/hospitals', methods=['GET'])
def get_hospitals():
    location = request.args.get('location', '').lower()
    specialty = request.args.get('specialty', '').lower()
    insurance = request.args.get('insurance', '').lower()
    treatment = request.args.get('treatment', '').lower()
    rating = float(request.args.get('rating', 1))
    page = int(request.args.get('page', 1))
    per_page = 4

    def matches(row):
        loc = str(row['location']).lower()
        specs = str(row['specialties']).lower()
        ins = str(row['insurance']).lower()
        treat = str(row['treatments']).lower()
        rate = float(row['rating'])

        return (
            (location in loc or 'california' in loc or 'ca' in loc) if location else True
            and (specialty in specs) if specialty else True
            and (insurance in ins) if insurance else True
            and (treatment in treat) if treatment else True
            and (rate >= rating)
        )

    filtered_df = df[df.apply(matches, axis=1)]
    total_results = len(filtered_df)
    total_pages = math.ceil(total_results / per_page)

    # Pagination
    start = (page - 1) * per_page
    end = start + per_page
    results = filtered_df.iloc[start:end]

    return jsonify({
        'page': page,
        'total_pages': total_pages,
        'total_results': total_results,
        'hospitals': results.to_dict(orient='records')
    })
