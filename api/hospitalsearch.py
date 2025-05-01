from flask import Blueprint, request, jsonify
import pandas as pd
import math
import os
import numpy as np

# Create a Blueprint with a unique name
hospital_search_api = Blueprint('hospital_search_api', __name__, url_prefix='/api/hospital-search')

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the CSV file
csv_path = os.path.join(current_dir, 'hospitals.csv')

# Load hospital data from CSV
try:
    df = pd.read_csv(csv_path)
    # Replace NaN values with None, which will be converted to null in JSON
    df = df.replace({np.nan: None})
    print(f"Successfully loaded hospitals.csv from {csv_path}")
    print(f"Found {len(df)} hospital records")
except FileNotFoundError:
    print(f"Warning: hospitals.csv not found at {csv_path}")
    # Create an empty DataFrame with the expected columns
    df = pd.DataFrame(columns=['name', 'location', 'specialties', 'insurance', 'treatments', 'rating',
                              'visiting_hours', 'phone', 'website', 'email', 'emergency_services',
                              'parking_accessibility', 'patient_review', 'departments'])

@hospital_search_api.route('', methods=['GET'])
def get_hospitals():
    location = request.args.get('location', '').lower()
    specialty = request.args.get('specialty', '').lower()
    insurance = request.args.get('insurance', '').lower()
    treatment = request.args.get('treatment', '').lower()
    rating = float(request.args.get('rating', 1))
    emergency = request.args.get('emergency', '').lower()
    department = request.args.get('department', '').lower()
    page = int(request.args.get('page', 1))
    per_page = 4

    def matches(row):
        # Handle None/NaN values safely
        loc = str(row['location']).lower() if row['location'] is not None else ''
        specs = str(row['specialties']).lower() if row['specialties'] is not None else ''
        ins = str(row['insurance']).lower() if row['insurance'] is not None else ''
        treat = str(row['treatments']).lower() if row['treatments'] is not None else ''
        
        # Use get for fields that might not exist in all records
        emerg = str(row.get('emergency_services', '')).lower() if row.get('emergency_services') is not None else ''
        dept = str(row.get('departments', '')).lower() if row.get('departments') is not None else ''
        
        # Handle rating separately since it's numeric
        try:
            rate = float(row['rating'])
        except (ValueError, TypeError):
            rate = 0.0

        return (
            (location in loc or 'california' in loc or 'ca' in loc) if location else True
            and (specialty in specs) if specialty else True
            and (insurance in ins) if insurance else True
            and (treatment in treat) if treatment else True
            and (emergency in emerg) if emergency else True
            and (department in dept) if department else True
            and (rate >= rating)
        )

    filtered_df = df[df.apply(matches, axis=1)]
    total_results = len(filtered_df)
    total_pages = math.ceil(total_results / per_page)

    # Pagination
    start = (page - 1) * per_page
    end = start + per_page
    results = filtered_df.iloc[start:end]

    # Convert DataFrame to dict with proper handling of NaN values
    hospitals_list = results.replace({np.nan: None}).to_dict(orient='records')

    return jsonify({
        'page': page,
        'total_pages': total_pages,
        'total_results': total_results,
        'hospitals': hospitals_list
    })

@hospital_search_api.route('/<hospital_name>', methods=['GET'])
def get_hospital_details(hospital_name):
    """Get detailed information for a specific hospital by name"""
    # Convert hospital name to lowercase for case-insensitive matching
    hospital_name_lower = hospital_name.lower()
    
    # Find the hospital in the dataframe
    matching_hospitals = df[df['name'].str.lower() == hospital_name_lower]
    
    if not matching_hospitals.empty:
        # Return the first matching hospital's details
        # Replace NaN with None before converting to dict
        hospital_data = matching_hospitals.iloc[0].replace({np.nan: None}).to_dict()
        return jsonify({
            'status': 'success',
            'hospital': hospital_data
        })
    else:
        return jsonify({
            'status': 'error',
            'message': f'Hospital with name "{hospital_name}" not found'
        }), 404
