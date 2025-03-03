from flask import Blueprint, request, jsonify, Response, g
from flask_restful import Api, Resource
import requests
from sqlalchemy.exc import SQLAlchemyError
from __init__ import db
from model.vehicle import Vehicle
from api.jwt_authorize import token_required
from model.user import User

# Create a Blueprint for the VIN decoding functionality
vinStore_api = Blueprint('vinStore_api', __name__, url_prefix='/api')
api = Api(vinStore_api)

# Base URL for the NHTSA VIN decoding API
NHTSA_API_URL = "https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVinValues/"

class VINDecodeAPI:
    class _CRUD(Resource):
        @token_required()
        def post(self):
            # Retrieve the current user from the context (via JWT or session)
            current_user = g.current_user  # This assumes you are using g.current_user, set in your token_required decorator

            # Check if the current user is authenticated
            if not current_user:
                return Response("{'message': 'User is not authenticated'}", 401)

            # Parse the VIN from the request
            data = request.get_json()
            vin = data.get('vin')

            # Validate the VIN
            if not vin:
                return Response("{'message': 'VIN is required'}", status=400, mimetype='application/json')

            if len(vin) != 17:
                return Response("{'message': 'VIN must be 17 characters long'}", status=400, mimetype='application/json')

            # Request data from the NHTSA API
            response = requests.get(f"{NHTSA_API_URL}{vin}?format=json")
            
            if response.status_code != 200:
                return Response("{'message': 'Failed to retrieve data from NHTSA API'}", status=500, mimetype='application/json')

            vin_data = response.json().get('Results', [{}])[0]

            # Extract necessary details
            make = vin_data.get('Make')
            model = vin_data.get('Model')
            year = vin_data.get('ModelYear')
            engine_type = vin_data.get('FuelTypePrimary')

            if not all([make, model, year, engine_type]):
                return Response("{'message': 'Unable to get data from VIN'}", status=500, mimetype='application/json')

            # Check if the vehicle already exists in the database
            existing_vehicle = Vehicle.query.filter_by(vin=vin).first()
            if existing_vehicle:
                return jsonify({"message": "Vehicle already exists in the database", "vehicle": existing_vehicle.read()})

            # Create a new Vehicle object, associating it with the current user
            new_vehicle = Vehicle(vin=vin, make=make, model=model, year=year, engine_type=engine_type, uid=current_user.id)

            try:
                # Save the vehicle to the database
                db.session.add(new_vehicle)
                db.session.commit()
            except SQLAlchemyError as e:
                db.session.rollback()
                return Response(f"{{'message': 'Database error. User could already have VIN: {str(e)}'}}", status=500, mimetype='application/json')

            return jsonify(new_vehicle.read())
        
        @token_required()
        def get(self):
            """
            Retrieve vehicles for the current user.

            Retrieves a list of vehicles associated with the authenticated user.

            Returns:
                JSON response with a list of vehicle dictionaries.
            """
            # Retrieve the current user from the token_required authentication check  
            current_user = g.current_user

            if not current_user:
                return Response("{'message': 'User is not authenticated'}", 401)

            # Query the database for vehicles associated with the current user
            user_vehicles = Vehicle.query.filter_by(_uid=current_user.id).all()

            if not user_vehicles:
                return Response("{'message': 'No vehicles found for the current user'}", 404)

            # Prepare a JSON list of the user's vehicles
            json_ready = [vehicle.read() for vehicle in user_vehicles]

            # Return response, a list of the user's vehicles in JSON format
            return jsonify(json_ready)

        @token_required()
        def put(self):
            # Retrieve the current user from the context
            current_user = g.current_user

            if not current_user:
                return jsonify({"message": "User is not authenticated"}), 401

            # Parse the data from the request
            data = request.get_json()
            old_vin = data.get('old_vin')  # The current VIN of the vehicle in the database
            new_vin = data.get('new_vin')  # The new VIN to update to

            # Validate the input
            if not old_vin or not new_vin:
                return Response("{'message': 'Both old_vin and new_vin are required'}", status=400, mimetype='application/json')

            if len(new_vin) != 17:
                return Response("{'message': 'new_vin must be 17 characters long'}", status=400, mimetype='application/json')

            # Find the vehicle by old VIN and ensure it belongs to the current user
            vehicle = Vehicle.query.filter_by(_vin=old_vin, _uid=current_user.id).first()
            if not vehicle:
                return Response("{'message': 'Vehicle not found or does not belong to the user'}", status=404, mimetype='application/json')

            # Fetch updated vehicle information from the NHTSA API
            response = requests.get(f"{NHTSA_API_URL}{new_vin}?format=json")

            if response.status_code != 200:
                return Response("{'message': 'Failed to retrieve data from NHTSA API'}", status=500, mimetype='application/json')

            vin_data = response.json().get('Results', [{}])[0]

            # Extract necessary details
            make = vin_data.get('Make')
            model = vin_data.get('Model')
            year = vin_data.get('ModelYear')
            engine_type = vin_data.get('FuelTypePrimary')

            if not all([make, model, year, engine_type]):
                return Response("{'message': 'Incomplete data from NHTSA API'}", status=500, mimetype='application/json')

            # Update the vehicle's VIN and other details
            try:
                vehicle.vin = new_vin  # Update VIN
                vehicle._make = make
                vehicle._model = model
                vehicle._year = year
                vehicle._engine_type = engine_type

                db.session.commit()
            except SQLAlchemyError as e:
                db.session.rollback()
                return Response(f"{{'message': 'Database error: {str(e)}'}}", status=500, mimetype='application/json')

            return jsonify({"message": "Vehicle VIN updated successfully", "vehicle": vehicle.read()})
        
        @token_required("Admin")
        def delete(self):
            """
            Delete a vehicle record by VIN.
            Only users with the 'Admin' role are authorized to delete vehicles.

            Returns:
                JSON response confirming deletion or an error message.
            """
            # Parse and normalize the VIN
            data = request.get_json()
            vin = data.get('vin', '').strip().upper()

            if not vin:
                return Response("{'message': 'VIN is required for deletion'}", status=400, mimetype='application/json')

            # Find the vehicle by VIN (stored as _vin in the database)
            from sqlalchemy import func
            vehicle = Vehicle.query.filter(func.upper(Vehicle._vin) == vin).first()

            if not vehicle:
                return Response(f"{{'message': 'Vehicle with VIN {vin} not found'}}", status=404, mimetype='application/json')

            try:
                # Delete the vehicle record
                db.session.delete(vehicle)
                db.session.commit()
                return jsonify({"message": f"Vehicle with VIN {vin} has been successfully deleted."})
            except SQLAlchemyError as e:
                db.session.rollback()
                return Response(f"{{'message': 'Database error: {str(e)}'}}", status=500, mimetype='application/json')



    api.add_resource(_CRUD, '/vinStore')
