from datetime import datetime
from sqlalchemy.exc import IntegrityError

from __init__ import app, db

class Vehicle(db.Model):
    __tablename__ = "vehicles"
    id = db.Column(db.Integer, primary_key=True)
    _uid = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    _vin = db.Column(db.String(17), unique=True, nullable=False)
    _make = db.Column(db.String(100), nullable=False)
    _model = db.Column(db.String(100), nullable=False)
    _year = db.Column(db.Integer, nullable=False)
    _engine_type = db.Column(db.String(50), nullable=False)
    _date_added = db.Column(db.DateTime, default=datetime.now, nullable=False)

    def __init__(self, vin, make, model, year, engine_type, uid, input_datetime=""):
        self._vin = vin
        self._make = make
        self._model = model
        self._year = year
        self._engine_type = engine_type
        self._uid = uid
        if not input_datetime:
            self._date_posted = datetime.now()
        elif isinstance(input_datetime, str):
            self._date_posted = datetime.fromisoformat(input_datetime)
        elif isinstance(input_datetime, datetime):
            self._date_posted = input_datetime
        else:
            raise TypeError(f"Invalid input_datetime format: {input_datetime}")

    # Property for vin
    @property
    def vin(self):
        return self._vin

    @vin.setter
    def vin(self, value):
        self._vin = value

    def __repr__(self):
        return (f"Vehicle(id={self.id}, user_id={self._uid}), vin={self.vin}, make={self._make}, "
                f"model={self._model}, year={self._year}, engine_type={self._engine_type}, "
                f"date_added={self._date_added})")

    def create(self):
        try:
            db.session.add(self)
            db.session.commit()
        except Exception as error:
            db.session.rollback()
            raise error

    def read(self):
        return {
            "id": self.id,
            "user_id": self._uid,
            "vin": self.vin,  # Use property here
            "make": self._make,
            "model": self._model,
            "year": self._year,
            "engine_type": self._engine_type,
            "date_added": self._date_added,
        }

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, f"_{key}") and value is not None:
                setattr(self, f"_{key}", value)
        try:
            db.session.commit()
        except Exception as error:
            db.session.rollback()
            raise error

    def delete(self):
        try:
            db.session.delete(self)
            db.session.commit()
        except Exception as error:
            db.session.rollback()
            raise error

    @staticmethod
    def restore(data):
        vehicles = {}
        for vehicle_data in data:
            id = vehicle_data.get("id")
            vehicle = Vehicle.query.filter_by(id=id).first()
            if vehicle:
                # Convert date string to datetime object before updating
                if "date_added" in vehicle_data and isinstance(vehicle_data["date_added"], str):
                    vehicle_data["date_added"] = datetime.fromisoformat(vehicle_data["date_added"])
                vehicle.update(**vehicle_data)  # Spread the dictionary as keyword arguments
            else:
                # Handle date_added conversion when creating a new vehicle
                date_added = vehicle_data.get("date_added")
                if isinstance(date_added, str):  # Convert string to datetime if needed
                    date_added = datetime.fromisoformat(date_added)
                elif not isinstance(date_added, datetime):  # Ensure it's a valid datetime object
                    raise TypeError(f"Invalid date_added format: {date_added}")
                # Check for 'user_id' directly if 'user' is not present
                user_id = vehicle_data.get("user_id")
                if user_id is None:
                    raise ValueError(f"Missing or invalid user_id in vehicle_data: {vehicle_data}")
                vehicle = Vehicle(
                    vin=vehicle_data.get("vin"),
                    make=vehicle_data.get("make"),
                    model=vehicle_data.get("model"),
                    year=vehicle_data.get("year"),
                    engine_type=vehicle_data.get("engine_type"),
                    uid=user_id,
                    input_datetime=date_added
                )
                vehicle.create()
        return vehicles




def initVehicles():
    """
    Initialize default vehicles and ensure the Vehicle table has valid data before inserting more entries.
    """
    with app.app_context():
        db.create_all()

        # Add default vehicles
        vehicles = [
            Vehicle(
                vin="3VWJP7ATXEM256789",
                make="Volkswagen",
                model="Beetle",
                year=2014,
                engine_type="Electric",
                uid=3
            )
        ]

        for vehicle in vehicles:
            try:
                vehicle.create()  # Assuming you have a `create()` method in your `Vehicle` class
                print(f"Added vehicle: {vehicle.vin}")
            except IntegrityError as e:
                db.session.rollback()
                print(f"IntegrityError: {e} - Could not add vehicle {vehicle.vin}")
            except Exception as e:
                db.session.rollback()
                print(f"Error: {e} - Could not add vehicle {vehicle.vin}")
