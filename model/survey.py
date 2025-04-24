from sqlalchemy import Column, Integer, String, Boolean, ForeignKey
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.exc import IntegrityError
from __init__ import db

class Survey(db.Model):
    __tablename__ = 'surveys'

    id = db.Column(db.Integer, primary_key=True)
    uid = db.Column(db.String(100), nullable=False, unique=True)  # Unique UID for the user
    name = db.Column(db.String(100), nullable=False)
    username = db.Column(db.String(100), nullable=False, unique=True)
    email = db.Column(db.String(100), nullable=False, unique=True)
    number = db.Column(db.String(20))
    age = db.Column(db.Integer, nullable=False)
    weight = db.Column(db.Integer, nullable=False)
    height = db.Column(db.Integer, nullable=False)
    allergies = db.Column(db.String(255))
    conditions = db.Column(db.String(255))
    ethnicity = db.Column(db.String(100), nullable=False)
    survey_completed = db.Column(db.Boolean, default=False)  # Default to False unless explicitly set

    def __init__(self, uid, name, username, email, number, age, weight, height, allergies, conditions, ethnicity, survey_completed=False):
        self.uid = uid  # Initialize UID
        self.name = name
        self.username = username
        self.email = email
        self.number = number
        self.age = age
        self.weight = weight
        self.height = height
        self.allergies = allergies
        self.conditions = conditions
        self.ethnicity = ethnicity
        self.survey_completed = survey_completed  # Default to False if not provided

    def create(self):
        try:
            # Ensure the user has not already completed the survey
            existing_survey = Survey.query.filter_by(uid=self.uid).first()
            if existing_survey and existing_survey.survey_completed:
                return None  # User has already completed the survey, so we do not create another
            
            # If the survey is new, set survey_completed to False
            self.survey_completed = False

            db.session.add(self)
            db.session.commit()
            return self
        except IntegrityError:
            db.session.rollback()
            return None

    def read(self):
        return {
            "id": self.id,
            "uid": self.uid,  # Return UID in the response
            "name": self.name,
            "username": self.username,
            "email": self.email,
            "number": self.number,
            "age": self.age,
            "weight": self.weight,
            "height": self.height,
            "allergies": self.allergies,
            "conditions": self.conditions,
            "ethnicity": self.ethnicity,
            "survey_completed": self.survey_completed
        }

    def update(self, data):
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
        db.session.commit()
        return self

    def delete(self):
        db.session.delete(self)
        db.session.commit()
        return None

    @staticmethod
    def restore(data):
        for survey_data in data:
            survey = Survey(**survey_data)
            survey.create()
        db.session.commit()

def initSurvey():
    db.create_all()
    
    # Optional: You can prepopulate your database with initial survey data here
    surveys = [
        # Example survey objects (Optional for prepopulation)
    ]

    for survey in surveys:
        try:
            survey.create()
        except IntegrityError:
            db.session.rollback()
            print(f"Error adding survey: {survey.name}")
