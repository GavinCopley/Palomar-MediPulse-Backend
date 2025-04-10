from sqlalchemy import Column, Integer, String, Boolean, ForeignKey
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.exc import IntegrityError
from __init__ import db 

class Survey(db.Model):
    __tablename__ = 'surveys'

    id = db.Column(db.Integer, primary_key=True)
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
    survey_completed = db.Column(db.Boolean, default=False)  # New field added

    def __init__(self, name, username, email, number, age, weight, height, allergies, conditions, ethnicity, survey_completed=False):
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
        self.survey_completed = survey_completed

    def create(self):
        try:
            db.session.add(self)
            db.session.commit()
            return self
        except IntegrityError:
            db.session.rollback()
            return None

    def read(self):
        return {
            "id": self.id,
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
    
    surveys = [
        Survey(name="John Doe", username="john_doe", email="john@example.com", number="1234567890", age=30, weight=180, height=70, allergies="Peanuts", conditions="None", ethnicity="Caucasian"),
        Survey(name="Jane Smith", username="jane_smith", email="jane@example.com", number="0987654321", age=25, weight=135, height=66, allergies="Dust", conditions="Asthma", ethnicity="Asian")
    ]

    for survey in surveys:
        try:
            survey.create()
        except IntegrityError:
            db.session.rollback()
            print(f"Error adding survey: {survey.name}")
