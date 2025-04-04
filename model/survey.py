from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from __init__ import db 

Base = declarative_base()
class Survey(Base):
    __tablename__ = 'surveys'

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    username = Column(String, nullable=False, unique=True)
    email = Column(String, nullable=False, unique=True)
    number = Column(String, nullable=True)
    age = Column(Integer, nullable=False)
    weight = Column(Integer, nullable=False)
    height = Column(String, nullable=False)
    allergies = Column(String, nullable=True)
    conditions = Column(String, nullable=True)
    ethnicity = Column(String, nullable=False)

    def __init__(self, name, username, email, number, age, weight, height, allergies, conditions, ethnicity):
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

    def create(self):
        """Add the survey record to the database"""
        try:
            db.session.add(self)
            db.session.commit()
        except IntegrityError:
            db.session.rollback()
            raise Exception("Survey already exists or error adding survey.")

    def update(self, inputs):
        """Update survey fields."""
        if not isinstance(inputs, dict):
            return self

        for key, value in inputs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        try:
            db.session.commit()
        except IntegrityError:
            db.session.rollback()
            return None
        return self

    def read(self):
        """Return survey data as a dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'username': self.username,
            'email': self.email,
            'number': self.number,
            'age': self.age,
            'weight': self.weight,
            'height': self.height,
            'allergies': self.allergies,
            'conditions': self.conditions,
            'ethnicity': self.ethnicity
        }

    @staticmethod
    def restore(data):
        """Restore survey data from input and add to database."""
        for survey_data in data:
            survey = Survey(**survey_data)
            survey.create()
        db.session.commit()

def initSurvey():
    """Create the survey table and add some sample data."""
    db.create_all()

    # Sample data
    surveys = [
        Survey(name="John Doe", username="john_doe", email="john@example.com", number="1234567890", age=30, weight=180, height="5'10", allergies="Peanuts", conditions="None", ethnicity="Caucasian"),
        Survey(name="Jane Smith", username="jane_smith", email="jane@example.com", number="0987654321", age=25, weight=135, height="5'6", allergies="Dust", conditions="Asthma", ethnicity="Asian")
    ]

    for survey in surveys:
        try:
            db.session.add(survey)
            db.session.commit()
        except IntegrityError:
            db.session.rollback()
            print(f"Error adding survey: {survey.name}")
