from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.exc import IntegrityError
from __init__ import db 

class Survey(db.Model):
    __tablename__ = 'surveys'

    id = Column(Integer, primary_key=True)
    _name = Column(String(255), nullable=False)
    _username = Column(String(255), nullable=False, unique=True)
    _email = Column(String(255), nullable=False, unique=True)
    _number = Column(String(255), nullable=True)
    _age = Column(Integer, nullable=False)
    _weight = Column(Integer, nullable=False)
    _height = Column(String(255), nullable=False)
    _allergies = Column(String(255), nullable=True)
    _conditions = Column(String(255), nullable=True)
    _ethnicity = Column(String(255), nullable=False)

    def __init__(self, name, username, email, number, age, weight, height, allergies, conditions, ethnicity):
        self._name = name
        self._username = username
        self._email = email
        self._number = number
        self._age = age
        self._weight = weight
        self._height = height
        self._allergies = allergies
        self._conditions = conditions
        self._ethnicity = ethnicity

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
            "name": self._name,
            "username": self._username,
            "email": self._email,
            "number": self._number,
            "age": self._age,
            "weight": self._weight,
            "height": self._height,
            "allergies": self._allergies,
            "conditions": self._conditions,
            "ethnicity": self._ethnicity
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
        Survey(name="John Doe", username="john_doe", email="john@example.com", number="1234567890", age=30, weight=180, height="5'10", allergies="Peanuts", conditions="None", ethnicity="Caucasian"),
        Survey(name="Jane Smith", username="jane_smith", email="jane@example.com", number="0987654321", age=25, weight=135, height="5'6", allergies="Dust", conditions="Asthma", ethnicity="Asian")
    ]

    for survey in surveys:
        try:
            survey.create()
        except IntegrityError:
            db.session.rollback()
            print(f"Error adding survey: {survey._name}")
