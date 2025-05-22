from __init__ import db
from model.user import User
from datetime import datetime

class HospitalRating(db.Model):
    __tablename__ = 'hospitalRatings'
    id = db.Column(db.Integer, primary_key=True)
    _title = db.Column(db.String(255), nullable=False)
    _description = db.Column(db.String(255), nullable=True)
    _uid = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    _hospital = db.Column(db.String(255), nullable=False)
    _rating = db.Column(db.Integer, nullable=False)
    _date_posted = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    def __init__(self, title, description, uid, hospital, rating, input_datetime=None):
        self._title = title
        self._description = description
        self._uid = uid
        self._hospital = hospital
        self._rating = rating
        self._date_posted = input_datetime or datetime.utcnow()

    def create(self):
        try:
            db.session.add(self)
            db.session.commit()
        except Exception as error:
            db.session.rollback()
            raise error

    def read(self):
        user = User.query.get(self._uid)
        return {
            "id": self.id,
            "title": self._title,
            "description": self._description,
            "user": {
                "name": user.read()["name"],
                "id": user.read()["id"],
                "uid": user.read()["uid"],
                "email": user.read()["email"],
                "pfp": user.read()["pfp"]
            },
            "hospital": self._hospital,
            "rating": self._rating,
            "date_posted": self._date_posted.isoformat()
        }