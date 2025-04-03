# model/data_post.py
import logging
from sqlite3 import IntegrityError
from sqlalchemy import Integer, String, JSON, DateTime
from sqlalchemy.exc import IntegrityError
from datetime import datetime
from __init__ import db

class DataPost(db.Model):
    """
    DataPost Model

    Represents a post in the system, including metadata like like count, view count, 
    video length, comment count, and creation date.

    Attributes:
        id (db.Column): The primary key.
        _title (db.Column): The title of the post.
        _comment (db.Column): The comment associated with the post.
        _content (db.Column): The JSON content of the post.
        _channel_id (db.Column): The channel ID to which the post belongs.
        _like_count (db.Column): Number of likes on the post.
        _view_count (db.Column): Number of views on the post.
        _video_length (db.Column): Length of video (if applicable).
        _comment_count (db.Column): Number of comments on the post.
        _date (db.Column): Date when the post was created.
    """
    __tablename__ = 'data_posts'

    id = db.Column(Integer, primary_key=True)
    _title = db.Column(String(255), nullable=False)
    _comment = db.Column(String(255), nullable=False)
    _content = db.Column(JSON, nullable=False, default={})
    _channel_id = db.Column(Integer, db.ForeignKey('channels.id'), nullable=False)
    _like_count = db.Column(Integer, default=0)
    _view_count = db.Column(Integer, default=0)
    _video_length = db.Column(Integer, default=0)
    _comment_count = db.Column(Integer, default=0)
    _date = db.Column(DateTime, default=datetime.utcnow)

    def __init__(self, title, comment, channel_id, content={}, like_count=0, view_count=0, video_length=0, comment_count=0, date=None):
        self._title = title
        self._comment = comment
        self._channel_id = channel_id
        self._content = content
        self._like_count = like_count
        self._view_count = view_count
        self._video_length = video_length
        self._comment_count = comment_count
        self._date = date or datetime.utcnow()

    def __repr__(self):
        return f"DataPost(id={self.id}, title={self._title}, comment={self._comment}, channel_id={self._channel_id}, like_count={self._like_count}, view_count={self._view_count}, video_length={self._video_length}, comment_count={self._comment_count}, date={self._date})"

    def create(self):
        try:
            db.session.add(self)
            db.session.commit()
        except IntegrityError as e:
            db.session.rollback()
            logging.warning(f"IntegrityError: Could not create post '{self._title}' due to {str(e)}.")
            return None
        return self

    def read(self):
        return {
            "id": self.id,
            "title": self._title,
            "comment": self._comment,
            "content": self._content,
            "channel_id": self._channel_id,
            "like_count": self._like_count,
            "view_count": self._view_count,
            "video_length": self._video_length,
            "comment_count": self._comment_count,
            "date": self._date.isoformat() if self._date else None
        }

    def update(self):
        try:
            db.session.commit()
        except IntegrityError:
            db.session.rollback()
            logging.warning(f"IntegrityError: Could not update post '{self._title}'.")
            return None
        return self

    def delete(self):
        try:
            db.session.delete(self)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            raise e

def initDataPosts():
    """
    Initializes the DataPost table with some sample data.
    """
    with db.app.app_context():
        db.create_all()
        sample_posts = [
            DataPost(title="New Feature Release", comment="We're excited to introduce a new feature!", channel_id=1, content={"type": "announcement"}, like_count=10, view_count=100),
            DataPost(title="Bug Fixes and Updates", comment="Various bug fixes and performance improvements.", channel_id=2, content={"type": "update"}, like_count=5, view_count=50),
        ]
        for post in sample_posts:
            try:
                post.create()
                print(f"Created: {repr(post)}")
            except IntegrityError:
                db.session.rollback()
                print(f"Error: Post '{post._title}' might already exist.")

