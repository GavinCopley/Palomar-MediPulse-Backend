# imports from flask
import json
import os
import ast
from urllib.parse import urljoin, urlparse
from flask import abort, redirect, render_template, request, send_from_directory, url_for, jsonify, flash  # import render_template from "public" flask libraries
from flask_login import current_user, login_user, logout_user
from flask.cli import AppGroup
from flask_login import current_user, login_required
from flask import current_app
from flask import send_from_directory
from werkzeug.security import generate_password_hash
import shutil
import datetime



# import "objects" from "this" project
from __init__ import app, db, login_manager  # Key Flask objects 
# API endpoints
from api.user import user_api 
from api.hospital_api import bp as hospital_bp 
from api.pfp import pfp_api
from api.nestImg import nestImg_api # Justin added this, custom format for his website
from api.post import post_api
from api.channel import channel_api
from api.group import group_api
from api.video import bp as video_opt_bp             
from api.section import section_api
from api.nestPost import nestPost_api # Justin added this, custom format for his website
from api.survey import survey_api
from api.hospitalsearch import hospital_search_api 

#from api.carChat import carChat_api
from api.carPost import carPost_api
from api.chatBot import chatbot_api
from api.carComments import carComments_api
from api.vinStore import vinStore_api
from api.hospitalDataAnalytics import analytics_api
from api.videoStoreAI import videoStore_api
from api.comparisonData import comparison_api
#from data.youtubeAnalytics import youtube_api

from api.vote import vote_api
# database Initialization functions
#from model.carChat import carChat
from model.user import User, initUsers
from model.section import Section, initSections
from model.group import Group, initGroups
from model.channel import Channel, initChannels
from model.hospital_recommender import HospitalRecommender
from model.post import Post, initPosts
from model.nestPost import NestPost, initNestPosts # Justin added this, custom format for his website
from model.vote import Vote, initVotes
from model.carPost import CarPost
# api/video.py  (very top)
from model.optimize import VideoOptimiser       
from model.carComments import CarComments
from model.vehicle import Vehicle, initVehicles
from model.survey import Survey, initSurvey 

# server only Views

# register URIs for api endpoints
app.register_blueprint(user_api)
app.register_blueprint(pfp_api) 
app.register_blueprint(post_api)
app.register_blueprint(channel_api)
app.register_blueprint(video_opt_bp)
app.register_blueprint(group_api)
app.register_blueprint(section_api)
#app.register_blueprint(carChat_api)
# Added new files to create nestPosts, uses a different format than Mortensen and didn't want to touch his junk
app.register_blueprint(nestPost_api)
app.register_blueprint(nestImg_api)
app.register_blueprint(vote_api)
app.register_blueprint(carPost_api)
app.register_blueprint(chatbot_api)
app.register_blueprint(hospital_bp) 
app.register_blueprint(carComments_api)
app.register_blueprint(vinStore_api)
app.register_blueprint(analytics_api)
app.register_blueprint(survey_api)
app.register_blueprint(hospital_search_api)
app.register_blueprint(videoStore_api)
app.register_blueprint(comparison_api)
#app.register_blueprint(youtube_api)
from api.stats import stats_api
app.register_blueprint(stats_api)

# Add this to your imports section

@app.route('/carPosts')
@login_required  # Ensure that only logged-in users can access this page
def carPosts():
    carPost_data = CarPost.query.all()  # Fetch all car posts from the database
    print("Car Post Data:", carPost_data)  # Debugging line to check if data is fetched
    return render_template("carPosts.html", carPost_data=carPost_data)



@app.route('/carComments')
@login_required  # Ensure that only logged-in users can access this page
def carCommentsPage():
    carComments_data = CarComments.query.all()  # Fetch all car comments from the database
    print("Car Comments Data:", carComments_data)  # Debugging line to check if data is fetched
    return render_template("carComments.html", carComments_data=carComments_data)


    
    if message is None:
        return jsonify({"error": "Message not found"}), 404  # Return 404 if message doesn't exist
    
    # Update the message content
    message._message = data.get('message', message._message)  # Update with new message content
    message.update()  # Call the update method from the model
    
    return jsonify(message.read()), 200  # Return the updated message data



@app.route('/api/carPost/allPosts/<string:car_type>', methods=['GET'])
def allPosts(car_type):
    if car_type not in ['gas', 'electric', 'hybrid', 'dream']:
        return jsonify({'message': 'Car type must be one of gas, electric, hybrid, dream'}), 400
    posts = CarPost.query.filter(CarPost._car_type == car_type).all()
    return jsonify([post.read() for post in posts])

@app.route('/api/carPost/postsByUser/<int:user_id>', methods=['GET'])
def postsByUser(user_id):
    user = User.query.get(user_id)
    if user is None:
        return jsonify({'message': 'User not found'}), 404
    posts = CarPost.query.filter(CarPost._uid == user_id).all()
    return jsonify([post.read() for post in posts])

from model.carPostImage import carPostImage_base64_decode, carPostImage_base64_upload

#@app.route('/vehicles')
#@login_required  # Ensure that only logged-in users can access this page
#def vehiclesPage():
    #vehicles_data = Vehicle.query.all()  # Fetch all vehicles from the database
    #print("Vehicles Data:", vehicles_data)  # Debugging line to check if data is fetched
    #return render_template("vehicles.html", vehicles_data=vehicles_data)

@app.route('/api/carPost/<int:post_id>/images', methods=['GET'])
def getPostImages(post_id):
    post = CarPost.query.get(post_id)
    if post is None:
        return jsonify({'message': 'Post not found'}), 404
    image_url_table = post._image_url_table
    if not image_url_table or len(image_url_table) == 0:
        return jsonify({'message': 'There are no images for this post.'}), 404
    
    images = []
    for url in ast.literal_eval(image_url_table):
        print(url)
        image = carPostImage_base64_decode(post_id, url)
        images.append(image)
        
    return jsonify(images)

@app.route('/api/carComment/<int:post_id>', methods=['GET'])
def getPostComments(post_id):
    post = CarPost.query.get(post_id)
    if post is None:
        return jsonify({'message': 'Post not found'}), 404
    comments = CarComments.query.filter(CarComments._post_id == post_id).all()
    return jsonify([comment.read() for comment in comments])

@app.route('/api/data/mort', methods=['GET'])
def get_data():
    # start a list, to be used like a information database
    InfoDb = []

    # add a row to list, an Info record
    InfoDb.append({
        "FirstName": "John",
        "LastName": "Mortensen",
        "DOB": "October 21",
        "Residence": "San Diego",
        "Email": "jmortensen@powayusd.com",
        "Owns_Cars": ["2015-Fusion", "2011-Ranger", "2003-Excursion", "1997-F350", "1969-Cadillac"]
    })

    # add a row to list, an Info record
    InfoDb.append({
        "FirstName": "Shane",
        "LastName": "Lopez",
        "DOB": "February 27",
        "Residence": "San Diego",
        "Email": "slopez@powayusd.com",
        "Owns_Cars": ["2021-Insight"]
    })
    
    return jsonify(InfoDb)


@app.route('/data/hospitaldatamodified.csv')
def serve_hospital_csv():
    csv_dir = os.path.join(current_app.root_path, 'data')
    return send_from_directory(csv_dir, 'hospitaldatamodified.csv')

# Tell Flask-Login the view function name of your login route
login_manager.login_view = "login"

@login_manager.unauthorized_handler
def unauthorized_callback():
    return redirect(url_for('login', next=request.path))

# register URIs for server pages
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.context_processor
def inject_user():
    return dict(current_user=current_user)

# Helper function to check if the URL is safe for redirects
def is_safe_url(target):
    ref_url = urlparse(request.host_url)
    test_url = urlparse(urljoin(request.host_url, target))
    return test_url.scheme in ('http', 'https') and ref_url.netloc == test_url.netloc

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    next_page = request.args.get('next', '') or request.form.get('next', '')
    if request.method == 'POST':
        user = User.query.filter_by(_uid=request.form['username']).first()
        if user and user.is_password(request.form['password']):
            login_user(user)
            if not is_safe_url(next_page):
                return abort(400)
            return redirect(next_page or url_for('index'))
        else:
            error = 'Invalid username or password.'
    return render_template("login.html", error=error, next=next_page)
    
@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.errorhandler(404)  # catch for URL not found
def page_not_found(e):
    # note that we set the 404 status explicitly
    return render_template('404.html'), 404

@app.route('/')  # connects default URL to index() function
def index():
    print("Home:", current_user)
    return render_template("index.html")

@app.route('/users/table')
@login_required
def utable():
    users = User.query.all()
    return render_template("utable.html", user_data=users)

@app.route('/users/table2')
@login_required
def u2table():
    users = User.query.all()
    return render_template("u2table.html", user_data=users)

@app.route('/survey/table')
@login_required
def surveytable():
    surveys = Survey.query.all()
    print(f"Loaded {len(surveys)} surveys")  # Debug line
    return render_template("surveytable.html", survey_data=surveys)


@app.route('/survey/<int:id>/edit', methods=['GET', 'POST'])
@login_required
def edit_survey(id):
    survey = Survey.query.get_or_404(id)
    if request.method == 'POST':
        try:
            # Get form data with proper type conversion and validation
            name = request.form.get('name')
            username = request.form.get('username')
            email = request.form.get('email')
            number = request.form.get('number')
            
            # Handle age and weight with proper validation
            try:
                age = int(request.form.get('age', 0))
                weight = int(request.form.get('weight', 0))
                height = int(request.form.get('height', 0))
            except (ValueError, TypeError):
                flash('Age, weight, and height must be valid numbers', 'error')
                return redirect(url_for('edit_survey', id=id))
            
            allergies = request.form.get('allergies')
            conditions = request.form.get('conditions')
            ethnicity = request.form.get('ethnicity')
            
            # Validate required fields
            if not all([name, username, email, ethnicity]):
                flash('Please fill in all required fields', 'error')
                return redirect(url_for('edit_survey', id=id))
            
            # Update survey fields using the correct attribute names
            survey.name = name
            survey.username = username
            survey.email = email
            survey.number = number
            survey.age = age
            survey.weight = weight
            survey.height = height
            survey.allergies = allergies
            survey.conditions = conditions
            survey.ethnicity = ethnicity
            
            db.session.commit()
            flash('Survey updated successfully!', 'success')
            return redirect(url_for('surveytable'))
            
        except Exception as e:
            db.session.rollback()
            flash(f'Error updating survey: {str(e)}', 'error')
            return redirect(url_for('edit_survey', id=id))
    
    return render_template('edit_survey.html', survey=survey)

@app.route('/survey/<int:id>/delete', methods=['POST'])
@login_required
def delete_survey(id):
    survey = Survey.query.get_or_404(id)
    try:
        db.session.delete(survey)
        db.session.commit()
        flash('Survey deleted successfully!', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error deleting survey: {str(e)}', 'error')
    return redirect(url_for('surveytable'))


# Helper function to extract uploads for a user (ie PFP image)
@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(current_app.config['UPLOAD_FOLDER'], filename)
 
@app.route('/users/delete/<int:user_id>', methods=['DELETE'])
@login_required
def delete_user(user_id):
    user = User.query.get(user_id)
    if user:
        user.delete()
        return jsonify({'message': 'User deleted successfully'}), 200
    return jsonify({'error': 'User not found'}), 404

@app.route('/users/reset_password/<int:user_id>', methods=['POST'])
@login_required
def reset_password(user_id):
    if current_user.role != 'Admin':
        return jsonify({'error': 'Unauthorized'}), 403
    
    user = User.query.get(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404

    # Set the new password
    if user.update({"password": app.config['DEFAULT_PASSWORD']}):
        return jsonify({'message': 'Password reset successfully'}), 200
    return jsonify({'error': 'Password reset failed'}), 500

# Create an AppGroup for custom commands
custom_cli = AppGroup('custom', help='Custom commands')

# Define a command to run the data generation functions
@custom_cli.command('generate_data')
def generate_data_cli():
    """CLI command version of generate_data"""
    generate_data()

# Regular function version that can be imported and called from other scripts
def generate_data():
    """Generate initial data for the database"""
    # Ensure we're in an application context
    if not current_app:
        raise RuntimeError("This function must be called within a Flask application context")
    
    print("Initializing users...")
    initUsers()
    
    print("Initializing sections...")
    initSections()
    
    print("Initializing vehicles...")
    initVehicles()
    
    print("Initializing surveys...")
    initSurvey()
    
    print("Data generation completed!")

# Backup the old database
def backup_database(db_uri, backup_uri):
    """Backup the current database."""
    if backup_uri:
        db_path = db_uri.replace('sqlite:///', 'instance/')
        backup_path = backup_uri.replace('sqlite:///', 'instance/')
        shutil.copyfile(db_path, backup_path)
        print(f"Database backed up to {backup_path}")
    else:
        print("Backup not supported for production database.")

# Extract data from the existing database
def extract_data():
    data = {}
    with app.app_context():
        data['users'] = [user.read() for user in User.query.all()]
        data['sections'] = [section.read() for section in Section.query.all()]
        data['groups'] = [group.read() for group in Group.query.all()]
        data['channels'] = [channel.read() for channel in Channel.query.all()]
        data['posts'] = [post.read() for post in Post.query.all()]
        data['carPosts'] = [post.read() for post in CarPost.query.all()]
        data['vehicles'] = [vehicle.read() for vehicle in Vehicle.query.all()]
        data['carComments'] = [comment.read() for comment in CarComments.query.all()]
    return data

# Save extracted data to JSON files
def save_data_to_json(data, directory='backup'):
    if not os.path.exists(directory):
        os.makedirs(directory)
    for table, records in data.items():
        with open(os.path.join(directory, f'{table}.json'), 'w') as f:
            for record in records:
                if record.get('date_posted'):
                    record['date_posted'] = record['date_posted'].isoformat()
                if record.get('date_added'):
                    if type(record['date_added']) != str:
                        record['date_added'] = record['date_added'].isoformat()
            json.dump(records, f)
    print(f"Data backed up to {directory} directory.")

# Load data from JSON files
def load_data_from_json(directory='backup'):
    data = {}
    for table in ['users', 'sections', 'groups', 'channels', 'posts', 'hosps', 'carPosts', 'carComments']:
        with open(os.path.join(directory, f'{table}.json'), 'r') as f:
            data[table] = json.load(f)
    return data

# Restore data to the new database
def restore_data(data):
    with app.app_context():
        users = User.restore(data['users'])
        _ = Section.restore(data['sections'])
        _ = Group.restore(data['groups'], users)
        _ = Channel.restore(data['channels'])
        _ = Post.restore(data['posts'])
        _ = CarPost.restore(data['carPosts'])
        _ = CarComments.restore(data['carComments'])
        _ = Vehicle.restore(data['vehicles'])
        _ = Survey.restore(data ['surveys'])
    print("Data restored to the new database.")

# Define a command to backup data
@custom_cli.command('backup_data')
def backup_data():
    data = extract_data()
    save_data_to_json(data)
    backup_database(app.config['SQLALCHEMY_DATABASE_URI'], app.config['SQLALCHEMY_BACKUP_URI'])

# Define a command to restore data
@custom_cli.command('restore_data')
def restore_data_command():
    data = load_data_from_json()
    restore_data(data)
    
# Register the custom command group with the Flask application
app.cli.add_command(custom_cli)
        
# this runs the flask application on the development server
if __name__ == "__main__":
    if os.environ.get('FLASK_ENV') == 'production':
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///instance/production.db'
        app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    else:
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///instance/development.db'
        app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
    # change name for testing
    app.config['TIMEZONE'] = 'America/Los_Angeles'
    app.run(host="0.0.0.0", port="8115")


    #@app.route('/api/mechanicsTips', methods=['GET'])
   # def get_mechanic_tip():
     #   make = request.args.get('make')
      #  model = request.args.get('model')
       # year = request.args.get('year')
        #issue = request.args.get('issue')

#        if not make or not model or not year or not issue:
 #           return jsonify({'message': 'Missing required parameters'}), 400

#        tip = MechanicsTip.query.filter_by(_make=make, _model=model, _year=year, _issue=issue).first()

 #       if tip:
  #          return jsonify(tip.read())
   #     else:
    #        return jsonify({'message': 'Tip not found'}), 404



# Example: Titanic Imports and Routes
#from model.titanic import initTitanic, testTitanic
#from api.titanic import titanic_api

# Register Titanic API blueprint
#app.register_blueprint(titanic_api)

# Initialize Titanic model before first request
#@app.before_first_request
#def before_first_request_titanic():
 #   initTitanic()

# You can test Titanic code right here if needed
#testTitanic()
