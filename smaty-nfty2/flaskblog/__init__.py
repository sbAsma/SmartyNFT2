import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy 
from flask_bcrypt import Bcrypt 
from flask_login import LoginManager 
from flask_mail import Mail


app = Flask(__name__)
app.config['SECRET_KEY'] = 'This is a secret key for selmane ! '
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost/new_db'

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'users.login'
login_manager.login_message_category = 'info'

app.config['MAIL_SERVER'] = 'smtp.googlemail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.environ.get('EMAIL_USER')
app.config['MAIL_PASSWORD'] = os.environ.get('EMAIL_PASS')

 # Flask-Mail SMTP server settings
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USE_SSL'] = True
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USERNAME'] = 'selmane@example.com'
app.config['MAIL_PASSWORD'] = 'password'
app.config['MAIL_DEFAULT_SENDER'] = '"MyApp" <noreply@example.com>'

# Flask-User settings
app.config['USER_APP_NAME'] = "Sentiment Analysis App"      # Shown in and email templates and page footers
app.config['USER_ENABLE_EMAIL'] = True        # Enable email authentication
app.config['USER_ENABLE_USERNAME'] = False    # Disable username authentication
app.config['USER_EMAIL_SENDER_NAME'] = 'Selmane'
app.config['USER_EMAIL_SENDER_EMAIL'] = "selmane_si@hotmail.com"


mail = Mail(app)
db.create_all() # In case user table doesn't exists already. Else remove it.    

from flaskblog.main.routes import main 
from flaskblog.users.routes import users 
from flaskblog.posts.routes import posts
from flaskblog.analyse.routes import analyse
from flaskblog.errors.handlers import errors

app.register_blueprint(main) 
app.register_blueprint(users) 
app.register_blueprint(posts) 
app.register_blueprint(analyse) 
app.register_blueprint(errors) 



