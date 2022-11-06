import os 
import secrets
from PIL import Image
from flask import render_template, url_for, flash, redirect, request, session, abort, send_file
from flaskblog import app, db, bcrypt, mail
from flaskblog.forms import RegistrationForm, LoginForm, UpdateAccountForm, PostForm, RequestResetForm, ResetPasswordForm, SaveModel
from flaskblog.models import User, Post 
from flask_login import login_user , current_user, logout_user , login_required, login_manager
#import matplotlib.pyplot as plt
from flask_mail import Message

# importing libraries related to mysql connection
from flaskext.mysql import MySQL
# Database connection info. Note that this is not a secure connection.
#app.config['MYSQL_DATABASE_USER'] = 'mycomp34_wp413'
#app.config['MYSQL_DATABASE_PASSWORD'] = 'p])8Yu43S5'
#app.config['MYSQL_DATABASE_DB'] = 'mycomp34_wp413'
#app.config['MYSQL_DATABASE_HOST'] = 'localhost'

app.config['MYSQL_DATABASE_USER'] = 'root'
app.config['MYSQL_DATABASE_PASSWORD'] = ''
app.config['MYSQL_DATABASE_DB'] = 'new_db'
app.config['MYSQL_DATABASE_HOST'] = 'localhost'

mysql = MySQL()
mysql.init_app(app)

conn = mysql.connect()
cursor = conn.cursor()



# IMPORT PACKAGES FOR USE 
from datetime import datetime
import pandas as pd 
pd.set_option('display.max_colwidth', -1)
import pickle 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib 

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import io


# END OF IMPORT PACKAGES

























































APP_ROOT = os.path.dirname(os.path.abspath(__file__))








































































































