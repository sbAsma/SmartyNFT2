from flask import Blueprint 
import os 
import secrets
from PIL import Image
from flask import render_template, url_for, flash, redirect, request, jsonify
from flaskblog import app, db, bcrypt, mail
from flaskblog.models import User, Post 
from flask_login import  current_user , login_required
from flask_mail import Message

# importing libraries related to mysql connection
from flaskext.mysql import MySQL
# Database connection info. Note that this is not a secure connection.

app.config['MYSQL_DATABASE_USER'] = 'root'
app.config['MYSQL_DATABASE_PASSWORD'] = ''
app.config['MYSQL_DATABASE_DB'] = 'books_db'
app.config['MYSQL_DATABASE_HOST'] = 'localhost'

mysql = MySQL() 
mysql.init_app(app)

conn = mysql.connect()
cursor = conn.cursor()

# IMPORT PACKAGES FOR USE 
import pandas as pd 
pd.set_option('display.max_colwidth', -1)
import pickle 
import numpy as np
import io


# END OF IMPORT PACKAGES

APP_ROOT = os.path.dirname(os.path.abspath(__file__))



main = Blueprint('main', __name__)





@main.route('/test_ajax')
def test_ajax():
    return render_template('test_ajax.html', title = 'test ajax')








@main.route("/")
@main.route('/home')
def home():
    test_path = os.path.join(APP_ROOT, 'models')
    dirss = os.listdir( test_path )
    folders = []
    dir1 = []
    dir2 = []
    texts = ' s '

    for dirs in dirss:
        if os.path.isdir(test_path + '\\' + dirs):
            folders.append(dirs)
        dir1 = APP_ROOT + '\\' + dirs
        dir2.append(dir1)

   

            

    title = "This is home page"


    labels = [
        'Jan', 'Feb', 'Mar', 'Apr', 
        'May', 'Jun', 'Jul', 'Aug',
        'Sep', 'Oct', 'Nov', 'Dec', 
    ]

    values = [
        967.67, 1190.89, 1079.75, 1349.19,
        2328.91, 2504.28, 2873.83, 4764.87,
        4349.29, 6458.30, 9907, 16297
    ]

    colors  = [
        "#F7464A", "#46BFBD", "#FDB45C", "#FEDCBA",
        "#ABCDEF", "#DDDDDD", "#ABCABC", "#4169E1",
        "#C71585", "#FF4500", "#FEDCBA", "#46BFBD"
    ]
    page = request.args.get('page', 1, type=int)
    posts = Post.query.order_by(Post.date_posted.desc()).paginate(page = page, per_page=5)
    return render_template('index.html', posts = posts, title = title, 
        max = 17000, labels = labels, values = values, 
        folders = folders,  
        dirss = dirss, 
        dir2 = dir2,
        test_path = test_path, 
        texts = texts)





@main.route('/dashboard')
def dashboard():
    if not current_user.is_authenticated:
        flash('You should be logged in', 'info')
        next_page = request.endpoint
        return redirect(url_for("users.login", next = next_page))

    title = "Dashboard Page"
    return render_template('dashboard.html', title =title)






@main.route('/about')
def about():
    return render_template('about.html')








@main.route('/_get_data/', methods=['POST'])
def _get_data():
    myList = ['Element 1', 'Element 2', 'Elemenet 3']

    return jsonify({'data': render_template('response.html', myList = myList)})





 