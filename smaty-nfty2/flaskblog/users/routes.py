from flask import Blueprint 
import os 
import secrets
from PIL import Image
from flask import render_template, url_for, flash, redirect, request, session, logging
from flask_babelex import Babel
from flaskblog import app, db, bcrypt
from flaskblog.users.forms import RegistrationForm, LoginForm, UpdateAccountForm, RequestResetForm, ResetPasswordForm
from flaskblog.models import User, Post , Role, UserRoles
from flask_login import login_user , current_user, logout_user, login_required
from flask_mail import Message
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
from passlib.hash import sha256_crypt
#engine = create_engine("mysql+pymysql://root:@localhost/facebook")#
#db = scoped_session(sessionmaker(bind = engine))
from flaskblog.users.utils import save_picture, send_reset_email
# importing libraries related to mysql connection
from flaskext.mysql import MySQL
# Database connection info. Note that this is not a secure connection.
app.config['MYSQL_DATABASE_USER'] = 'root'
app.config['MYSQL_DATABASE_PASSWORD'] = ''
app.config['MYSQL_DATABASE_DB'] = 'new_db'
app.config['MYSQL_DATABASE_HOST'] = 'localhost'

admin_role = Role(name='Admin')
client_role = Role(name='Client')

mysql = MySQL()
mysql.init_app(app)

conn = mysql.connect()
cursor = conn.cursor()

#user_manager = UserManager(app, db, User)
# usermanager 
#user_manager = UserManager(app, db, User, UserEmailClass='selmane_si@hotmail.com')

# IMPORT PACKAGES FOR USE 
import pandas as pd 
import numpy as np
import io
# END OF IMPORT PACKAGES
users = Blueprint('users', __name__)

def redirect_not_logged_in():
    if current_user.is_authenticated:
        flash('You should be logged in to acced this page', 'info')
        return redirect(url_for('main.dashboard'))
    else:
        flash('You should be logged in to acced this page', 'info')
        return redirect(url_for('main.home'))


@users.route('/register', methods=['POST', 'GET'])
def register():
        #raise ValidationError('That username was taken, please choose another one')
    if current_user.is_authenticated:
        user_k = ""
        user = User.query.filter_by(username=current_user.username).first()
        userrole = UserRoles.query.filter_by(user_id=user.id).first()
        role = Role.query.filter_by(id=userrole.role_id).first()
        if role.name != "Admin":
            flash("You are not allowed to add new users", 'info')
            return redirect(url_for('main.dashboard'))    #    return redirect(url_for('main.home'))
        elif role.name == 'Admin':
            form = RegistrationForm()
            if form.validate_on_submit():
                hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
                user = User(username=form.username.data, email=form.email.data, password = hashed_password)
                if form.roles.data == 'admin':
                    user.roles = [admin_role,]
                elif form.roles.data == "client":
                    user.roles = [client_role,]
                db.session.add(user)
                db.session.commit()
                flash('Your account has been created, you can now login', 'success')
                return redirect(url_for('users.login'))
            return render_template('register.html', title = 'Registration' , 
                form = form)
    else:
        flash('You should contact the manager to have access to this app', 'info')
        return redirect(url_for('users.login'))

    #if form.validate_on_submit():
    #    hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
    #    user = User(username=form.username.data, email=form.email.data, password = hashed_password )
    #    db.session.add(user)
    #    db.session.commit()
    #    flash('Your account has been created, you can now login', 'success')
    #    return redirect(url_for('main.home'))

@users.route('/login', methods=['POST', 'GET'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('analyse.train_database'))
    form = LoginForm()
    my_next = request.endpoint
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user, remember = form.remember.data)
            session.permanent = True
            flash('Login success, you can make some tests', 'success')
            next_page = request.args.get('next')
            return redirect(url_for(next_page)) if next_page else redirect(url_for('main.dashboard'))
        else :
            flash('Login Unsuccessful. Please check username and password', 'danger')
    return render_template('login.html', title = 'Login', form = form, 
                            my_next = my_next)




@users.route('/account', methods=['POST', 'GET'])
@login_required
def account():
    form = UpdateAccountForm()
    if form.validate_on_submit():
        if form.picture.data:
            picture_file = save_picture(form.picture.data)
            current_user.image_file = picture_file
        current_user.competitors = form.competitors.data
        current_user.keywords = form.keywords.data
        current_user.email = form.email.data
        db.session.commit()
        flash('Your account has been updated!', 'success')
        return redirect(url_for('users.account'))
    elif request.method == 'GET':
        form.username.data = current_user.username
        form.email.data = current_user.email
    image_file = url_for('static', filename='profile_pics/' + current_user.image_file)
    form.keywords.data = current_user.keywords
    form.competitors.data = current_user.competitors
    return render_template('account.html', 
        title = 'Account', 
        image_file = image_file, 
        form = form)





@users.route('/logout')
def logout():
    logout_user()
    flash('you are now logged out', 'success')
    return redirect(url_for('main.home'))







@users.route( "/user/<string:username>" )
def user_posts(username):
    page = request.args.get('page', 1, type = int)
    user = User.query.filter_by(username = username).first_or_404()
    posts = Post.query.filter_by(author=user)\
        .order_by(Post.date_posted.desc())\
        .paginate(page = page, per_page = 5)
    return render_template('user_posts.html', posts = posts, user = user)




@users.route("/reset_request", methods = ['GET', 'POST'])
def reset_request():
    if current_user.is_authenticated:
        return redirect(url_for('main.home'))
    form = RequestResetForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email = form.email.data).first()
        send_reset_email(user)
        flash('An email has been sent with instructions to reset your password', 'info')
        return redirect(url_for('main.home'))
    return render_template('reset_request.html', title = 'Reset Password', 
        form = form)


@users.route("/reset_password/<token>", methods = ['GET', 'POST'])
def reset_token(token):
    if current_user.is_authenticated:
        return redirect(url_for('main.home'))
    user = User.verify_reset_token(token)
    if user is None:
        flash('That is an invalid or expired token', 'warning')
        return redirect(url_for('users.reset_request'))
    form = ResetPasswordForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user.password = hashed_password
        db.session.commit()
        flash('Your Password has been updated!, you can now login', 'success')
        return redirect(url_for('users.login'))
    return render_template('reset_password.html', title = 'Reset Password', form = form)



