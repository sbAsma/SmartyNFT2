from flask import Blueprint

import os 
import secrets
from PIL import Image
from flask import render_template, jsonify, url_for, flash, redirect, request, session, abort
from flaskblog import app, db, bcrypt, mail
from flaskblog.posts.forms import PostForm, WordsForm, CommentsForm
from flaskblog.models import User, Post, Words
from flask_login import  current_user, login_required
from flask_mail import Message
from flask_restful import Resource, Api

# importing libraries related to mysql connection
from flaskext.mysql import MySQL
# Database connection info. Note that this is not a secure connection.
app.config['MYSQL_DATABASE_USER'] = 'root'
app.config['MYSQL_DATABASE_PASSWORD'] = ''
app.config['MYSQL_DATABASE_DB'] = 'facebook'
app.config['MYSQL_DATABASE_HOST'] = 'localhost'

 

mysql = MySQL()
mysql.init_app(app)
api = Api(app)

conn = mysql.connect()
cursor = conn.cursor()



# IMPORT PACKAGES FOR USE 
import pandas as pd 
pd.set_option('display.max_colwidth', -1)
import numpy as np
import io


# END OF IMPORT PACKAGES



posts = Blueprint('posts', __name__)




@posts.route("/post/new", methods = ['GET', 'POST'])
@login_required
def new_post():
    form = PostForm()
    if form.validate_on_submit():
        post = Post(title = form.title.data, content = form.content.data, author = current_user)
        db.session.add(post)
        db.session.commit()
        flash('Your Post has been created!', 'success')
        return redirect(url_for('main.home'))
    return render_template('create_post.html', title = "New Post", 
        form = form, 
        legend = "Create Post")


@posts.route("/post/<int:post_id>", methods=['GET', 'POST'])
def post(post_id):
    post = Post.query.get_or_404(post_id)
    return render_template('post.html', title=post.title, post = post)


@posts.route("/post/<int:post_id>/update", methods = ['GET', 'POST'])
@login_required
def update_post(post_id):
    post = Post.query.get_or_404(post_id)
    if post.author != current_user:
        abort(403)
    form = PostForm()
    if form.validate_on_submit():
        post.title = form.title.data
        post.content = form.content.data 
        db.session.commit()
        flash('Your post has been updated!', 'success')
        return redirect(url_for('posts.post', post_id=post.id))
    elif request.method == 'GET':
        form.title.data = post.title 
        form.content.data = post.content 
    return render_template('update_post.html', 
        title = "Update Post", form = form, 
        post = post,
        legend = "Update Post")


@posts.route("/post/<int:post_id>/delete", methods = ['POST'])
@login_required
def delete_post(post_id):
    post = Post.query.get_or_404(post_id)
    if post.author != current_user:
        abort(403)
    db.session.delete(post)
    db.session.commit()
    flash('Your Post has been deleted!', 'success')
    return redirect(url_for('main.home'))













@posts.route("/variation/new", methods = ['GET', 'POST'])
@login_required
def new_variation():
    form = WordsForm()
    if form.validate_on_submit():
        word = request.form['word']
        latin = request.form['word_latin']
        variations = request.form['variations']
        word_check = cursor.execute("""SELECT ID AS word_id 
            FROM words 
            WHERE word = %s""", (word))
        if not word_check:
            cursor.execute("INSERT INTO words (word, word_latin, variations) VALUES (%s, %s, %s)", (word, latin, variations))
            conn.commit()
            flash('Your Variation has been created!', 'success')
        else :
            cursor.execute("""UPDATE words SET variations=CONCAT(variations, ', ' %s)
                WHERE word = %s
                """, (variations, word))
            conn.commit()
            flash('Your Variation has been updated!', 'success')
        return redirect(url_for('posts.new_variation'))
    return render_template('create_variation.html', title = "New Post", 
        form = form, 
        legend = "Create Post")








@posts.route("/all_variations", methods = ['GET', 'POST'])
@login_required
def all_variations():
    cursor.execute("SELECT * FROM words")
    all_vars = cursor.fetchall()
    to_df = pd.DataFrame(list(all_vars), columns=['index', 'word', 'latin', 'variation'])
    #conn.commit()
    return render_template('all_variations.html', title = "All Variations", 
        to_df = to_df, 
        legend = "Create Post")












@posts.route("/classify", methods = ['GET', 'POST'])
@login_required
def classify():
    import mysql.connector
    mydb = mysql.connector.connect(
      host="localhost",
      user="root",
      passwd="",
      db = "facebook"
    )
    cursor = mydb.cursor()
    cursor.execute("SELECT idCommentaire, commentaire, tonalite FROM commentaires LIMIT 1")
    data = cursor.fetchall()
    to_df = pd.DataFrame(list(data), columns=['idCommentaire', 'commentaire', 'tonalite'])
    len_df = len(to_df)
    form = WordsForm()
    if form.validate_on_submit():
        latin = request.form['word_latin']
        variations = request.form['comment_id']
        flash('Your Variation has been updated!', 'success')
        return redirect(url_for('posts.classify'))
    return render_template('classify.html', title = "Classify Comments", 
        form = form, to_df = to_df,
        legend = "Classify Comments", 
        len_df = len_df)








@posts.route("/test_api", methods = ['GET', 'POST'])
def test_api():
    cursor.execute("SELECT * FROM words")
    all_vars = cursor.fetchall()
    to_df = pd.DataFrame(list(all_vars), columns=['index', 'word', 'latin', 'variation'])

    #conn.commit()
    return jsonify(json_list = all_vars), 201




class HelloWorld(Resource):
    def get(self):
        return {'about': 'Hello World!'}

    def post(self):
        some_json = request.get_json()
        return {"you sent": some_json}, 201

class Multi(Resource):
    def get(self, num):
        return {'result': num*10}

class SentApi(Resource):
    """docstring for ClassName"""
    def get(self):
        cursor.execute("SELECT * FROM words")
        all_vars = cursor.fetchall()
        to_df = pd.DataFrame(list(all_vars), columns=['index', 'word', 'latin', 'variation'])
        return all_vars, 201


api.add_resource(HelloWorld, '/h')
api.add_resource(Multi, '/multi/<int:num>')
api.add_resource(SentApi, '/sentapi/')

