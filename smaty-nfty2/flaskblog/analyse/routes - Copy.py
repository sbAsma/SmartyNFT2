from flask import Blueprint 

import os 
import re
import secrets
from PIL import Image
from flask import render_template, url_for, flash, redirect, request, send_file, g, jsonify
from flaskblog import app, db, bcrypt, mail
from flaskblog.analyse.forms import SaveModel
from flaskblog.models import User, Post 
from flask_login import  current_user, login_required
import matplotlib.pyplot as plt
from flask_mail import Message
from datetime import datetime
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import numpy as np 
import pandas as pd
import validators
stop = stopwords.words('english')

stop_dareja = ['ana', 'homa', 'w', 'sel', 'jit', 'lebareh', 'ﻭ', 
                'fi', 'اﺫﺎﻤﻟ' ,'ﻰﻠﻋ', 'la', 'la', 'a', 'w', 'li', 'ce', 'vous',
                'à', 'et', 'des' ]




# importing libraries related to mysql connection
from flaskext.mysql import MySQL
# Database connection info. Note that this is not a secure connection.



app.config['MYSQL_DATABASE_USER'] = 'root'
app.config['MYSQL_DATABASE_PASSWORD'] = ''
app.config['MYSQL_DATABASE_DB'] = 'new_db'
app.config['MYSQL_DATABASE_HOST'] = 'localhost'


mysql = MySQL()
mysql.init_app(app)

conn = mysql.connect()
cursor = conn.cursor()


# IMPORT PACKAGES FOR USE 
import pandas as pd 
pd.set_option('display.max_colwidth', -1)
import pickle 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib 


from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import io





# Load libraries related to ML 
from nltk.tokenize import sent_tokenize, word_tokenize
from textblob import TextBlob
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
# Feature extraction
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from time import time

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
stop = stopwords.words('english')

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from time import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


# Comparing with other classifiers
# Now , we compare using different algorithms:
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.neighbors import NearestCentroid
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

Logistic_Regression = LogisticRegression(solver='newton-cg', multi_class='multinomial')
Linear_SVC = LinearSVC()
LinearSVC_L1 = Pipeline([
      ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False))),
      ('classification', LinearSVC(penalty="l2"))])    
Multinomial_NB = MultinomialNB()     
Bernoulli_NB = BernoulliNB()     
Ridge_Classifier = RidgeClassifier()     
AdaBoost = AdaBoostClassifier()      
Perceptron = Perceptron()  
Passive_Aggresive = PassiveAggressiveClassifier()
Nearest_Centroid = NearestCentroid()


def convert_str_to_number(x):
    total_stars = 0
    x = x.strip().replace(',', '.')
    num_map = {'K':1000, 'M':1000000, 'B':1000000000}
    if x.isdigit():
        total_stars = int(x)
    else:
        if len(x) > 1:
            total_stars = float(x[:-1]) * num_map.get(x[-1].upper(), 1)
    return int(total_stars)


def log_facebook():
    username = driver.find_elements_by_css_selector("input[name=email]")
    username[0].send_keys('666119777')#find the password field and enter the password password.
    time.sleep(1)
    password = driver.find_elements_by_css_selector("input[name=pass]")
    password[0].send_keys('paulo10COELO')#find the login button and click it.
    time.sleep(1)
    loginButton = driver.find_elements_by_css_selector("button[name=login]")
    loginButton[0].click()
    time.sleep(1)


def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)


def remove_duplicates(l):
    return list(set(l))


# save excel to a destination function
def save_excel():
    target = os.path.join(APP_ROOT, 'testxlsx/')
    dirss = os.listdir(APP_ROOT)
    folders = []
    for dirs in dirss:
        if os.path.isdir(os.path.join(os.getcwd(),  dirs )):
            folders.append(dirs)
    if not os.path.isdir(target):
        os.mkdir(target)
    for file in request.files.getlist("file"):
        print(file)
        filename = file.filename
        destination = "/".join([target, filename])
        print(destination)
        file.save(destination)
        df2 = pd.read_excel(os.path.join(APP_ROOT, 'testxlsx/'+filename))
        if 'CONTENT' in df2.columns:
            df_x = df2['CONTENT']
            all_comments = df2['CONTENT']
            all_not_clean = df2['CONTENT']
            column_name = 'CONTENT'
        elif 'Comments' in df2.columns:
            df_x = df2['Comments']
            all_comments = df2['Comments']
            all_not_clean = df2['Comments']
            column_name = 'Comments'
        elif 'Comment' in df2.columns:
            df_x = df2['Comment']
            all_comments = df2['Comment']
            all_not_clean = df2['Comment']
            column_name = 'Comment'
        elif 'comment' in df2.columns:
            df_x = df2['comment']
            all_comments = df2['comment']
            all_not_clean = df2['comment']
            column_name = 'comment'
        elif 'Commentaires' in df2.columns:
            df_x = df2['Commentaires']
            all_comments = df2['Commentaires']
            all_not_clean = df2['Commentaires']
            column_name = 'Commentaires'
        elif 'Commentaire' in df2.columns:
            df_x = df2['Commentaire']
            all_comments = df2['Commentaire']
            all_not_clean = df2['Commentaire']
            column_name = 'Commentaire'
        else:
            df_x = df2[df2.columns[0]]
            all_comments = df2[df2.columns[0]]
            all_not_clean = df2[df2.columns[0]]
            column_name = df2.columns[0]
        #all_clean = []
        return all_comments, df_x, df2, column_name


def remove_stopwords(df_content = '', column_name = 'Comment', stopword="dareja"):
    df = df_content
    df[column_name]= df[column_name].astype(str)
    #clean = re.compile('<.*?>')
    for i in range(len(df[column_name])):
        df[column_name][i] = re.sub(r'^https?:\/\/.*[\r\n]*', '', df[column_name][i])
        df[column_name][i] = re.sub(r'(.)\1+', r'\1', df[column_name][i] )
        df[column_name][i] = re.sub('\\\\|\?|\!|\،|\,|\.|\=|\_|\+|\-|\)|\(|\(|\)|\[|\]|\{|\}|\^|\°|\@|\؟|<emoji>|</emoji>|\;|\:|\*|\#|\'', ' ', df[column_name][i] )
    if stopword in ['english', 'french', 'arabic']:
        stop = stopwords.words(stopword)
        df['comment'] = df[column_name].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
        df = df.drop(column_name, 1)
        print('selman')
    elif stopword in ['dareja', 'darja'] :
        stop = ['we', 'and', 'w']
        df['comment'] = df[column_name].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
        df = df.drop(column_name, 1)
    elif stopword in ['all', 'All', 'tout', 'Tout'] :
        stop = stop_dareja + stopwords.words("english") + stopwords.words("french") + stopwords.words("arabic")
        df['comment'] = df[column_name].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
        df = df.drop(column_name, 1)
        print('mehdi')
    return df




def remove_stopwords2(df_content = '', column_name = 'Comment', stopword="dareja"):
    df = df_content
    df[column_name]= df[column_name].astype(str)
    #clean = re.compile('<.*?>')
    for i in range(len(df[column_name])):
        df[column_name][i] = re.sub(r'^https?:\/\/.*[\r\n]*', '', df[column_name][i])
        df[column_name][i] = re.sub(r'(.)\1+', r'\1', df[column_name][i] )
        df[column_name][i] = re.sub('\\\\|\?|\!|\،|\,|\.|\=|\_|\+|\-|\)|\(|\(|\)|\[|\]|\{|\}|\^|\°|\@|\؟|<emoji>|</emoji>|\;|\:|\*|\#|\'', ' ', df[column_name][i] )
    if stopword in ['english', 'french', 'arabic']:
        stop = stopwords.words(stopword)
        df['comment'] = df[column_name].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
        df = df.drop(column_name, 1)
        print('selman')
    elif stopword in ['dareja', 'darja'] :
        stop = ['we', 'and', 'w']
        df['comment'] = df[column_name].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
        df = df.drop(column_name, 1)
    elif stopword in ['all', 'All', 'tout', 'Tout'] :
        stop = stop_dareja + stopwords.words("english") + stopwords.words("french") + stopwords.words("arabic")
        df['comment'] = df[column_name].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
        df = df.drop(column_name, 1)
        print('mehdi')
    return df







# cleaning comments from special caracters and more than double letters
# the cleaning funciton
def clean_comments(df_content = '', column_name = 'CONTENT', stopword = '', cleaned = 'FALSE', lemma = False):
    df = df_content
    df_clean = pd.DataFrame()
    if cleaned == 'FALSE':
        df[column_name]= df[column_name].astype(str)
        #clean = re.compile('<.*?>')
        for i in range(len(df[column_name])):
            #df[column_name][i] = re.sub(clean, ' ', df[column_name][i])
            #df[column_name][i] = df[column_name][i].replace('\n',' ')
            #df[column_name][i] = df[column_name][i].replace('\t',' ')
            #df[column_name][i] = df[column_name][i].replace('\r',' ')
            df[column_name][i] = re.sub(r'(.)\1+', r'\1\1', df[column_name][i] )
            df[column_name][i] = re.sub('\\\\|\?|\!|\،|\,|\.|\=|\_|\+|\-|\)|\(|\(|\)|\[|\]|\{|\}|\^|\°|<emoji>|</emoji>|\;|\:|\*|\#|\'', ' ', df[column_name][i] )
    return df








# the cleaning funciton
def clean_text(df_content = '', column_name = 'CONTENT', stopword = '', cleaned = 'FALSE'):
    import pandas as pd
    import mysql.connector
    df = df_content
    if cleaned == 'FALSE':
        df[column_name]= df[column_name].astype(str)
        #clean = re.compile('<.*?>')
        for i in range(len(df[column_name])):
            #df[column_name][i] = re.sub(clean, ' ', df[column_name][i])
            #df[column_name][i] = df[column_name][i].replace('\n',' ')
            #df[column_name][i] = df[column_name][i].replace('\t',' ')
            #df[column_name][i] = df[column_name][i].replace('\r',' ')
            df[column_name][i] = re.sub(r'(.)\1+', r'\1', df[column_name][i] )
            df[column_name][i] = re.sub('\\\\|\?|\!|\،|\,|\.|\=|\_|\+|\-|\)|\(|\(|\)|\[|\]|\{|\}|\^|\°|<emoji>|</emoji>|\;|\:|\*|\#|\'', ' ', df[column_name][i] )
    
    mydb = mysql.connector.connect(
      host="localhost",
      user="root",
      passwd="",
      db = "facebook"
    )
    cursor = mydb.cursor()
    cursor.execute("SELECT * FROM words")
    data = cursor.fetchall()
    to_df = pd.DataFrame(list(data), columns=['index', 'word', 'latin', 'variation'])
    df2 = to_df
    my_text = df[column_name]
    tokenize_sent = [ word_tokenize(i.lower()) for i in my_text ]
    my_new_text = []
    s = 0
    for sent_tkn in tokenize_sent:
        new_sent = []
        #print(s)
        s = s + 1 
        for k in sent_tkn:
            word_matched = df2[df2['variation'].str.contains(pat = r'\b'+ k +r'\b', regex = True)]
            word = list(word_matched.word)
            latin = list(word_matched.latin)
            #type(str(latin))
            if latin:
                new_sent.append(latin[0])
            else:
                new_sent.append(k)
        str_1 = ' '.join(new_sent)
        my_new_text.append(str_1)
    df_clean = pd.DataFrame(my_new_text)
    #df_clean['polarity'] = df['polarity']
    df_clean.columns = ["comment_clean"]
    if stopword in ['english', 'french', 'arabic']:
        stop = stopwords.words(stopword)
        df_clean['comment'] = df_clean['comment_clean'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
        df_clean = df_clean.drop('comment_clean', 1)
        print('selman')
    elif stopword in ['dareja', 'darja'] :
        stop = stop_dareja
        df_clean['comment'] = df_clean['comment_clean'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
        df_clean = df_clean.drop('comment_clean', 1)
        print('mehdi')
    elif stopword in ['all', 'All', 'tout', 'Tout'] :
        stop = stop_dareja
        df_clean['comment'] = df_clean['comment_clean'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
        stop = stopwords.words('french')
        df_clean['comment'] = df_clean['comment_clean'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
        stop = stopwords.words('english')
        df_clean['comment'] = df_clean['comment_clean'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
        df_clean = df_clean.drop('comment_clean', 1)
        print('mehdi')
    else:
        df_clean['comment'] = df_clean['comment_clean']
    return df_clean









# the cleaning funciton
# the cleaning funciton
# the cleaning funciton
def clean_text_for_training(df_content = '', column_name = 'CONTENT', stopword = '', cleaned = 'FALSE'):
    import pandas as pd
    import mysql.connector
    df = df_content
    if cleaned == 'FALSE':
        df[column_name]= df[column_name].astype(str)
        #clean = re.compile('<.*?>')
        for i in range(len(df[column_name])):
            #df[column_name][i] = re.sub(clean, ' ', df[column_name][i])
            #df[column_name][i] = df[column_name][i].replace('\n',' ')
            #df[column_name][i] = df[column_name][i].replace('\t',' ')
            #df[column_name][i] = df[column_name][i].replace('\r',' ')
            df[column_name][i] = re.sub(r'(.)\1+', r'\1', df[column_name][i] )
            df[column_name][i] = re.sub('\\\\|\?|\!|\،|\,|\.|\=|\_|\+|\-|\)|\(|\(|\)|\[|\]|\{|\}|\^|\°|<emoji>|</emoji>|\;|\:|\*|\#|\'', ' ', df[column_name][i] )
    
    mydb = mysql.connector.connect(
      host="localhost",
      user="root",
      passwd="",
      db = "facebook"
    )
    cursor = mydb.cursor()
    cursor.execute("SELECT * FROM words")
    data = cursor.fetchall()
    to_df = pd.DataFrame(list(data), columns=['index', 'word', 'latin', 'variation'])
    df2 = to_df
    my_text = df[column_name]
    tokenize_sent = [ word_tokenize(i.lower()) for i in my_text ]
    my_new_text = []
    s = 0
    for sent_tkn in tokenize_sent:
        new_sent = []
        print(s)
        s = s + 1 
        for k in sent_tkn:
            word_matched = df2[df2['variation'].str.contains(pat = r'\b'+ k +r'\b', regex = True)]
            word = list(word_matched.word)
            latin = list(word_matched.latin)
            #type(str(latin))
            if latin:
                new_sent.append(latin[0])
            else:
                new_sent.append(k)
        str_1 = ' '.join(new_sent)
        my_new_text.append(str_1)
    df_clean = pd.DataFrame(my_new_text)
    #df_clean['polarity'] = df['polarity']
    df_clean.columns = ["comment_clean"]
    if stopword in ['english', 'french', 'arabic']:
        stop = stopwords.words(stopword)
        df_clean['comment'] = df_clean['comment_clean'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
        df_clean = df_clean.drop('comment_clean', 1)
        print('selman')
    elif stopword in ['dareja', 'darja'] :
        stop = stop_dareja
        df_clean['comment'] = df_clean['comment_clean'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
        df_clean = df_clean.drop('comment_clean', 1)
        print('mehdi')
    elif stopword in ['all', 'All', 'tout', 'Tout'] :
        stop = stop_dareja
        df_clean['comment'] = df_clean['comment_clean'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
        stop = stopwords.words('french')
        df_clean['comment'] = df_clean['comment_clean'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
        stop = stopwords.words('english')
        df_clean['comment'] = df_clean['comment_clean'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
        df_clean = df_clean.drop('comment_clean', 1)
        print('mehdi')
    else:
        df_clean['comment'] = df_clean['comment_clean']
    return df_clean





def get_determining_words(word_class = "negative"):
    cursor.execute("""SELECT word FROM determining_words   
        WHERE class=%s ORDER BY id""", (word_class, ))
    my_words = cursor.fetchall()
    words = []
    for i in range(len(my_words)):
        words.append(my_words[i][0])
    return words







def get_words_by_topic(word_class = "", topic_id = "1"):
    cursor.execute("""SELECT word FROM topic_words   
        WHERE class=%s AND topic_id =%s ORDER BY id""", (word_class, topic_id ))
    my_words = cursor.fetchall()
    words = []
    for i in range(len(my_words)):
        words.append(my_words[i][0])
    return words




def get_telecom_topics(results_df = ''):
    # Geeting the topics from the comments 
    internet_list = ['connexion', 'lconx', '4g', '3g', 'internet', 'cnx', 'connx'] 
    reseau_list = ['reseau', 'reseaux', 'riseau', 'mahekemetoush', 'mahkemetesh', 'taghtiya', 'taghtia'] 
    offres_list = ['offre', 'offrir', '3ard'] 
    appelle_list = ['ne3eyet', 'appelle', 'appeller'] 
    # Get topics numbers for all comments 
    cats = []
    all_cats = []
    for list_element in results_df.Comment:
        cats = []
        internet_test = [ele for ele in internet_list if(ele in list_element)]
        if internet_test:
            cats.append("internet")
        reseau_test = [ele for ele in reseau_list if(ele in list_element)]
        if reseau_test :
            cats.append("reseau")
        offres_test = [ele for ele in offres_list if(ele in list_element)]
        if offres_test:
            cats.append("offre")
        appelle_test = [ele for ele in appelle_list if(ele in list_element)]
        if appelle_test :
            cats.append("appelle")
        if not cats:
            cats.append('other')
        all_cats.append(cats)
    all_word_cats = []
    for i in all_cats:
        for k in i:
            all_word_cats.append(k)
    count_internet = all_word_cats.count('internet')
    count_reseau = all_word_cats.count('reseau')
    count_offre = all_word_cats.count('offre')
    count_appelle = all_word_cats.count('appelle')
    count_other = all_word_cats.count('other')
    all_comments_count = len(results_df.Comment)
    internet_percent = round(count_internet * 100/ all_comments_count, 2)
    reseau_percent = round(count_reseau * 100/ all_comments_count, 2)
    offre_percent = round(count_offre * 100/ all_comments_count, 2)
    appelle_percent = round(count_appelle * 100/ all_comments_count, 2)
    other_percent = round(count_other * 100/ all_comments_count, 2)
    results_df_pos = results_df[results_df['Polarity'] == 'Super']
    results_df_neu = results_df[results_df['Polarity'] == 'Good']
    results_df_neg = results_df[results_df['Polarity'] == 'Bad']
    # Get topics numbers for Super comments 
    cats = []
    all_cats = []
    for list_element in results_df_pos.Comment:
        cats = []
        internet_test = [ele for ele in internet_list if(ele in list_element)]
        if internet_test:
            cats.append("internet")
        reseau_test = [ele for ele in reseau_list if(ele in list_element)]
        if reseau_test :
            cats.append("reseau")
        offres_test = [ele for ele in offres_list if(ele in list_element)]
        if offres_test:
            cats.append("offre")
        appelle_test = [ele for ele in appelle_list if(ele in list_element)]
        if appelle_test :
            cats.append("appelle")
        if not cats:
            cats.append('other')
        all_cats.append(cats)
    all_word_cats = []
    for i in all_cats:
        for k in i:
            all_word_cats.append(k)
    pos_count_internet = all_word_cats.count('internet')
    pos_count_reseau = all_word_cats.count('reseau')
    pos_count_offre = all_word_cats.count('offre')
    pos_count_appelle = all_word_cats.count('appelle')
    pos_count_other = all_word_cats.count('other')
    # Get topics numbers for Good Comments 
    cats = []
    all_cats = []
    for list_element in results_df_neu.Comment:
        cats = []
        internet_test = [ele for ele in internet_list if(ele in list_element)]
        if internet_test:
            cats.append("internet")
        reseau_test = [ele for ele in reseau_list if(ele in list_element)]
        if reseau_test :
            cats.append("reseau")
        offres_test = [ele for ele in offres_list if(ele in list_element)]
        if offres_test:
            cats.append("offre")
        appelle_test = [ele for ele in appelle_list if(ele in list_element)]
        if appelle_test :
            cats.append("appelle")
        if not cats:
            cats.append('other')
        all_cats.append(cats)
    all_word_cats = []
    for i in all_cats:
        for k in i:
            all_word_cats.append(k)
    neu_count_internet = all_word_cats.count('internet')
    neu_count_reseau = all_word_cats.count('reseau')
    neu_count_offre = all_word_cats.count('offre')
    neu_count_appelle = all_word_cats.count('appelle')
    neu_count_other = all_word_cats.count('other')
    # Get topics numbers for Bad Comments 
    cats = []
    all_cats = []
    for list_element in results_df_neg.Comment:
        cats = []
        internet_test = [ele for ele in internet_list if(ele in list_element)]
        if internet_test:
            cats.append("internet")
        reseau_test = [ele for ele in reseau_list if(ele in list_element)]
        if reseau_test :
            cats.append("reseau")
        offres_test = [ele for ele in offres_list if(ele in list_element)]
        if offres_test:
            cats.append("offre")
        appelle_test = [ele for ele in appelle_list if(ele in list_element)]
        if appelle_test :
            cats.append("appelle")
        if not cats:
            cats.append('other')
        all_cats.append(cats)
    all_word_cats = []
    for i in all_cats:
        for k in i:
            all_word_cats.append(k)
    neg_count_internet = all_word_cats.count('internet')
    neg_count_reseau = all_word_cats.count('reseau')
    neg_count_offre = all_word_cats.count('offre')
    neg_count_appelle = all_word_cats.count('appelle')
    neg_count_other = all_word_cats.count('other')
    subjects_count = {
        'neg_count_internet': neg_count_internet,
        'neg_count_reseau': neg_count_reseau,
        'neg_count_offre': neg_count_offre,
        'neg_count_appelle': neg_count_appelle,
        'neg_count_other': neg_count_other,
        'neu_count_internet': neu_count_internet,
        'neu_count_reseau': neu_count_reseau,
        'neu_count_offre': neu_count_offre,
        'neu_count_appelle': neu_count_appelle,
        'neu_count_other': neu_count_other,
        'pos_count_internet': pos_count_internet,
        'pos_count_reseau': pos_count_reseau,
        'pos_count_offre': pos_count_offre,
        'pos_count_appelle': pos_count_appelle,
        'pos_count_other': pos_count_other,
        'count_internet': count_internet,
        'count_reseau': count_reseau,
        'count_offre': count_offre,
        'count_appelle': count_appelle,
        'count_other': count_other, 
        'internet_percent' : internet_percent,
        'reseau_percent' : reseau_percent,
        'offre_percent' : offre_percent,
        'appelle_percent' : appelle_percent,
        'other_percent' : other_percent
    }
    return subjects_count, all_cats




















def get_eau_topics(results_df = ''):
    # Geeting the topics from the comments 
    coupure_list = get_words_by_topic("coupure", '1')
    annonce_list = get_words_by_topic("annonce", '1') 
    # Get topics numbers for all comments 
    cats = []
    all_cats = []
    for list_element in results_df.Comment:
        cats = []
        coupure_test = [ele for ele in coupure_list if(ele in list_element)]
        if coupure_test:
            cats.append("coupure")
        annonce_test = [ele for ele in annonce_list if(ele in list_element)]
        if annonce_test :
            cats.append("annonce")
        if not cats:
            cats.append('other')
        all_cats.append(cats)
    all_word_cats = []
    for i in all_cats:
        for k in i:
            all_word_cats.append(k)
    count_coupure = all_word_cats.count('coupure')
    count_annonce = all_word_cats.count('annonce')
    count_other = all_word_cats.count('other')
    all_comments_count = len(results_df.Comment)
    coupure_percent = round(count_coupure * 100/ all_comments_count, 2)
    annonce_percent = round(count_annonce * 100/ all_comments_count, 2)
    other_percent = round(count_other * 100/ all_comments_count, 2)
    results_df_pos = results_df[results_df['Polarity'] == 'Super']
    results_df_neu = results_df[results_df['Polarity'] == 'Good']
    results_df_neg = results_df[results_df['Polarity'] == 'Bad']
    # Get topics numbers for Super comments 
    cats = []
    all_cats = []
    for list_element in results_df_pos.Comment:
        cats = []
        coupure_test = [ele for ele in coupure_list if(ele in list_element)]
        if coupure_test:
            cats.append("coupure")
        annonce_test = [ele for ele in annonce_list if(ele in list_element)]
        if annonce_test :
            cats.append("annonce")
        if not cats:
            cats.append('other')
        all_cats.append(cats)
    all_word_cats = []
    for i in all_cats:
        for k in i:
            all_word_cats.append(k)
    pos_count_coupure = all_word_cats.count('coupure')
    pos_count_annonce = all_word_cats.count('annonce')
    pos_count_other = all_word_cats.count('other')
    # Get topics numbers for Good Comments 
    cats = []
    all_cats = []
    for list_element in results_df_neu.Comment:
        cats = []
        coupure_test = [ele for ele in coupure_list if(ele in list_element)]
        if coupure_test:
            cats.append("coupure")
        annonce_test = [ele for ele in annonce_list if(ele in list_element)]
        if annonce_test :
            cats.append("annonce")
        if not cats:
            cats.append('other')
        all_cats.append(cats)
    all_word_cats = []
    for i in all_cats:
        for k in i:
            all_word_cats.append(k)
    neu_count_coupure = all_word_cats.count('coupure')
    neu_count_annonce = all_word_cats.count('annonce')
    neu_count_other = all_word_cats.count('other')
    # Get topics numbers for Bad Comments 
    cats = []
    all_cats = []
    for list_element in results_df_neg.Comment:
        cats = []
        coupure_test = [ele for ele in coupure_list if(ele in list_element)]
        if coupure_test:
            cats.append("coupure")
        annonce_test = [ele for ele in annonce_list if(ele in list_element)]
        if annonce_test :
            cats.append("annonce")
        if not cats:
            cats.append('other')
        all_cats.append(cats)
    all_word_cats = []
    for i in all_cats:
        for k in i:
            all_word_cats.append(k)
    neg_count_coupure = all_word_cats.count('coupure')
    neg_count_annonce = all_word_cats.count('annonce')
    neg_count_other = all_word_cats.count('other')
    subjects_count = {
        'neg_count_coupure': neg_count_coupure,
        'neg_count_annonce': neg_count_annonce,
        'neg_count_other': neg_count_other,
        'neu_count_coupure': neu_count_coupure,
        'neu_count_annonce': neu_count_annonce,
        'neu_count_other': neu_count_other,
        'pos_count_coupure': pos_count_coupure,
        'pos_count_annonce': pos_count_annonce,
        'pos_count_other': pos_count_other,
        'count_coupure': count_coupure,
        'count_annonce': count_annonce,
        'count_other': count_other,
        'coupure_percent' : coupure_percent,
        'annonce_percent' : annonce_percent,
        'other_percent' : other_percent
    }
    return subjects_count, all_cats




















def get_sentiment_analysis(table = '', model = 'threeClass'):
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import CountVectorizer
    import pickle
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    lr = LogisticRegression()
    from sklearn.externals import joblib 
    from sklearn.naive_bayes import MultinomialNB
    ### Load and Test
    # Use a default model 
    if model == 'default':
        test_path = os.path.join(APP_ROOT, 'models/mahdi10/threeClass')
        ytb_model = open(test_path +"/tg_cvec.pkl", "rb")
        tg_cvec = joblib.load(ytb_model)
        ytb_model = open(test_path  +"/sentiment_fit.pkl", "rb")
    else:
        test_path = os.path.join(APP_ROOT, 'models\\' + current_user.username)
        ytb_model = open(test_path + "/"  + model +"/tg_cvec.pkl", "rb")
        tg_cvec = joblib.load(ytb_model)
        ytb_model = open(test_path + "/"  + model +"/sentiment_fit.pkl", "rb")
    sentiment_fit = joblib.load(ytb_model)
    tg_pipeline = Pipeline([
        ('vectorizer', tg_cvec),
        ('classifier', lr)
    ])
    positive_words = get_determining_words("positive")
    negative_words = get_determining_words("negative")
    #df = pd.read_excel("./test-dareja.xlsx", encoding='utf-8')
    results_df = pd.DataFrame()
    results_df['Comment'] = table.comment
    s2 = pd.Series(results_df['Comment'])
    my_prediction = sentiment_fit.predict(s2)
    my_prediction_length = len(my_prediction)
    i = 0
    for list_element in results_df.Comment:
        negative_test = [ele for ele in negative_words if(ele in list_element)]
        positive_test = [ele for ele in positive_words if(ele in list_element)]
        if negative_test:
            my_prediction[i] = 0
        elif positive_test:
            my_prediction[i] = 4
        i += 1

    test_preds = []
    i = 0
    for i in my_prediction:
        if i == 4: 
            my_prediction_cat = 'Super'
        elif i == 2:
            my_prediction_cat = 'Good'
        else:
            my_prediction_cat = 'Bad'
        test_preds.append(my_prediction_cat)
        i = i + 1
    results_df['Polarity']  = test_preds
    results_df_pos = results_df[results_df['Polarity'] == 'Super']
    results_df_neu = results_df[results_df['Polarity'] == 'Good']
    results_df_neg = results_df[results_df['Polarity'] == 'Bad']
    pos_percentage = round(len(results_df[results_df['Polarity'] == "Super"]) * 100/len(results_df['Polarity']), 2)
    neu_percentage = round(len(results_df[results_df['Polarity'] == "Good"]) * 100/len(results_df['Polarity']), 2)
    neg_percentage = round(len(results_df[results_df['Polarity'] == "Bad"]) * 100/len(results_df['Polarity']), 2)
    results_percents =  {
          "pos_percentage": pos_percentage,
          "neu_percentage": neu_percentage,
          "neg_percentage": neg_percentage
        }
    return results_df, results_percents, results_df_pos, results_df_neu, results_df_neg, my_prediction








def get_sentiment_analysis2(table = '', model = 'seaal', type="rnn", vocab_to_int = []):
    ### Load and Test
    # Use a default model 
    import csv
    import tensorflow as tf
    import numpy as np
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words('english'))
    import pandas as pd 
    target = os.path.join(APP_ROOT, 'models/' + model + '/')
    # Specify a path
    from sklearn.externals import joblib 
    cursor.execute("""SELECT name, type, seq_length, n_layers, user_id, 
        vocab_size, embedding_dim, trunc_type, padding_type, oov_tok
        FROM `models` WHERE name = %s""", ( model, ))
    my_data = cursor.fetchall()
    model_info = pd.DataFrame(list(my_data), columns=['name', 'type', 'seq_length', 'n_layers', 'user_id', 'vocab_size', 'embedding_dim', 'trunc_type', 'padding_type', 'oov_tok'])
    type = model_info['type'][0]
    vocab_size = model_info['vocab_size'][0]
    embedding_dim = model_info['embedding_dim'][0]
    max_length = model_info['seq_length'][0]
    trunc_type = model_info['trunc_type'][0]
    padding_type = model_info['padding_type'][0]
    oov_tok = model_info['oov_tok'][0]
    if type == 'rnn':
        from sklearn.externals import joblib 
        vocab_to_int = open( target + "vocab_to_int.pkl", "rb")
        vocab_to_int = joblib.load(vocab_to_int)
        vocab_size = len(vocab_to_int)+1
        output_size = 1
        embedding_dim = 400
        hidden_dim = 256
        vocab_size = 10000
        embedding_dim = 16
        max_length = 40
        trunc_type = 'post'
        padding_type = 'post'
        oov_tok = '<OOV>'
        n_layers = model_info['n_layers'][0]
        seq_length = model_info['seq_length'][0]
        net = SentimentRNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
        #if(train_on_gpu):
        #    print('Training on GPU.')
        #else:
        #    print('No GPU available, training on CPU.')
        PATH = target + "state_dict_model.pt"
        net.load_state_dict(torch.load(PATH))
        net.eval()
        my_prediction = []
        results_df = pd.DataFrame()
        results_df['Comment'] = table.Comment
        for i in table['Comment']:
            my_str = i
            print(i)
            sent = predict(net, i, seq_length)
            my_prediction.append(sent)
        results_df['Polarity']  = my_prediction
    elif type == "keras":
        model_name = 'my_keras_model.h5'
        from tensorflow import keras
        model = keras.models.load_model(target + model_name)
        from sklearn.externals import joblib 
        tokenizer = open(target + "tokenizer.pkl", "rb")
        tokenizer = joblib.load(tokenizer)
        my_prediction = []
        results_df = pd.DataFrame()
        results_df['Comment'] = table.comment
        comments_sequences = tokenizer.texts_to_sequences(results_df['Comment'])
        comments_padded = pad_sequences(comments_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
        predictions = model.predict(comments_padded)
        my_pred = predictions.argmax(axis=1)
        #for i in new_phrases:
        #    print(i)
        #
        #    my_pred = model.predict_classes(np.array(i))[0]
        #    my_prediction.append(my_pred)
        test_preds = []
        i = 0
        for i in my_pred:
            if i == 3: 
                my_prediction_cat = 'Super'
            elif i == 2:
                my_prediction_cat = 'Good'
            else:
                my_prediction_cat = 'Bad'
            test_preds.append(my_prediction_cat)
            i = i + 1
        results_df['Polarity']  = test_preds
    elif type == "regression":
        lr = LogisticRegression()
        if model == 'default':
            ytb_model = open(target +"/tg_cvec.pkl", "rb")
            tg_cvec = joblib.load(ytb_model)
            ytb_model = open(test_path  +"/sentiment_fit.pkl", "rb")
        else:
            ytb_model = open( target +"/tg_cvec.pkl", "rb")
            tg_cvec = joblib.load(ytb_model)
            ytb_model = open(target +"/sentiment_fit.pkl", "rb")
        sentiment_fit = joblib.load(ytb_model)
        tg_pipeline = Pipeline([
            ('vectorizer', tg_cvec),
            ('classifier', lr)
        ])
        positive_words = get_determining_words("positive")
        negative_words = get_determining_words("negative")
        #df = pd.read_excel("./test-dareja.xlsx", encoding='utf-8')
        results_df = pd.DataFrame()
        results_df['Comment'] = table.comment
        s2 = pd.Series(results_df['Comment'])
        my_prediction = sentiment_fit.predict(s2)
        my_prediction_length = len(my_prediction)
        #i = 0
        #for list_element in results_df.Comment:
        #    print(list_element)
        #    negative_test = [ele for ele in negative_words if(ele in list_element)]
        #    positive_test = [ele for ele in positive_words if(ele in list_element)]
        #    if negative_test:
        #        my_prediction[i] = 0
        #    elif positive_test:
        #        my_prediction[i] = 4
        #    i += 1
        test_preds = []
        i = 0
        for i in my_prediction:
            if i == 4: 
                my_prediction_cat = 'Super'
            elif i == 2:
                my_prediction_cat = 'Good'
            else:
                my_prediction_cat = 'Bad'
            test_preds.append(my_prediction_cat)
            i = i + 1
        results_df['Polarity']  = test_preds
    results_df_pos = results_df[results_df['Polarity'] == 'Super']
    results_df_neu = results_df[results_df['Polarity'] == 'Good']
    results_df_neg = results_df[results_df['Polarity'] == 'Bad']
    pos_percentage = round(len(results_df[results_df['Polarity'] == "Super"]) * 100/len(results_df['Polarity']), 2)
    neu_percentage = round(len(results_df[results_df['Polarity'] == "Good"]) * 100/len(results_df['Polarity']), 2)
    neg_percentage = round(len(results_df[results_df['Polarity'] == "Bad"]) * 100/len(results_df['Polarity']), 2)
    results_percents =  {
          "pos_percentage": pos_percentage,
          "neu_percentage": neu_percentage,
          "neg_percentage": neg_percentage
        }
    return results_df, results_percents, results_df_pos, results_df_neu, results_df_neg, my_prediction













# END OF IMPORT PACKAGES
APP_ROOT = os.path.dirname(os.path.abspath(__file__))



analyse = Blueprint('analyse', __name__)





@analyse.route('/train')
def train():
    title = "Training Page"
    return render_template('train.html', title = title)

@analyse.route('/analyse_excel')
def analyse_excel():
    if not current_user.is_authenticated:
        flash('You should be logged in to acced this page', 'info')
        next_page =  request.endpoint
        return redirect(url_for('users.login', next = next_page))
    previous_link = request.referrer 
    title = "Analyse des sentiments a partir d'un fichier excel "
    folders = get_models()
    user_id = current_user.id
    cursor.execute("""SELECT name, type FROM `models` WHERE user_id = %s OR user_id = 1""", (user_id,))
    my_models = cursor.fetchall()
    models = []
    for i in my_models:
        models.append(i[0])
    return render_template('analyse_excel.html', title = title, 
        models = models)


@analyse.route('/compare_polarity')
def compare_polarity():
    title = "Compare Polarity"
    folders = get_models()
    return render_template('compare_polarity.html', title = title, 
        folders = folders)



@analyse.route('/fb_search_posts')
def fb_search_posts():
    title = "Facebook Search Posts"
    return render_template('fb_search_posts.html', title = title)


@analyse.route('/fb_posts_results', methods=['POST'])
def fb_posts_results():
    global new_df
    from selenium.common.exceptions import NoSuchElementException
    from mysql.connector import Error
    title = "Facebook Keyword Results"
    if request.method == "POST":
        fb_keyword = request.form['fb_keyword']
    from selenium import webdriver

    driver = webdriver.Chrome(executable_path="C:/Users/dell/Downloads/chromedriver_win32_4/chromedriver")

    driver.get("https://m.facebook.com")
    import time
    time.sleep(5)

    
    #find the username field and enter the email example@yahoo.com.
    username = driver.find_elements_by_css_selector("input[name=email]")
    username[0].send_keys('666119777')#find the password field and enter the password password.
    time.sleep(1)
    password = driver.find_elements_by_css_selector("input[name=pass]")
    password[0].send_keys('paulo10COELO')#find the login button and click it.
    time.sleep(1)
    loginButton = driver.find_elements_by_css_selector("button[name=login]")
    loginButton[0].click()
    time.sleep(5)

    #my_search = "https://mobile.facebook.com/search/top/?q="+fb_keyword+"&epa=FILTERS&filters=eyJycF9sb2NhdGlvbiI6IntcIm5hbWVcIjpcImxvY2F0aW9uXCIsXCJhcmdzXCI6XCIxMDY0OTg1MTI3MjE0MDdcIn0iLCJycF9jcmVhdGlvbl90aW1lIjoie1wibmFtZVwiOlwiY3JlYXRpb25fdGltZVwiLFwiYXJnc1wiOlwie1xcXCJzdGFydF9tb250aFxcXCI6XFxcIjIwMTktMTBcXFwiLFxcXCJlbmRfbW9udGhcXFwiOlxcXCIyMDE5LTEwXFxcIn1cIn0ifQ%3D%3D"
    
    my_search = "https://mobile.facebook.com/search/top/?q="+fb_keyword+"&epa=FILTERS&filters=eyJycF9jcmVhdGlvbl90aW1lIjoie1wibmFtZVwiOlwiY3JlYXRpb25fdGltZVwiLFwiYXJnc1wiOlwie1xcXCJzdGFydF9tb250aFxcXCI6XFxcIjIwMTktMTBcXFwiLFxcXCJlbmRfbW9udGhcXFwiOlxcXCIyMDE5LTEwXFxcIn1cIn0ifQ%3D%3D"

    #my_search = "https://mobile.facebook.com/search/top/?q="+fb_keyword

    driver.get(my_search)
    time.sleep(5)
    driver.get(my_search)
    time.sleep(5)
    
    driver.get(my_search)
    time.sleep(5)
    driver.get(my_search)

    
    
    urls_search = []
    for k in range(1, 6):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(5)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(5)
        try:
            my_link = "div[data-testid='results'] div:nth-child(" + str(k) +") div[data-nt='NT:BOX_3_CHILD'] div[data-nt='NT:DECOR']"
            check_url = driver.find_element_by_css_selector(my_link)
            check_url.click()
            time.sleep(5)
            urls_search.append(driver.current_url)
            #driver.execute_script("window.history.go(-1)")
            driver.get(my_search)
            time.sleep(3)
        except NoSuchElementException:  #spelling error making this code not work as expected
            pass

    # end of loading the required packages
    # Create a folder to save images 
    target = os.path.join(APP_ROOT, 'saved_screenshots/')
    print(target)

    if not os.path.isdir(target):
        os.mkdir(target)
        # end of creating a folder for saving images 

    titles = []
    posts = []
    dates = []
    links = []
    for url in range(0, len(urls_search)):
        fb_link = urls_search[url]
        driver.get(fb_link)
        time.sleep(5)
        try:
            #my_link = "div[data-testid='results'] div:nth-child(" + str(k) +") div[data-nt='NT:BOX_3_CHILD'] div[data-nt='NT:DECOR']"
            #check_url = driver.find_element_by_css_selector(my_link)
            #check_url.click()
            #time.sleep(5)
            try:
              title_brute = driver.find_element_by_css_selector("div[class='story_body_container'] > header h3")
              title = title_brute.text
            except NoSuchElementException:  #spelling error making this code not work as expected
                title = "Theere are no Title available for this element"
            titles.append(title)
            time.sleep(2)
            try:
              date_brute = driver.find_element_by_css_selector("div[class='story_body_container'] > header div[data-sigil='m-feed-voice-subtitle'] abbr")
              date = date_brute.text
            except NoSuchElementException:  #spelling error making this code not work as expected
                date = "Date is not available for this post"
            dates.append(date)
            time.sleep(2)
            try:
              post_brute = driver.find_element_by_css_selector("div[class='story_body_container'] > div > p")
              post = post_brute.text
            except NoSuchElementException:  #spelling error making this code not work as expected
                post = "Theere are no text for this element"
            posts.append(post)
            time.sleep(2)
            #driver.set_window_size(520, 880)
            picture = driver.find_element_by_css_selector("div[class='story_body_container']")
            location = picture.location
            size = picture.size
            m = re.search('story_fbid=(.+?)&', fb_link)
            found = m.group(1)
            matime = pd.to_datetime('today')  
            timeStr = matime.strftime("%H_%M_%S")
            image_name = "image_" + found + timeStr + ".png"

            driver.save_screenshot(target + "/" + image_name)
            x = location['x']
            y = location['y']
            width = location['x']+size['width']
            height = location['y']+size['height']
            im = Image.open(target + "/" + image_name)
            im = im.crop((int(x), int(y), int(width), int(height)))
            im.save(target + "/" + image_name)
            links.append(driver.current_url)
            time.sleep(3)
            #driver.execute_script("window.history.go(-1)")
            try:
                test_link = cursor.execute( "SELECT * FROM posts WHERE link = %s " , (fb_link,)  )
                row_count = cursor.rowcount
                print ("number of affected rows: {}".format(row_count))
                if row_count == 0:
                    result = cursor.execute( "INSERT INTO posts (link, title, content, image_name, date_created) VALUES ( %s , %s , %s, %s , %s)" , (fb_link, title, post, image_name ,date)  )
                    conn.commit()
                    print("Laptop Table created successfully ")
                else:
                    print ("This Post has already been on database")
            except NoSuchElementException:
                print("Failed to create entry in the table")
            driver.get(my_search)
            time.sleep(5)
        except NoSuchElementException:
            print('sorry, invalid URL')
    
    
   
        

    return render_template('fb_posts_results.html', title = title, 
        urls_search = urls_search, titles = titles, posts = posts, 
        dates = dates)





@analyse.route('/fb_search_kw')
def fb_search_kw():
    title = "Facebook Search By Keyword"
    return render_template('fb_search_kw.html', title = title)


@analyse.route('/fb_kw_results', methods=['POST'])
def fb_kw_results():
    global new_df
    from selenium.common.exceptions import NoSuchElementException
    title = "Facebook Keyword Results"
    if request.method == "POST":
        fb_keyword = request.form['fb_keyword']
    from selenium import webdriver
    driver = webdriver.Chrome(executable_path="C:/Users/dell/Downloads/chromedriver_win32_4/chromedriver")
    driver.get("https://mobile.facebook.com")
    import time
    time.sleep(5)
    #find the username field and enter the email example@yahoo.com.
    username = driver.find_elements_by_css_selector("input[name=email]")
    username[0].send_keys('666119777')#find the password field and enter the password password.
    time.sleep(1)
    password = driver.find_elements_by_css_selector("input[name=pass]")
    password[0].send_keys('paulo10COELO')#find the login button and click it.
    time.sleep(1)
    loginButton = driver.find_elements_by_css_selector("button[name=login]")
    loginButton[0].click()
    time.sleep(5)
    #my_search = "https://mobile.facebook.com/search/top/?q="+fb_keyword+"&epa=FILTERS&filters=eyJycF9sb2NhdGlvbiI6IntcIm5hbWVcIjpcImxvY2F0aW9uXCIsXCJhcmdzXCI6XCIxMDY0OTg1MTI3MjE0MDdcIn0iLCJycF9jcmVhdGlvbl90aW1lIjoie1wibmFtZVwiOlwiY3JlYXRpb25fdGltZVwiLFwiYXJnc1wiOlwie1xcXCJzdGFydF9tb250aFxcXCI6XFxcIjIwMTktMTBcXFwiLFxcXCJlbmRfbW9udGhcXFwiOlxcXCIyMDE5LTEwXFxcIn1cIn0ifQ%3D%3D"
    my_search = "https://mobile.facebook.com/search/top/?q="+fb_keyword
    driver.get(my_search)
    time.sleep(5)
    driver.get(my_search)
    time.sleep(5)
    
    driver.get(my_search)
    time.sleep(5)
    driver.get(my_search)

    urls_search = []
    for k in range(1, 4):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(5)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(5)
        try:
            my_link = "div[data-testid='results'] div:nth-child(" + str(k) +") div[data-nt='NT:BOX_3_CHILD'] div[data-nt='NT:DECOR']"
            check_url = driver.find_element_by_css_selector(my_link)
            check_url.click()
            time.sleep(5)
            urls_search.append(driver.current_url)
            #driver.execute_script("window.history.go(-1)")
            driver.get(my_search)
            time.sleep(3)
        except NoSuchElementException:  #spelling error making this code not work as expected
            pass

    # loading the required packages for machine learning
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import CountVectorizer
    import pickle
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    lr = LogisticRegression()
    from sklearn.externals import joblib 
    from sklearn.naive_bayes import MultinomialNB
    ### Load and Test
    ytb_model = open("models/dareja/tg_cvec.pkl", "rb")
    tg_cvec = joblib.load(ytb_model)
    ytb_model = open("models/dareja/sentiment_fit.pkl", "rb")
    sentiment_fit = joblib.load(ytb_model)
    tg_pipeline = Pipeline([
        ('vectorizer', tg_cvec),
        ('classifier', lr)
    ]) 
    # end of loading the required packages
    for url in range(0, 2):

        fb_link = urls_search[url]

        driver.get(fb_link)
        

        import time
        time.sleep(5)

        titles = driver.find_elements_by_css_selector("._2b05 > a")

        links = {}
        i = 0
        for title in titles:
          print(title.get_attribute('href'))
          links[i] = title.get_attribute('href')
          i += 1

        #driver.find_element_by_css_selector("div[class='async_elem']>a[data-sigil='ajaxify']").click()
        import time 
        comment_click = driver.find_element_by_css_selector("div[class='async_elem']>a[data-sigil='ajaxify']")
        while comment_click:
            try:
                driver.find_element_by_css_selector("div[class='async_elem']>a[data-sigil='ajaxify']").click()
                time.sleep(5)
                break
            except ValueError:
                print('sorry, invalid input')

        comments = driver.find_elements_by_css_selector("div[data-sigil='comment-body']")

        commentstxt = {}
        i = 0

        commentstxt = {}
        i = 0
        for comment in comments:
            commentstxt[i] = comment.text
            i += 1


        test_path = os.path.join(APP_ROOT, 'models\\' + current_user.username)
        dirss = os.listdir( test_path )
        folders = []

        for dirs in dirss:
            if os.path.isdir(test_path + '\\' + dirs):
                folders.append(dirs)
            
        new_df = pd.DataFrame.from_dict(commentstxt, orient='index', columns=['Comment'])    
        

        new_df_comments = commentstxt
        s2 = pd.Series(new_df_comments)
        my_prediction = sentiment_fit.predict(s2)

        test_preds = []
        for i in range(len(my_prediction)):
            if my_prediction[i] == 4: 
                my_prediction_cat = 'Super'
            else:
                my_prediction_cat = 'Bad'
            test_preds.append(my_prediction_cat)
        new_df['Polarity'] = pd.DataFrame(test_preds)
        new_df_length = len(new_df)
        globals() ["new_df" + str(url + 1) ] = new_df

        #pause(10000) # pause/sleeps for 10 seconds

    driver.quit()

    

    return render_template('fb_kw_results.html', title = title, 
        urls_search = urls_search, 
        links = links, commentstxt = commentstxt, 
        new_df1 = new_df1,
        new_df2 = new_df2)


@analyse.route('/fb_scraping')
def fb_scraping():
    # redirect user if not logged in
    if not current_user.is_authenticated:
        flash('You should be logged in to acced this page', 'info')
        next_page =  request.endpoint
        return redirect(url_for('users.login', next = next_page))
    title = "Facebook Scraping"
    # retrieve user created models
    user_id = current_user.id
    #folders = get_models()
    cursor.execute("""SELECT name, type FROM `models` WHERE user_id = %s OR user_id = 1""", (user_id,))
    my_models = cursor.fetchall()
    models = []
    for i in my_models:
        models.append(i[0])
    return render_template('fb_scraping.html', title = title, 
        folders = models )




@analyse.route('/fb_link_results', methods=['POST', 'GET'])
def fb_link_results():
    if not current_user.is_authenticated:
        flash('You should be logged in to acced this page', 'info')
        next_page =  request.endpoint
        return redirect(url_for('users.login', next = next_page))
    if request.method != "POST":
        flash("You don't have permission accessing this page", 'info')
        return redirect(url_for('main.dashboard'))
    global new_df
    title = "Facebook Scrapper By Keyboard"
    if request.method == "POST":
        fb_link = request.form['fb_link']
    from datetime import datetime
    from selenium import webdriver
    from selenium.common.exceptions import NoSuchElementException
    
    driver = webdriver.Chrome(executable_path="C:/Users/dell/Downloads/chromedriver_win32_4/chromedriver")

    driver.get("https://mobile.facebook.com")
    
    #find the username field and enter the email example@yahoo.com.
    username = driver.find_elements_by_css_selector("input[name=email]")
    username[0].send_keys('666119777')#find the password field and enter the password password.
    password = driver.find_elements_by_css_selector("input[name=pass]")
    password[0].send_keys('paulo10COELO')#find the login button and click it.
    try:
        loginButton = driver.find_elements_by_css_selector("button[name=login]")
        loginButton[0].click()
    except NoSuchElementException:
        print('There are no load more buttons')

    try:
        loginButton = driver.find_element_by_css_selector("input[name=login]")
        loginButton.click()
    except:
        print("Couldn't find login button")

    import time
    time.sleep(5)

    driver.get(fb_link)
    time.sleep(5)

    titles = driver.find_elements_by_css_selector("._2b05 > a")

    links = {}
    i = 0
    for title in titles:
      print(title.get_attribute('href'))
      links[i] = title.get_attribute('href')
      i += 1

    #driver.find_element_by_css_selector("div[class='async_elem']>a[data-sigil='ajaxify']").click()
    import time 
    try:
        comment_click = driver.find_element_by_css_selector("div[class='async_elem']>a[data-sigil='ajaxify']")
        while comment_click:
            try:
                driver.find_element_by_css_selector("div[class='async_elem']>a[data-sigil='ajaxify']").click()
                time.sleep(5)
            except ValueError:
                print('sorry, invalid input')
    except NoSuchElementException:
        print('There are no load more buttons')

    try:
        comment_click = driver.find_element_by_css_selector("div[data-sigil='replies-see-more']")
        while comment_click:
            try:
                driver.find_element_by_css_selector("div[data-sigil='replies-see-more']").click()
                time.sleep(5)
            except ValueError:
                print('sorry, invalid input')
    except NoSuchElementException:
        print('There are no see more comments')

    try:
        comment_click = driver.find_element_by_css_selector("div[data-sigil='replies-see-more']")
        while comment_click:
            try:
                driver.find_element_by_css_selector("div[data-sigil='replies-see-more']").click()
                time.sleep(5)
            except ValueError:
                print('sorry, invalid input')
    except NoSuchElementException:
        print('There are no see more comments')




    try:
        comment_click = driver.find_element_by_css_selector("div[data-sigil='replies-see-more']>a[data-sigil='ajaxify']")
        while comment_click:
            try:
                driver.find_element_by_css_selector("div[data-sigil='replies-see-more']>a[data-sigil='ajaxify']").click()
                time.sleep(5)
            except ValueError:
                print('sorry, invalid input')
    except NoSuchElementException:
        print('There are no load more buttons')


    try:
        comment_click = driver.find_element_by_css_selector("div[data-sigil='replies-see-prev']")
        while comment_click:
            try:
                driver.find_element_by_css_selector("div[data-sigil='replies-see-prev']").click()
                time.sleep(5)
            except ValueError:
                print('sorry, invalid input')
    except NoSuchElementException:
        print('There are no see more comments')

    try:
        comment_click = driver.find_element_by_css_selector("div[data-sigil='replies-see-next']")
        while comment_click:
            try:
                driver.find_element_by_css_selector("div[data-sigil='replies-see-next']").click()
                time.sleep(5)
            except ValueError:
                print('sorry, invalid input')
    except NoSuchElementException:
        print('There are no see more comments')


    comments = driver.find_elements_by_css_selector("div[data-sigil='comment-body']")

    commentstxt = {}
    i = 0

    commentstxt = {}
    i = 0
    for comment in comments:
        commentstxt[i] = comment.text
        i += 1


    test_path = os.path.join(APP_ROOT, 'models\\' + current_user.username)
    dirss = os.listdir( test_path )
    folders = []

    for dirs in dirss:
        if os.path.isdir(test_path + '\\' + dirs):
            folders.append(dirs)
        
    new_df = pd.DataFrame.from_dict(commentstxt, orient='index', columns=['Comment'])    
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import CountVectorizer
    import pickle
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    lr = LogisticRegression()
    from sklearn.externals import joblib 
    from sklearn.naive_bayes import MultinomialNB
    ### Load and Test
    ytb_model = open("models/dareja/tg_cvec.pkl", "rb")
    tg_cvec = joblib.load(ytb_model)
    ytb_model = open("models/dareja/sentiment_fit.pkl", "rb")
    sentiment_fit = joblib.load(ytb_model)
    tg_pipeline = Pipeline([
        ('vectorizer', tg_cvec),
        ('classifier', lr)
    ])

    new_df_comments = commentstxt
    s2 = pd.Series(new_df_comments)
    my_prediction = sentiment_fit.predict(s2)

    test_preds = []
    for i in range(len(my_prediction)):
        if my_prediction[i] == 4: 
            my_prediction_cat = 'Super'
        else:
            my_prediction_cat = 'Bad'
        test_preds.append(my_prediction_cat)
    new_df['Polarity'] = pd.DataFrame(test_preds)
    new_df_length = len(new_df)

    target = os.path.join(APP_ROOT, 'seaal/')
    print(target)

    if not os.path.isdir(target):
        os.mkdir(target)

    dateTimeObj = datetime.now()
    timeObj = dateTimeObj.time()
    timeStr = timeObj.strftime("%H_%M_%S")

    new_df.to_excel(target + "/seaal_comments"+ timeStr + ".xlsx")


    return render_template('fb_link_results.html', title = title,
        links = links, commentstxt = commentstxt, new_df = new_df )









# scraping web facebook comments
@analyse.route('/fb_link_web', methods=['POST', 'GET'])
def fb_link_web():
    if not current_user.is_authenticated:
        flash('You should be logged in to acced this page', 'info')
        next_page =  request.endpoint
        return redirect(url_for('users.login', next = next_page))
    if request.method != "POST":
        flash("You don't have permission accessing this page", 'info')
        return redirect(url_for('main.dashboard'))
    global new_df
    title = "Facebook Scrapper By Keyboard"
    if request.method == "POST":
        fb_link = request.form['fb_link']
    from selenium import webdriver
    from selenium.common.exceptions import NoSuchElementException
    
    driver = webdriver.Chrome(executable_path="C:/Users/dell/Downloads/chromedriver_win32_4/chromedriver")

    driver.get("https://mobile.facebook.com")
    
    #find the username field and enter the email example@yahoo.com.
    username = driver.find_elements_by_css_selector("input[name=email]")
    username[0].send_keys('666119777')#find the password field and enter the password password.
    password = driver.find_elements_by_css_selector("input[name=pass]")
    password[0].send_keys('paulo10COELO')#find the login button and click it.
    loginButton = driver.find_elements_by_css_selector("button[name=login]")
    loginButton[0].click()

    import time
    time.sleep(5)

    driver.get(fb_link)
    time.sleep(5)

    titles = driver.find_elements_by_css_selector("._2b05 > a")

    links = {}
    i = 0
    for title in titles:
      print(title.get_attribute('href'))
      links[i] = title.get_attribute('href')
      i += 1

    #driver.find_element_by_css_selector("div[class='async_elem']>a[data-sigil='ajaxify']").click()
    import time 
    try:
        comment_click = driver.find_element_by_css_selector("div[class='async_elem']>a[data-sigil='ajaxify']")
        while comment_click:
            try:
                driver.find_element_by_css_selector("div[class='async_elem']>a[data-sigil='ajaxify']").click()
                time.sleep(5)
            except ValueError:
                print('sorry, invalid input')
    except NoSuchElementException:
        print('There are no load more buttons')

    try:
        comment_click = driver.find_element_by_css_selector("div[data-sigil='replies-see-more']")
        while comment_click:
            try:
                driver.find_element_by_css_selector("div[data-sigil='replies-see-more']").click()
                time.sleep(5)
            except ValueError:
                print('sorry, invalid input')
    except NoSuchElementException:
        print('There are no see more comments')

    try:
        comment_click = driver.find_element_by_css_selector("div[data-sigil='replies-see-more']")
        while comment_click:
            try:
                driver.find_element_by_css_selector("div[data-sigil='replies-see-more']").click()
                time.sleep(5)
            except ValueError:
                print('sorry, invalid input')
    except NoSuchElementException:
        print('There are no see more comments')




    try:
        comment_click = driver.find_element_by_css_selector("div[data-sigil='replies-see-more']>a[data-sigil='ajaxify']")
        while comment_click:
            try:
                driver.find_element_by_css_selector("div[data-sigil='replies-see-more']>a[data-sigil='ajaxify']").click()
                time.sleep(5)
            except ValueError:
                print('sorry, invalid input')
    except NoSuchElementException:
        print('There are no load more buttons')


    try:
        comment_click = driver.find_element_by_css_selector("div[data-sigil='replies-see-prev']")
        while comment_click:
            try:
                driver.find_element_by_css_selector("div[data-sigil='replies-see-prev']").click()
                time.sleep(5)
            except ValueError:
                print('sorry, invalid input')
    except NoSuchElementException:
        print('There are no see more comments')

    try:
        comment_click = driver.find_element_by_css_selector("div[data-sigil='replies-see-next']")
        while comment_click:
            try:
                driver.find_element_by_css_selector("div[data-sigil='replies-see-next']").click()
                time.sleep(5)
            except ValueError:
                print('sorry, invalid input')
    except NoSuchElementException:
        print('There are no see more comments')


    comments = driver.find_elements_by_css_selector("div[data-sigil='comment-body']")

    commentstxt = {}
    i = 0

    commentstxt = {}
    i = 0
    for comment in comments:
        commentstxt[i] = comment.text
        i += 1


    test_path = os.path.join(APP_ROOT, 'models\\' + current_user.username)
    dirss = os.listdir( test_path )
    folders = []

    for dirs in dirss:
        if os.path.isdir(test_path + '\\' + dirs):
            folders.append(dirs)
        
    new_df = pd.DataFrame.from_dict(commentstxt, orient='index', columns=['Comment'])    
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import CountVectorizer
    import pickle
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    lr = LogisticRegression()
    from sklearn.externals import joblib 
    from sklearn.naive_bayes import MultinomialNB
    ### Load and Test
    ytb_model = open("models/dareja/tg_cvec.pkl", "rb")
    tg_cvec = joblib.load(ytb_model)
    ytb_model = open("models/dareja/sentiment_fit.pkl", "rb")
    sentiment_fit = joblib.load(ytb_model)
    tg_pipeline = Pipeline([
        ('vectorizer', tg_cvec),
        ('classifier', lr)
    ])

    new_df_comments = commentstxt
    s2 = pd.Series(new_df_comments)
    my_prediction = sentiment_fit.predict(s2)

    test_preds = []
    for i in range(len(my_prediction)):
        if my_prediction[i] == 4: 
            my_prediction_cat = 'Super'
        elif my_prediction[i] == 2:
            my_prediction_cat = 'Good'
        else:
            my_prediction_cat = 'Bad'
        test_preds.append(my_prediction_cat)
    new_df['Polarity'] = pd.DataFrame(test_preds)
    new_df_length = len(new_df)

    target = os.path.join(APP_ROOT, 'seaal/')
    print(target)

    if not os.path.isdir(target):
        os.mkdir(target)

    dateTimeObj = datetime.now()
    timeObj = dateTimeObj.time()
    timeStr = timeObj.strftime("%H_%M_%S")

    new_df.to_excel(target + "/seaal_comments"+ timeStr + ".xlsx")


    return render_template('fb_link_results.html', title = title,
        links = links, commentstxt = commentstxt, new_df = new_df )





































































































































@analyse.route('/fb_scraping_results', methods=['POST', 'GET'])
def fb_scraping_results():
    # redirect user if not logged in 
    if not current_user.is_authenticated:
        flash('You should be logged in to acced this page', 'info')
        next_page =  request.endpoint
        return redirect(url_for('users.login', next = next_page))
    if request.method != "POST":
        flash("You don't have permission accessing this page", 'info')
        return redirect(url_for('main.dashboard'))
    # initiate global variables for excel download
    global new_df, results_df
    title = "Facebook Scraping by Excel file (links)"

    # import important libraries for this page
    from selenium import webdriver
    from selenium.common.exceptions import NoSuchElementException
    from selenium.common.exceptions import ElementNotInteractableException
    from selenium.common.exceptions import ElementClickInterceptedException
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.common.by import By
    import re
    from nltk.tokenize import sent_tokenize, word_tokenize
    from collections import Counter
    import time

    import MySQLdb
    import pandas as pd
    import re
    import mysql.connector
    import datetime
    conn = MySQLdb.connect(host='localhost',
                           user='root',
                           passwd='',
                           db='new_db',
                           use_unicode=True, 
                           charset='utf8mb4')

    cursor = conn.cursor()
    target = os.path.join(APP_ROOT, 'fb_links/')

    if not os.path.isdir(target):
        os.mkdir(target)
    # save and read the excel file te get links to be scraped 
    for file in request.files.getlist("file"):
        filename = file.filename
        destination = "/".join([target, filename])
        file.save(destination)
        df2 = pd.read_excel(os.path.join(APP_ROOT, 'fb_links/'+filename))
        df_data2 = df2[["lien", "numArticle", "pages"]]
        df_urls = df_data2['lien']
        df_names = df_data2['numArticle']
        df_page = df_data2['pages']
        model = request.form['model']

    options = webdriver.ChromeOptions()
    options.add_argument("--disable-dev-shm-usage")
    options.set_headless()
    options.add_argument("no-sandbox")
    options.add_argument("--disable-gpu")
    driver = webdriver.Chrome(executable_path="C:/Program Files (x86)/chromedriver")
    
    # execute selenium and connect to a facebook profile 
    #driver = webdriver.Chrome(executable_path="C:/Users/dell/Downloads/Compressed/chromedriver_win32_4/chromedriver")

    driver.get("https://mobile.facebook.com")
    
    #find the username field and enter the email example@yahoo.com.
    username = driver.find_elements_by_css_selector("input[name=email]")
    username[0].send_keys('666119777')#find the password field and enter the password password.
    password = driver.find_elements_by_css_selector("input[name=pass]")
    password[0].send_keys('paulo10COELO')#find the login button and click it.
    try:    
        loginButton = driver.find_elements_by_css_selector("button[name=login]")
        if loginButton:
            WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.CSS_SELECTOR, "button[name=login]")))
            loginButton[0].click()
    except NoSuchElementException:
        print('No Button tag for login')

    try:    
        loginButton = driver.find_element_by_css_selector("input[name=login]")
        loginButton.click()
    except NoSuchElementException:
        print('No Input tag for login')

    
    time.sleep(5)

    # get the current date and time (to be used in naming the file)
    dateTimeObj = pd.to_datetime('today')
    timeStr = dateTimeObj.strftime("%Y-%m-%d-%H_%M_%S")

    target = os.path.join(APP_ROOT, 'scraping\\' + timeStr + '/')
    print(target)

    if not os.path.isdir(target):
        os.mkdir(target)

    commentstxt = {}
    linkstxt = {}
    k = 0
    links = {}
    i = 0
    num_index = 0
    df = pd.DataFrame()

    # scraping facebook comments and doing sentiment analysis on them 
    for fb_link in df_urls:
        if validators.url(fb_link) != True or 'mobile.facebook.com' in fb_link:
            continue
        fb_link = fb_link.replace("www.", "mobile.")
        fb_link = fb_link.replace("web.", "mobile.")
        fb_link = fb_link.replace("upload.", "mobile.")
        driver.get(fb_link)
        time.sleep(3)
        if fb_link != driver.current_url:
            continue
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        
        #numArticleIndex = df2[df_urls == fb_link].index.item()
        #LinkIndex = df_names[numArticleIndex]
        print(num_index)
        LinkIndex = df_names[num_index]
        LinkPage = df_page[num_index]
        num_index = num_index + 1


        # this code enters the comments, load all comments, and then load all replies and then load all view more (where much is written in each comment)
        titles = driver.find_elements_by_css_selector("._2b05 > a")
        for title in titles:
          print(title.get_attribute('href'))
          links[i] = title.get_attribute('href')
          i += 1

        # This code works for videos
        import time 
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        try:
            comment_click = driver.find_element_by_css_selector("textarea#composerInput")
            if comment_click:
                try:
                    driver.find_element_by_css_selector("textarea#composerInput").click()
                    time.sleep(1)
                except ValueError:
                    print('sorry, invalid input')
        except NoSuchElementException:
            print('There are no comment click')

        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        try:
            comment_click = driver.find_element_by_css_selector("a[class='_15kq _77li']")
            if comment_click:
                try:
                    time.sleep(1)
                    driver.find_element_by_css_selector("a[class='_15kq _77li']").click()
                    time.sleep(1)
                except ElementClickInterceptedException:
                    print('sorry, invalid input')
        except NoSuchElementException:
            print('There are no comments click form')

        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        try:
            comment_click = driver.find_element_by_css_selector("#composerInput']")
            if comment_click:
                try:
                    driver.find_element_by_css_selector("#composerInput").click()
                    time.sleep(1)
                except ValueError:
                    print('sorry, invalid input')
        except NoSuchElementException:
            print('There are no load more buttons')

        # ends of code that works for videos


        import time 
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        try:
            comment_click = driver.find_element_by_css_selector("span[data-sigil='comments-token']")
            while comment_click:
                try:
                    driver.find_element_by_css_selector("span[data-sigil='comments-token']").click()
                    time.sleep(5)
                except ValueError:
                    print('sorry, invalid input')
        except NoSuchElementException:
            print('There are no load more buttons')


        #driver.find_element_by_css_selector("div[class='async_elem']>a[data-sigil='ajaxify']").click()
        import time 
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        try:
            comment_click = driver.find_element_by_css_selector("span[data-sigil='comments-token']")
            while comment_click:
                try:
                    driver.find_element_by_css_selector("span[data-sigil='comments-token']").click()
                    time.sleep(5)
                except ValueError:
                    print('sorry, invalid input')
        except NoSuchElementException:
            print('There are no load more buttons')

        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        try:
            comment_click = driver.find_element_by_css_selector("div[class='async_elem']>a[data-sigil='ajaxify']")
            while comment_click:
                try:
                    driver.find_element_by_css_selector("div[class='async_elem']>a[data-sigil='ajaxify']").click()
                    time.sleep(5)
                except ValueError:
                    print('sorry, invalid input')
        except NoSuchElementException:
            print('There are no load more buttons')

        try:
            comment_click = driver.find_element_by_css_selector("div[data-sigil='replies-see-more'] a")
            while comment_click:
                try:
                    driver.find_element_by_css_selector("div[data-sigil='replies-see-more'] a").click()
                    time.sleep(5)
                except ValueError:
                    print('sorry, invalid input')
        except NoSuchElementException:
            print('There are no see more comments')


        try:
            comment_click = driver.find_element_by_css_selector("div[data-sigil='replies-see-more']>a[data-sigil='ajaxify']")
            while comment_click:
                try:
                    driver.find_element_by_css_selector("div[data-sigil='replies-see-more']>a[data-sigil='ajaxify']").click()
                    time.sleep(5)
                except ValueError:
                    print('sorry, invalid input')
        except NoSuchElementException:
            print('There are no load more buttons')


        try:
            comment_click = driver.find_element_by_css_selector("div[data-sigil='replies-see-prev']")
            while comment_click:
                try:
                    driver.find_element_by_css_selector("div[data-sigil='replies-see-prev']").click()
                    time.sleep(5)
                except ValueError:
                    print('sorry, invalid input')
        except NoSuchElementException:
            print('There are no see more comments')

        try:
            comment_click = driver.find_element_by_css_selector("div[data-sigil='replies-see-next']")
            while comment_click:
                try:
                    driver.find_element_by_css_selector("div[data-sigil='replies-see-next']").click()
                    time.sleep(5)
                except ValueError:
                    print('sorry, invalid input')
        except NoSuchElementException:
            print('There are no see more comments')

        try: 
            comments = driver.find_elements_by_css_selector("div[data-sigil='comment-body']")
            b = 0
            commentslink = {}
            for comment in comments:
                commentslink[b] = comment.text
                b += 1
        except NoSuchElementException:
            print('There are no comments for this post')

        # generate a dataframe from the scraped comments and get the Sentiment Analysis .
        if bool(commentslink):
            new_df2 = pd.DataFrame.from_dict(commentslink, orient='index', columns=['comment'])    
            stopwords = request.form['stopwords']
            my_link_code = []
            my_page_code = []
            
            
            # get the sentiment analysis of the scraped comments
            new_df, results_percents, results_df_pos, results_df_neu, results_df_neg, my_prediction = get_sentiment_analysis2(table = new_df2, model = model)

            for i in range(len(my_prediction)):
                my_link_code.append(LinkIndex)
                my_page_code.append(LinkPage)

            # add the post code and the page to the dataframe
            new_df['Code'] = pd.DataFrame(my_link_code)
            new_df['Pages'] = pd.DataFrame(my_page_code)
            

            comments_len = len(commentslink)
            # save the generated dataframe to the pecified folder
            new_df.to_excel(target + "/"+ str(num_index)  +"complete_"+ str(LinkIndex) + "_" + str(comments_len) + ".xlsx")
            df = df.append(new_df)
            # save the scraped data to a databse
            
            #for i in range(0, len(new_df)):
            #    cursor.execute("""INSERT INTO all_comments (comment, polarity, model, page, post_code, date, n_followers ) 
            #        VALUES (%s, %s, %s, %s, %s, %s, %s)""", 
            #        (new_df.Comment[i], new_df.Polarity[i], model, new_df.Pages[i] , int(new_df.Code[i]), timeStr, 0 ) )
            #    conn.commit()
            #    print(i)

    # saving all data on one file
    df.to_excel(target + "/all.xlsx")

    new_clean = df
    new_clean['comment'] = df['Comment']
    model = request.form['model']


    results_df, results_percents, results_df_pos, results_df_neu, results_df_neg, my_prediction = get_sentiment_analysis2(table = new_clean, model = model)



    
    # use the geerated data for word counts and wordclouds
    neg_results = results_df[results_df.Polarity == 'Bad']
    for_wordcloud_neg = ' '.join(map(str, neg_results['Comment']))
    all_words = word_tokenize(for_wordcloud_neg)
    counts = Counter(all_words)
    neg_counts = pd.DataFrame(list(counts.items()), columns = ['word', 'freq'] )  
    neg_counts = neg_counts.sort_values(by='freq', ascending=False).iloc[:20]

    neu_results = results_df[results_df.Polarity == 'Good']
    for_wordcloud_neu = ' '.join(map(str, neu_results['Comment']))
    all_words = word_tokenize(for_wordcloud_neu)
    counts = Counter(all_words)
    neu_counts = pd.DataFrame(list(counts.items()), columns = ['word', 'freq'] )     
    neu_counts = neu_counts.sort_values(by='freq', ascending=False).iloc[:20]

    pos_results = results_df[results_df.Polarity == 'Super']
    for_wordcloud_pos = ' '.join(map(str, pos_results['Comment']))
    all_words = word_tokenize(for_wordcloud_pos)
    counts = Counter(all_words)
    pos_counts = pd.DataFrame(list(counts.items()), columns = ['word', 'freq'] )  
    pos_counts = pos_counts.sort_values(by='freq', ascending=False).iloc[:20]

    
    # get the topics (ass defined in the function get_telecom_topics)
    telecom_counts, all_cats = get_telecom_topics(results_df = results_df)


    all_cats_str = []
    for elem in all_cats:
        all_cats_str.append(', '.join(elem))

    cats_length = len(all_cats_str)

    type_excel = type(results_df)

    # rendering the template with the elements we scraped
    return render_template('fb_scraping_results.html', title = title,
        links = links, commentstxt = commentstxt, new_df = new_df, 
        results_df = results_df,
        for_wordcloud_neg = for_wordcloud_neg,
        for_wordcloud_pos = for_wordcloud_pos, 
        for_wordcloud_neu = for_wordcloud_neu, 
        neg_counts = neg_counts,
        neu_counts = neu_counts, 
        pos_counts = pos_counts,
        pos_percentage = results_percents['pos_percentage'], 
        neu_percentage = results_percents['neu_percentage'], 
        neg_percentage = results_percents['neg_percentage'], 
        cats_length = cats_length, 
        telecom_counts = telecom_counts,
        all_cats_str = all_cats_str, 
        neg_count_internet = telecom_counts['neg_count_internet'],
        neg_count_reseau = telecom_counts['neg_count_reseau'],
        neg_count_offre = telecom_counts['neg_count_offre'],
        neg_count_appelle = telecom_counts['neg_count_appelle'],
        neg_count_other = telecom_counts['neg_count_other'],
        neu_count_internet = telecom_counts['neu_count_internet'],
        neu_count_reseau = telecom_counts['neu_count_reseau'],
        neu_count_offre = telecom_counts['neu_count_offre'],
        neu_count_appelle = telecom_counts['neu_count_appelle'],
        neu_count_other = telecom_counts['neu_count_other'],
        pos_count_internet = telecom_counts['pos_count_internet'],
        pos_count_reseau = telecom_counts['pos_count_reseau'],
        pos_count_offre = telecom_counts['pos_count_offre'],
        pos_count_appelle = telecom_counts['pos_count_appelle'],
        pos_count_other = telecom_counts['pos_count_other'],
        count_internet = telecom_counts['count_internet'],
        count_reseau = telecom_counts['count_reseau'],
        count_offre = telecom_counts['count_offre'],
        count_appelle = telecom_counts['count_appelle'],
        count_other = telecom_counts['count_other'],
        results_df_pos = results_df_pos  , 
        new_df2 = new_clean
         )













































@analyse.route('/fb_scrapper', methods=['POST', 'GET'])
def fb_scrapper():
    if not current_user.is_authenticated:
        flash('You should be logged in to acced this page', 'info')
        next_page =  request.endpoint
        return redirect(url_for('users.login', next = next_page))
    global new_df
    title = "Facebook Scrapper"
    if request.method == "POST":
        fb_link = request.form['fb_link']
    from selenium import webdriver
    from selenium.common.exceptions import NoSuchElementException
    
    driver = webdriver.Chrome(executable_path="C:/Program Files (x86)/chromedriver")

    driver.get(fb_link)
    
    #find the username field and enter the email example@yahoo.com.
    username = driver.find_elements_by_css_selector("input[name=email]")
    username[0].send_keys('666119777')#find the password field and enter the password password.
    password = driver.find_elements_by_css_selector("input[name=pass]")
    password[0].send_keys('paulo10COELO')#find the login button and click it.
    loginButton = driver.find_elements_by_css_selector("button[name=login]")
    loginButton[0].click()

    import time
    time.sleep(5)

    titles = driver.find_elements_by_css_selector("._2b05 > a")

    links = {}
    i = 0
    for title in titles:
      print(title.get_attribute('href'))
      links[i] = title.get_attribute('href')
      i += 1


    #driver.find_element_by_css_selector("div[class='async_elem']>a[data-sigil='ajaxify']").click()
    import time 
    try:
        comment_click = driver.find_element_by_css_selector("div[class='async_elem']>a[data-sigil='ajaxify']")
        while comment_click:
            try:
                driver.find_element_by_css_selector("div[class='async_elem']>a[data-sigil='ajaxify']").click()
                time.sleep(5)
                break
            except ValueError:
                print('sorry, invalid input')
    except NoSuchElementException:
        print("There are no load more button")


    try:
        comment_click = driver.find_element_by_css_selector("div[data-sigil='replies-see-more']")
        while comment_click:
            try:
                driver.find_element_by_css_selector("div[data-sigil='replies-see-more']").click()
                time.sleep(5)
                break
            except ValueError:
                print('sorry, invalid input')
    except NoSuchElementException:
        print('There are no see more comments')




    try:
        comment_click = driver.find_element_by_css_selector("div[data-sigil='replies-see-more']>a[data-sigil='ajaxify']")
        while comment_click:
            try:
                driver.find_element_by_css_selector("div[data-sigil='replies-see-more']>a[data-sigil='ajaxify']").click()
                time.sleep(5)
                break
            except ValueError:
                print('sorry, invalid input')
    except NoSuchElementException:
        print('There are no load more buttons')


    try:
        comment_click = driver.find_element_by_css_selector("div[data-sigil='replies-see-prev']")
        while comment_click:
            try:
                driver.find_element_by_css_selector("div[data-sigil='replies-see-prev']").click()
                time.sleep(5)
                break
            except ValueError:
                print('sorry, invalid input')
    except NoSuchElementException:
        print('There are no see more comments')

    try:
        comment_click = driver.find_element_by_css_selector("div[data-sigil='replies-see-next']")
        while comment_click:
            try:
                driver.find_element_by_css_selector("div[data-sigil='replies-see-next']").click()
                time.sleep(5)
                break
            except ValueError:
                print('sorry, invalid input')
    except NoSuchElementException:
        print('There are no see more comments')

    comments = driver.find_elements_by_css_selector("div[data-sigil='comment-body']")

    commentstxt = {}
    i = 0

    commentstxt = {}
    i = 0
    for comment in comments:
        commentstxt[i] = comment.text
        i += 1


    test_path = os.path.join(APP_ROOT, 'models\\' + current_user.username)
    dirss = os.listdir( test_path )
    folders = []

    for dirs in dirss:
        if os.path.isdir(test_path + '\\' + dirs):
            folders.append(dirs)
        
    new_df = pd.DataFrame.from_dict(commentstxt, orient='index', columns=['Comment'])    
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import CountVectorizer
    import pickle
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    lr = LogisticRegression()
    from sklearn.externals import joblib 
    from sklearn.naive_bayes import MultinomialNB
    ### Load and Test
    ytb_model = open("models/dareja/tg_cvec.pkl", "rb")
    tg_cvec = joblib.load(ytb_model)
    ytb_model = open("models/dareja/sentiment_fit.pkl", "rb")
    sentiment_fit = joblib.load(ytb_model)
    tg_pipeline = Pipeline([
        ('vectorizer', tg_cvec),
        ('classifier', lr)
    ])

    new_df_comments = commentstxt
    s2 = pd.Series(new_df_comments)
    my_prediction = sentiment_fit.predict(s2)

    test_preds = []
    for i in range(len(my_prediction)):
        if my_prediction[i] == 4: 
            my_prediction_cat = 'Super'
        elif my_prediction[i] == 2:
            my_prediction_cat = 'Good'
        else:
            my_prediction_cat = 'Bad'
        test_preds.append(my_prediction_cat)
    new_df['Polarity'] = pd.DataFrame(test_preds)
    new_df_length = len(new_df)




    return render_template('fb_scrapper.html', title = title,
        links = links, commentstxt = commentstxt, new_df = new_df )



@analyse.route('/results', methods = ['POST'])
def results():
    if not current_user.is_authenticated:
        flash('You should be logged in to acced this page', 'info')
        next_page =  request.endpoint
        return redirect(url_for('users.login', next = next_page))
    #df = pd.read_csv('YoutubeSpamMergedData.csv')
    target = os.path.join(APP_ROOT, 'trainingcsv/')
    print(target)


    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist("file"):
        print(file)
        filename = file.filename
        destination = "/".join([target, filename])
        print(destination)
        file.save(destination)
        df2 = pd.read_excel(os.path.join(APP_ROOT, 'trainingcsv/'+filename))
        df_data2 = df2[["CONTENT", "CLASS"]]
        df_x2 = df_data2['CONTENT']
        df_y2 = df_data2.CLASS


    df_data = df2[["CONTENT", "CLASS"]]
    # Features and Labels 
    df_x = df_data['CONTENT']
    df_y = df_data.CLASS
    # eXTRACT FEATURE WITH CountVectorizer
    corpus = df_x 
    cv = CountVectorizer()
    X = cv.fit_transform(corpus) # Fit the data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, df_y, test_size = 0.33, random_state = 42)
    # Naive Bayes Classifier 
    from sklearn.naive_bayes import MultinomialNB
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    clf.score(X_test, y_test)
    # Alternative Usage of saved model 
    # ytb_model = open("naivebayes_spam_model.pkl", "rb")
    # clf = joblib.load(ytb_model)

    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    conmat = np.array(confusion_matrix(y_test, y_pred, labels=[0,1]))

    confusion = pd.DataFrame(conmat, index=['negative', 'positive'],
                         columns=['predicted_negative','predicted_positive'])

    true_negative = confusion["predicted_negative"][0]
    false_negative = confusion["predicted_negative"][1]

    false_positive = confusion["predicted_positive"][0]
    true_positive = confusion["predicted_positive"][1]

    count_pos = len(y_pred[y_pred == 1])
    count_neg = len(y_pred[y_pred == 0])
    count_all = len(y_pred)


    #if request.method == 'POST':
    #    comment = request.form['comment']
    #    data = [comment]
    #    vect = cv.transform(data).toarray()
    #    my_prediction = clf.predict(vect)
    #else:
    #    comment = 'Great content, just continue publishing others'
    #    data = [comment]
    #    vect = cv.transform(data).toarray()
    #    my_prediction = clf.predict(vect)

    return render_template('results.html', 
        #prediction = my_prediction, 
        count_all = count_all, 
        count_neg = count_neg,
        count_pos = count_pos, 
        confusion = confusion,
        accuracy = accuracy, 
        true_negative = true_negative, 
        false_negative = false_negative, 
        false_positive = false_positive, 
        true_positive = true_positive, 
        df2 = df2, 
        df_y2 = df_y2, 
        df_data2 = df_data2
        
        )




@analyse.route('/results_test', methods = ['POST', 'GET'])
def results_test():
    if not current_user.is_authenticated:
        flash('You should be logged in to acced this page', 'info')
        next_page =  request.endpoint
        return redirect(url_for('users.login', next = next_page))
    global new_df
    #df = pd.read_excel(r'YoutubeSpamMergedData.xlsx')
    target = os.path.join(APP_ROOT, 'testxlsx/')
    print(target)

    dirss = os.listdir(APP_ROOT)
    folders = []
    for dirs in dirss:
        if os.path.isdir(os.path.join(os.getcwd(),  dirs )):
            folders.append(dirs)


    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist("file"):
        print(file)
        filename = file.filename
        destination = "/".join([target, filename])
        print(destination)
        file.save(destination)
        df2 = pd.read_excel(os.path.join(APP_ROOT, 'testxlsx/'+filename))
        df_data2 = df2[["CONTENT", "CLASS"]]
        df_x2 = df_data2['CONTENT']
        df_y2 = df_data2.CLASS


    df_data = df2[["CONTENT", "CLASS"]]
    # Features and Labels 
    df_x = df_data['CONTENT']
    df_y = df_data.CLASS
    # eXTRACT FEATURE WITH CountVectorizer
    corpus = df_x 
    cv = CountVectorizer()
    X = cv.fit_transform(corpus) # Fit the data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, df_y, test_size = 0.33, random_state = 42)
    # Naive Bayes Classifier 
    from sklearn.naive_bayes import MultinomialNB
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    clf.score(X_test, y_test)
    # Alternative Usage of saved model 
    # ytb_model = open("naivebayes_spam_model.pkl", "rb")
    # clf = joblib.load(ytb_model)

    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    conmat = np.array(confusion_matrix(y_test, y_pred, labels=[0,1]))

    confusion = pd.DataFrame(conmat, index=['negative', 'positive'],
                         columns=['predicted_negative','predicted_positive'])

    true_negative = confusion["predicted_negative"][0]
    false_negative = confusion["predicted_negative"][1]

    false_positive = confusion["predicted_positive"][0]
    true_positive = confusion["predicted_positive"][1]

    count_pos = len(y_pred[y_pred == 1])
    count_neg = len(y_pred[y_pred == 0])
    count_all = len(y_pred)


    #if request.method == 'POST':
    #    comment = request.form['comment']
    #    data = [comment]
    #    vect = cv.transform(data).toarray()
    #    my_prediction = clf.predict(vect)
    #else:
    #    comment = 'Great content, just continue publishing others'
    #    data = [comment]
    #    vect = cv.transform(data).toarray()
    #    my_prediction = clf.predict(vect)

    df_test = df2[["CONTENT", "CLASS"]]
    

    test_data = df2[["CONTENT", "CLASS"]]
    # Features and Labels 
    test_x = df_test['CONTENT']
    #test_y = df_data.CLASS
    my_comments = []
    test_preds = []
    for i in test_x:
        vect = cv.transform([i]).toarray()
        my_prediction_test = clf.predict(vect)
        if my_prediction_test[0] == 1: 
            my_prediction_cat = 'Super'
        else:
            my_prediction_cat = 'Bad'
        test_preds.append(my_prediction_cat)
        my_comments.append(i)
        result_list = pd.DataFrame(
            {'Commentaire': my_comments,
             'Sentiment': test_preds
            })
        n_pos_neg = result_list['Sentiment'].value_counts()
        n_neg = len(result_list[result_list['Sentiment'] == 'Bad'])
        n_pos = len(result_list[result_list['Sentiment'] == 'Super'])
        n_total = n_pos + n_neg

        new_df = result_list


    return render_template('results_test.html', 
        #prediction = my_prediction, 
        count_all = count_all, 
        count_neg = count_neg,
        count_pos = count_pos, 
        confusion = confusion,
        accuracy = accuracy, 
        true_negative = true_negative, 
        false_negative = false_negative, 
        false_positive = false_positive, 
        true_positive = true_positive, 
        X_test = X_test, 
        y_test = y_test,
        my_comments = my_comments,
        test_preds = test_preds, 
        result_list =result_list, 
        df_test = df_test, 
        n_pos_neg = n_pos_neg, 
        n_neg = n_neg, 
        n_pos = n_pos, 
        n_total = n_total
        #, 
        #df2 = df2, 
        #df_y2 = df_y2, 
        ,df_data2 = df_data2, 
        new_df = new_df,
        folders = folders
        )





































# get sentiment analysis for an excel file 
@analyse.route('/results_analyse_excel', methods = ['POST', 'GET'])
def results_analyse_excel():
    if not current_user.is_authenticated:
        flash('You should be logged in to acced this page', 'info')
        next_page =  request.endpoint
        return redirect(url_for('users.login', next = next_page))
    if request.method != "POST":
        flash("You don't have permission accessing this page", 'info')
        return redirect(url_for('main.dashboard'))
    global new_df, results_df
    import pandas as pd
    #df = pd.read_excel(r'YoutubeSpamMergedData.xlsx')
    title = "Resultats d'analyse des sentiments d'un fichier excel"
    
    test_path = os.path.join(APP_ROOT, 'models\\' + current_user.username)


    if request.method == "POST":
        secteur = request.form['secteur']
        model = request.form['model']
        stopwords = request.form['stopwords']
        all_comments, df_x, df2, column_name = save_excel()

        #new_clean = clean_text(df_content = df2, column_name = column_name, stopword = stopwords)
        #new_clean = clean_text(df_content = df2, column_name = column_name, stopword = stopwords)
        new_clean = pd.DataFrame()
        new_clean['comment'] = df_x
        #new_clean['comment'] = df_x
        #all_clean = []
        import re
        #clean = re.compile('<.*?>')
        #for i in range(0, len(all_comments)):
        #    all_comments[i] = re.sub(clean, ' ', all_comments[i])
        #    all_comments[i] = all_comments[i].replace('\n',' ')
        #    all_comments[i] = all_comments[i].replace('\\n',' ')
        #    all_comments[i] = all_comments[i].replace('\t',' ')
        #    all_comments[i] = all_comments[i].replace('\r',' ')
        #    all_comments[i] = all_comments[i].replace('\'',' ')
        stop_dareja = ['we', 'and', 'w']

        new_clean = remove_stopwords(df2, 'Comment', stopwords)
        results_df, results_percents, results_df_pos, results_df_neu, results_df_neg, my_prediction = get_sentiment_analysis2(table = new_clean, model = model)



        from nltk.tokenize import sent_tokenize, word_tokenize
        from collections import Counter

        neg_results = results_df[results_df.Polarity == 'Bad']
        for_wordcloud_neg = ' '.join(map(str, neg_results['Comment']))
        all_words = word_tokenize(for_wordcloud_neg)
        counts = Counter(all_words)
        neg_counts = pd.DataFrame(list(counts.items()), columns = ['word', 'freq'] )  
        neg_counts = neg_counts.sort_values(by='freq', ascending=False).iloc[:20]

        neu_results = results_df[results_df.Polarity == 'Good']
        for_wordcloud_neu = ' '.join(map(str, neu_results['Comment']))
        all_words = word_tokenize(for_wordcloud_neu)
        counts = Counter(all_words)
        neu_counts = pd.DataFrame(list(counts.items()), columns = ['word', 'freq'] )     
        neu_counts = neu_counts.sort_values(by='freq', ascending=False).iloc[:20]

        pos_results = results_df[results_df.Polarity == 'Super']
        for_wordcloud_pos = ' '.join(map(str, pos_results['Comment']))
        all_words = word_tokenize(for_wordcloud_pos)
        counts = Counter(all_words)
        pos_counts = pd.DataFrame(list(counts.items()), columns = ['word', 'freq'] )  
        pos_counts = pos_counts.sort_values(by='freq', ascending=False).iloc[:20]

        
        if secteur == 'telecom':
            topics_counts, all_cats = get_telecom_topics(results_df = results_df)
        elif secteur == 'eau':
            topics_counts, all_cats = get_eau_topics(results_df = results_df)
        elif secteur == 'automobile':
            topics_counts, all_cats = get_telecom_topics(results_df = results_df)


        all_cats_str = []
        for elem in all_cats:
            all_cats_str.append(', '.join(elem))

        cats_length = len(all_cats_str)
        all_comments_length = len(all_comments)

        type_excel = type(results_df)
        results_df['Comment'] = df_x

        n_neg = len(results_df[results_df['Polarity'] == 'Bad'])
        n_neu = len(results_df[results_df['Polarity'] == 'Good'])
        n_pos = len(results_df[results_df['Polarity'] == 'Super'])
        n_total = n_pos + n_neu + n_neg
        #results_df['Polarity1'] = df2.polarity

        #confm = np.array(confusion_matrix(df2.polarity, my_prediction, labels=[0,2,4]))
        #confusion = pd.DataFrame(confm, index=['negative', 'neutral', 'positive'],
        #                     columns=['predicted_negative','predicted_neutral','predicted_positive'])
        
    else: 
        flash("There were an error processing your data, please try again.", 'danger')
    return render_template('results_analyse_excel.html', 
        title = title,
        results_df = results_df, 
        for_wordcloud_neg = for_wordcloud_neg,
        for_wordcloud_pos = for_wordcloud_pos, 
        for_wordcloud_neu = for_wordcloud_neu, 
        neg_counts = neg_counts,
        neu_counts = neu_counts, 
        pos_counts = pos_counts,
        pos_percentage = results_percents['pos_percentage'], 
        neu_percentage = results_percents['neu_percentage'], 
        neg_percentage = results_percents['neg_percentage'], 
        cats_length = cats_length, 
        topics_counts = topics_counts,
        all_comments_length = all_comments_length, 
        secteur = secteur,
        all_cats_str = all_cats_str, 
        results_df_pos = results_df_pos, 
        type_excel = type_excel, 
        new_clean = new_clean, 
        stopwords = stopwords, 
        model = model, 
        n_neg = n_neg,
        n_neu = n_neu, 
        n_pos = n_pos, 
        n_total = n_total
        )




















# get sentiment analysis for an excel file 
@analyse.route('/results_compare_polarity', methods = ['POST', 'GET'])
def results_compare_polarity():
    if not current_user.is_authenticated:
        flash('You should be logged in to acced this page', 'info')
        next_page =  request.endpoint
        return redirect(url_for('users.login', next = next_page))
    if request.method != "POST":
        flash("You don't have permission accessing this page", 'info')
        return redirect(url_for('main.dashboard'))
    global new_df, results_df
    import pandas as pd
    #df = pd.read_excel(r'YoutubeSpamMergedData.xlsx')
    title = "Excel Sentiment Analysis"
    
    test_path = os.path.join(APP_ROOT, 'models\\' + current_user.username)


    if request.method == "POST":
        secteur = request.form['secteur']
        model = request.form['model']
        stopwords = request.form['stopwords']
        all_comments, df_x, df2, column_name = save_excel()

        new_clean = clean_text(df_content = df2, column_name = column_name, stopword = stopwords)
        Polarity = df2['Polarity']
        #all_clean = []
        import re
        #clean = re.compile('<.*?>')
        #for i in range(0, len(all_comments)):
        #    all_comments[i] = re.sub(clean, ' ', all_comments[i])
        #    all_comments[i] = all_comments[i].replace('\n',' ')
        #    all_comments[i] = all_comments[i].replace('\\n',' ')
        #    all_comments[i] = all_comments[i].replace('\t',' ')
        #    all_comments[i] = all_comments[i].replace('\r',' ')
        #    all_comments[i] = all_comments[i].replace('\'',' ')


        results_df, results_percents, results_df_pos, results_df_neu, results_df_neg, my_prediction = get_sentiment_analysis(table = new_clean, model = model)
        results_df["Original Polarity"] = Polarity
        test_data_len = len(results_df)


        from nltk.tokenize import sent_tokenize, word_tokenize
        from collections import Counter

        neg_results = results_df[results_df.Polarity == 'Bad']
        for_wordcloud_neg = ' '.join(map(str, neg_results['Comment']))
        all_words = word_tokenize(for_wordcloud_neg)
        counts = Counter(all_words)
        neg_counts = pd.DataFrame(list(counts.items()), columns = ['word', 'freq'] )  
        neg_counts = neg_counts.sort_values(by='freq', ascending=False).iloc[:20]

        neu_results = results_df[results_df.Polarity == 'Good']
        for_wordcloud_neu = ' '.join(map(str, neu_results['Comment']))
        all_words = word_tokenize(for_wordcloud_neu)
        counts = Counter(all_words)
        neu_counts = pd.DataFrame(list(counts.items()), columns = ['word', 'freq'] )     
        neu_counts = neu_counts.sort_values(by='freq', ascending=False).iloc[:20]

        pos_results = results_df[results_df.Polarity == 'Super']
        for_wordcloud_pos = ' '.join(map(str, pos_results['Comment']))
        all_words = word_tokenize(for_wordcloud_pos)
        counts = Counter(all_words)
        pos_counts = pd.DataFrame(list(counts.items()), columns = ['word', 'freq'] )  
        pos_counts = pos_counts.sort_values(by='freq', ascending=False).iloc[:20]

        
        if secteur == 'telecom':
            telecom_counts, all_cats = get_telecom_topics(results_df = results_df)
        elif secteur == 'eau':
            telecom_counts, all_cats = get_eau_topics(results_df = results_df)
        elif secteur == 'automobile':
            telecom_counts, all_cats = get_telecom_topics(results_df = results_df)


        all_cats_str = []
        for elem in all_cats:
            all_cats_str.append(', '.join(elem))

        cats_length = len(all_cats_str)
        all_comments_length = len(all_comments)

        type_excel = type(results_df)
        results_df['Comment'] = df_x
        #results_df['Polarity1'] = df2.polarity

        #confm = np.array(confusion_matrix(df2.polarity, my_prediction, labels=[0,2,4]))
        #confusion = pd.DataFrame(confm, index=['negative', 'neutral', 'positive'],
        #                     columns=['predicted_negative','predicted_neutral','predicted_positive'])
        conmat = np.array(confusion_matrix(Polarity, results_df.Polarity, labels=['Bad', 'Good', 'Super']))
        confusion = pd.DataFrame(conmat, index=['negative', 'neutral', 'positive'],
                             columns=['predicted_negative','predicted_neutral','predicted_positive'])
        class_report = classification_report(Polarity, results_df.Polarity, output_dict=True)
        class_report_df = pd.DataFrame(class_report).transpose()

        true_negative = confusion["predicted_negative"][0]
        false_negative1 = confusion["predicted_negative"][1]
        false_negative2 = confusion["predicted_negative"][2]

        false_neutral1 = confusion["predicted_neutral"][0]
        true_neutral = confusion["predicted_neutral"][1]
        false_neutral2 = confusion["predicted_neutral"][2]

        false_positive1 = confusion["predicted_positive"][0]
        false_positive2 = confusion["predicted_positive"][1]
        true_positive = confusion["predicted_positive"][2]

        count_pos = true_positive + false_positive1 + false_positive2
        count_neu = true_neutral + false_neutral1 + false_neutral2
        count_neg = true_negative + false_negative1 + false_negative2

        count_all = count_pos + count_neu + count_neg
        accuracy_avg = class_report_df.precision[3]
        accuracy_avg = float("{0:.3f}".format(accuracy_avg))


        # Showing the data as red when they are wrong predictions
        indexes = []
        for i in range(0, len(results_df.Polarity)):
            indexes.append(i)

        results_df.index = indexes

        test_data_len = len(results_df)

        wrong_pred = []
        for i in results_df.index:
            if Polarity[i] != results_df.Polarity.iloc[i]:
                wrong_pred.append(i)

        wrong_class = 'WrongPred alert alert-danger'

        len_wrong_class = len(wrong_class)



    else: 
        flash("There were an error processing your data, please try again.", 'danger')
    return render_template('results_compare_polarity.html', 
        title = title,
        results_df = results_df, 
        for_wordcloud_neg = for_wordcloud_neg,
        for_wordcloud_pos = for_wordcloud_pos, 
        for_wordcloud_neu = for_wordcloud_neu, 
        neg_counts = neg_counts,
        neu_counts = neu_counts, 
        pos_counts = pos_counts,
        pos_percentage = results_percents['pos_percentage'], 
        neu_percentage = results_percents['neu_percentage'], 
        neg_percentage = results_percents['neg_percentage'], 
        cats_length = cats_length, 
        telecom_counts = telecom_counts,
        all_comments_length = all_comments_length, 
        secteur = secteur,
        all_cats_str = all_cats_str, 
        neg_count_internet = telecom_counts['neg_count_internet'],
        neg_count_reseau = telecom_counts['neg_count_reseau'],
        neg_count_offre = telecom_counts['neg_count_offre'],
        neg_count_appelle = telecom_counts['neg_count_appelle'],
        neg_count_other = telecom_counts['neg_count_other'],
        neu_count_internet = telecom_counts['neu_count_internet'],
        neu_count_reseau = telecom_counts['neu_count_reseau'],
        neu_count_offre = telecom_counts['neu_count_offre'],
        neu_count_appelle = telecom_counts['neu_count_appelle'],
        neu_count_other = telecom_counts['neu_count_other'],
        pos_count_internet = telecom_counts['pos_count_internet'],
        pos_count_reseau = telecom_counts['pos_count_reseau'],
        pos_count_offre = telecom_counts['pos_count_offre'],
        pos_count_appelle = telecom_counts['pos_count_appelle'],
        pos_count_other = telecom_counts['pos_count_other'],
        count_internet = telecom_counts['count_internet'],
        count_reseau = telecom_counts['count_reseau'],
        count_offre = telecom_counts['count_offre'],
        count_appelle = telecom_counts['count_appelle'],
        count_other = telecom_counts['count_other'],
        results_df_pos = results_df_pos, 
        type_excel = type_excel, 
        new_clean = new_clean, 
        stopwords = stopwords, 
        model = model, 
        class_report_df = class_report_df, 
        accuracy_avg = accuracy_avg, 
        count_all = count_all, 
        count_pos = count_pos,
        count_neu = count_neu,  
        count_neg = count_neg,
        r_t_confusion = confusion, 
        r_t_class_report = class_report_df, 
        test_data_len = test_data_len, 
        #wrong_pred = wrong_pred,
        wrong_class = wrong_class, 
        Polarity = Polarity,
        len_wrong_class = len_wrong_class
        )






























































@analyse.route('/save_model', methods = ['POST', 'GET'])
def save_model():
    if not current_user.is_authenticated:
        flash('You should be logged in to acced this page', 'info')
        next_page =  request.endpoint
        return redirect(url_for('users.login', next = next_page))
    previous_link = request.referrer 
    if previous_link == None:
        return redirect(url_for('main.dashboard'))
    if request.method != "POST":
        return redirect(url_for('main.dashboard'))
    form = SaveModel()
    save_path = request.form['modelName']
    target = os.path.join(APP_ROOT, 'models/' + save_path + '/')
    print(target)

    if not os.path.isdir(target):
        os.mkdir(target)

    with open(target + 'sentiment_fit.pkl', 'wb') as f:
        pickle.dump(sentiment_fit2, f)
    with open(target + 'tg_cvec.pkl', 'wb') as f:
        pickle.dump(tg_cvec, f)
    return render_template('complete.html', save_path = save_path)









# Test comment 
@analyse.route('/results_comment', methods = ['POST'])
def results_comment():
    if not current_user.is_authenticated:
        flash('You should be logged in to acced this page', 'info')
        next_page =  request.endpoint
        return redirect(url_for('users.login', next = next_page))
    from sklearn.model_selection import train_test_split

    # Naive Bayes Classifier 
    from sklearn.naive_bayes import MultinomialNB
    # Alternative Usage of saved model 
    ytb_model = open("models/model_clf.pkl", "rb")
    clf2 = joblib.load(ytb_model)

    # Alternative Usage of saved model 
    ytb_model = open("models/model_cv.pkl", "rb")
    cv2 = joblib.load(ytb_model)

    # Alternative Usage of saved model 
    ytb_model = open("models/model_X.pkl", "rb")
    X2 = joblib.load(ytb_model)

    # Alternative Usage of saved model 
    ytb_model = open("models/model_corpus.pkl", "rb")
    corpus2 = joblib.load(ytb_model)

    if request.method == 'POST':
        comment = request.form['comment']
        data = [comment]
        vect = cv2.transform(data).toarray()
        my_prediction = clf2.predict(vect)

    return render_template('results_comment.html', 
        prediction = my_prediction, 
        cv = cv2, 
        X = X2, 
        corpus = corpus2, 
        clf2 = clf2
        )







# Test comment 
@analyse.route('/results_comment_arabic', methods = ['POST'])
def results_comment_arabic():
    if not current_user.is_authenticated:
        flash('You should be logged in to acced this page', 'info')
        next_page =  request.endpoint
        return redirect(url_for('users.login', next = next_page))
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import CountVectorizer
    import pickle
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    lr = LogisticRegression()



    from sklearn.externals import joblib 


    # Naive Bayes Classifier 
    from sklearn.naive_bayes import MultinomialNB

    ### Load and Test
    ytb_model = open("models/ar/tg_cvec.pkl", "rb")
    tg_cvec = joblib.load(ytb_model)

    ### Load and Test
    ytb_model = open("models/ar/sentiment_fit.pkl", "rb")
    sentiment_fit = joblib.load(ytb_model)

    tg_pipeline = Pipeline([
        ('vectorizer', tg_cvec),
        ('classifier', lr)
    ])


    

    if request.method == 'POST':
        comment = request.form['comment']
        #data = [comment]
        s2 = pd.Series(comment)
        my_prediction = sentiment_fit.predict(s2)


    return render_template('results_comment.html', 
        prediction = my_prediction, 
        cv = tg_cvec, 
        clf2 = sentiment_fit
        )












@analyse.route('/train_arabic')
def train_arabic():
    if not current_user.is_authenticated:
        flash('You should be logged in to acced this page', 'info')
        next_page =  request.endpoint
        return redirect(url_for('users.login', next = next_page))
    title = "Training Page"
    return render_template('train_arabic.html', title = title)


@analyse.route('/results_arabic', methods = ['POST'])
def results_arabic():
    if not current_user.is_authenticated:
        flash('You should be logged in to acced this page', 'info')
        next_page =  request.endpoint
        return redirect(url_for('users.login', next = next_page))
    global corpus, clf, y_pred
    form = SaveModel()
    #df = pd.read_csv('YoutubeSpamMergedData.csv')
    target = os.path.join(APP_ROOT, 'trainingcsv/')
    print(target)

    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist("file"):
        print(file)
        filename = file.filename
        destination = "/".join([target, filename])
        print(destination)
        file.save(destination)
        df2 = pd.read_excel(os.path.join(APP_ROOT, 'trainingcsv/'+filename))
        df_data2 = df2[["CONTENT", "CLASS"]]
        df_x2 = df_data2['CONTENT']
        df_y2 = df_data2.CLASS


    df_data = df2[["CONTENT", "CLASS"]]
    df = df_data
    df = df[df.CLASS != 0]
    df_n = df[df.CLASS == -1]
    df_p = df[df.CLASS == 1]
    df_n = df_n.assign(CLASS= 0)
    df_p = df_p.assign(CLASS= 1)
    df3 = df_n.append(df_p, ignore_index=True)
    df = df3
    my_df = df
    df_neg = my_df[my_df.CLASS == 0][0:2000]
    df_pos = my_df[my_df.CLASS == 1][0:2000]
    df3 = df_neg.append(df_pos, ignore_index=True)
    df = df3
    df_data = df

    # Features and Labels 
    df_x = df_data['CONTENT']
    df_y = df_data.CLASS




    # eXTRACT FEATURE WITH CountVectorizer
    corpus = df_x 
    cv = CountVectorizer()
    X = cv.fit_transform(corpus) # Fit the data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, df_y, test_size = 0.33, random_state = 42)
    # Naive Bayes Classifier 
    from sklearn.naive_bayes import MultinomialNB
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    clf.score(X_test, y_test)
    # Alternative Usage of saved model 
    # ytb_model = open("naivebayes_spam_model.pkl", "rb")
    # clf = joblib.load(ytb_model)

    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    conmat = np.array(confusion_matrix(y_test, y_pred, labels=[0,1]))

    confusion = pd.DataFrame(conmat, index=['negative', 'positive'],
                         columns=['predicted_negative','predicted_positive'])

    true_negative = confusion["predicted_negative"][0]
    false_negative = confusion["predicted_negative"][1]

    false_positive = confusion["predicted_positive"][0]
    true_positive = confusion["predicted_positive"][1]

    count_pos = len(y_pred[y_pred == 1])
    count_neg = len(y_pred[y_pred == 0])
    count_all = len(y_pred)


    #if request.method == 'POST':
    #    comment = request.form['comment']
    #    data = [comment]
    #    vect = cv.transform(data).toarray()
    #    my_prediction = clf.predict(vect)
    #else:
    #    comment = 'Great content, just continue publishing others'
    #    data = [comment]
    #    vect = cv.transform(data).toarray()
    #    my_prediction = clf.predict(vect)

    return render_template('results_arabic.html', 
        #prediction = my_prediction, 
        count_all = count_all, 
        count_neg = count_neg,
        count_pos = count_pos, 
        confusion = confusion,
        accuracy = accuracy, 
        true_negative = true_negative, 
        false_negative = false_negative, 
        false_positive = false_positive, 
        true_positive = true_positive, 
        df2 = df2, 
        df_y2 = df_y2, 
        df_data2 = df_data2, 
        df_neg = df_neg, 
        df_pos = df_pos,
        y_pred = y_pred,
        corpus = corpus, 
        clf = clf, 
        form = form
        )






@analyse.route('/results_test_arabic', methods = ['POST'])
def results_test_arabic():
    if not current_user.is_authenticated:
        flash('You should be logged in to acced this page', 'info')
        next_page =  request.endpoint
        return redirect(url_for('users.login', next = next_page))
    target = os.path.join(APP_ROOT, 'testxlsx/')
    print(target)

    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist("file"):
        print(file)
        filename = file.filename
        destination = "/".join([target, filename])
        print(destination)
        file.save(destination)
        df2 = pd.read_excel(os.path.join(APP_ROOT, 'testxlsx/'+filename))
        df_data2 = df2[["CONTENT", "CLASS"]]
        df_x2 = df_data2['CONTENT']
        test_x = df_x2
        df_y2 = df_data2.CLASS
        df_test = df_data2

    #df2 = pd.read_excel(os.path.join(APP_ROOT, 'testxlsx/HTL_test.xlsx'))
    #df_data2 = df2[["CONTENT", "CLASS"]]
    #df_x2 = df_data2['CONTENT']
    #test_x = df_x2
    #df_y2 = df_data2.CLASS
    #df_test = df_data2


    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import CountVectorizer
    import pickle
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    lr = LogisticRegression()



    from sklearn.externals import joblib 


    # Naive Bayes Classifier 
    from sklearn.naive_bayes import MultinomialNB

    ### Load and Test
    ytb_model = open("models/ar/tg_cvec.pkl", "rb")
    tg_cvec = joblib.load(ytb_model)

    ### Load and Test
    ytb_model = open("models/ar/sentiment_fit.pkl", "rb")
    sentiment_fit = joblib.load(ytb_model)

    tg_pipeline = Pipeline([
        ('vectorizer', tg_cvec),
        ('classifier', lr)
    ])

    s2 = pd.Series(df_x2)
    my_prediction = sentiment_fit.predict(s2)


    my_comments = []
    test_preds = []
    for i in range(len(test_x)):
        if my_prediction[i] == 4: 
            my_prediction_cat = 'Super'
        else:
            my_prediction_cat = 'Bad'
        test_preds.append(my_prediction_cat)
        my_comments.append(test_x[i])
        result_list = pd.DataFrame(
            {'Commentaire': my_comments,
             'Sentiment': test_preds
            })
        n_pos_neg = result_list['Sentiment'].value_counts()
        n_neg = len(result_list[result_list['Sentiment'] == 'Bad'])
        n_pos = len(result_list[result_list['Sentiment'] == 'Super'])
        n_total = n_pos + n_neg



    return render_template('results_test_ar.html', 
        prediction = my_prediction, 
        result_list = result_list, 
        n_total = n_total, 
        n_neg = n_neg, 
        n_pos = n_pos, 
        my_comments = my_comments
        )



@analyse.route('/train_words')
def train_words():
    if not current_user.is_authenticated:
        flash('You should be logged in to acced this page', 'info')
        next_page =  request.endpoint
        return redirect(url_for('users.login', next = next_page))
    title = "Training Page"
    test_emoji = r"U+1F929"
    import re
    b=u"united thats weak. See ya 😇  "
    assert re.findall(u'[\U0001f600-\U0001f650]',b) == [u'😇']
    assert re.findall(u'[😀-🙏]',b) == [u'😇']
    return render_template('train_words.html', title = title, 
        test_emoji = test_emoji, b = b)






#########################################
######### ALL ROUTES FOR DAREJA #########
#########################################
@analyse.route('/train_dareja')
def train_dareja():
    if not current_user.is_authenticated:
        flash('You should be logged in to acced this page', 'info')
        next_page =  request.endpoint
        return redirect(url_for('users.login', next = next_page))
    title = "Training Page"
    test_emoji = r"U+1F929"
    import re
    b=u"united thats weak. See ya 😇  "
    assert re.findall(u'[\U0001f600-\U0001f650]',b) == [u'😇']
    assert re.findall(u'[😀-🙏]',b) == [u'😇']
    return render_template('train_dareja.html', title = title, 
        test_emoji = test_emoji, b = b)

@analyse.route('/choose_fields', methods = ['POST'])
def choose_fields():
    if not current_user.is_authenticated:
        flash('You should be logged in to acced this page', 'info')
        next_page =  request.endpoint
        return redirect(url_for('users.login', next = next_page))
    #df = pd.read_csv('YoutubeSpamMergedData.csv')
    target = os.path.join(APP_ROOT, 'trainingcsv/')
    print(target)

    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist("file"):
        print(file)
        filename = file.filename
        destination = "/".join([target, filename])
        print(destination)
        file.save(destination)
        df = pd.read_excel(os.path.join(APP_ROOT, 'trainingcsv/'+filename))
        cols = []
        for col in df.columns: 
            polarity_col = df[str(col)].unique()
            if len(polarity_col) < 4:
                polarity_value = col
            cols.append(col)

        polarity_values = df.polarity.unique()
        #count_column1 = count_column[0]

    title = "Choose Fields"
    return render_template('choose_fields.html',
        title = title, 
        cols = cols, 
        polarity_values = polarity_values, 
        polarity_col = polarity_col, 
        polarity_value = polarity_value)


import math
def equilibrate_polarities(df_neg = '', df_neu = '', df_pos = '', test_valid_percent = ''):
    len_neg = len(df_neg)
    len_neu = len(df_neu)
    len_pos = len(df_pos)
    all_n = max(len_neg, len_neu, len_pos)
    test_valid_percent = int(all_n * 5 // 100)
    if all_n == len_neu:
        print("neutrals are dominant")
        max_n = all_n - test_valid_percent
        # create the df for training set positives, and test-validation set for positievs
        df_pos_train = df_pos[0:len(df_pos) - test_valid_percent]
        df_pos_test = df_pos[len(df_pos) - test_valid_percent:len(df_pos)]
        duplicates_pos = math.ceil(max_n / len(df_pos_train))
        new_df_pos = pd.DataFrame()
        for i in range(0, duplicates_pos):
            new_df_pos = new_df_pos.append(df_pos_train)
        # create the df for training set negatives, and test-validation set for negatives
        df_neg_train = df_neg[0:len(df_neg) - test_valid_percent]
        df_neg_test = df_neg[len(df_neg) - test_valid_percent:len(df_neg)]
        duplicates_neg = math.ceil(max_n / len(df_neg_train))
        new_df_neg = pd.DataFrame()
        for i in range(0, duplicates_neg):
            new_df_neg = new_df_neg.append(df_neg_train)
        new_df_neu = df_neu[0:len(df_neu) - test_valid_percent]
        df_neu_test = df_neu[len(df_neu) - test_valid_percent:len(df_neu)]
    elif all_n == len_neg:
        print("negatives are dominant")
        max_n = all_n - test_valid_percent
        # create the df for training set positives, and test-validation set for positievs
        df_pos_train = df_pos[0:len(df_pos) - test_valid_percent]
        df_pos_test = df_pos[len(df_pos) - test_valid_percent:len(df_pos)]
        duplicates_pos = math.ceil(max_n / len(df_pos_train))
        new_df_pos = pd.DataFrame()
        for i in range(0, duplicates_pos):
            new_df_pos = new_df_pos.append(df_pos_train)
        # create the df for training set neutral, and test-validation set for neutrals
        df_neu_train = df_neu[0:len(df_neu) - test_valid_percent]
        df_neu_test = df_neu[len(df_neu) - test_valid_percent:len(df_neu)]
        duplicates_neu = math.ceil(max_n / len(df_neg_train))
        new_df_neu = pd.DataFrame()
        for i in range(0, duplicates_neu):
            new_df_neu = new_df_neu.append(df_neu_train)
        new_df_neg = df_neg[0:len(df_neg) - test_valid_percent]
        df_neg_test = df_neg[len(df_neg) - test_valid_percent:len(df_neg)]
    elif all_n == len_pos:
        print("Positives are dominant")
        max_n = all_n - test_valid_percent
        # create the df for training set positives, and test-validation set for positievs
        df_neu_train = df_neu[0:len(df_neu) - test_valid_percent]
        df_neu_test = df_neu[len(df_neu) - test_valid_percent:len(df_neu)]
        duplicates_neu = math.ceil(max_n / len(df_neu_train))
        new_df_neu = pd.DataFrame()
        for i in range(0, duplicates_neu):
            new_df_neu = new_df_neu.append(df_neu_train)
        # create the df for training set negatives, and test-validation set for negatives
        df_neg_train = df_neg[0:len(df_neg) - test_valid_percent]
        df_neg_test = df_neg[len(df_neg) - test_valid_percent:len(df_neg)]
        duplicates_neg = math.ceil(max_n / len(df_neg_train))
        new_df_neg = pd.DataFrame()
        for i in range(0, duplicates_neg):
            new_df_neg = new_df_neg.append(df_neg_train)
        # assigning new class
        new_df_pos = df_pos[0:len(df_pos) - test_valid_percent]
        df_pos_test = df_pos[len(df_pos) - test_valid_percent:len(df_pos)]
    return new_df_neg, df_neg_test, new_df_neu, df_neu_test, new_df_pos, df_pos_test



@analyse.route('/results_dareja2', methods = ['POST', 'GET'])
def results_dareja2():
    global tg_cvec, sentiment_fit2, df_save
    if not current_user.is_authenticated:
        flash('You should be logged in to acced this page', 'info')
        next_page =  request.endpoint
        return redirect(url_for('users.login', next = next_page))

    if request.method != "POST":
        flash("You don't have permission accessing this page", 'info')
        return redirect(url_for('main.dashboard'))
            
    form = SaveModel()
    #ratio_train = request.form['ratio_train']
    #ratio_test = int(100 - int(ratio_train)) // 2
    ratio_train = 95
    ratio_test = 2.5
    #df = pd.read_csv('YoutubeSpamMergedData.csv')
    target = os.path.join(APP_ROOT, 'trainingcsv/')
    print(target)

    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist("file"):
        print(file)
        filename = file.filename
        destination = "/".join([target, filename])
        print(destination)
        file.save(destination)
        df = pd.read_excel(os.path.join(APP_ROOT, 'trainingcsv/'+filename))
        df['text'] = df['text'].astype(str)
        df_data = df[["text", "polarity"]]
        #df_data = clean_text(df_content = df_data, column_name = 'text', stopword = 'french')
        #df_data['text'] = df_data['comment']
        df_x = df_data['text']
        df_y = df_data['polarity']
        df_save = df

    df['text'] = df.text.astype(str)
    df_neg = df[df.polarity == -1]
    df_pos = df[df.polarity == 1]
    df_neu = df[df.polarity == 0]
    df_neg = df_neg.assign(polarity= 0)
    df_neu = df_neu.assign(polarity= 2)
    df_pos = df_pos.assign(polarity= 4)
    df3 = df_neg.append(df_neu, ignore_index=True)
    df3 = df3.append(df_pos, ignore_index=True)
    # NOW , it has only POSITIVES AND NEGATIVES 
    df = df3

    my_df = df

    # get the lengths of the three categories
    len_neg = len(df_neg)
    len_neu = len(df_neu)
    len_pos = len(df_pos)

    equilibrate_class = True
    if equilibrate_class == True:
        df_neg, df_neg_test, df_neu, df_neu_test, df_pos, df_pos_test = equilibrate_polarities(df_neg = df_neg, df_neu = df_neu, df_pos = df_pos, test_valid_percent = '')
        frames = [df_neg, df_neu, df_pos]
        result = pd.concat(frames, ignore_index=True)
        x = result.text
        y = result.polarity
        x_train = x
        y_train = y
        frames = [df_neg_test, df_neu_test, df_pos_test]
        result_1 = pd.concat(frames, ignore_index=True)
        result_2 = result_1
        x_test = result_1.text
        y_test = result_1.polarity
        x_validation = result_1.text
        y_validation = result_1.polarity
        SEED = 2000
        result_1.to_excel(target + "/testing_data_sonelgaz.xlsx")
        result.to_excel(target + "/training_data_sonelgaz.xlsx")
    else:
        min_n_cats = min(len_neg, len_neu, len_pos)

        n_train =  int(min_n_cats * ratio_train // 100)
        n_test = int(min_n_cats * ratio_test // 100)
        n_validate = int(min_n_cats * ratio_test // 100)

        total = n_train + n_test + n_validate

        my_df['text'] = my_df.text.astype(str)

        df_neg = my_df[my_df.polarity == 0][0:n_train]
        df_neu = my_df[my_df.polarity == 2][0:n_train]
        df_pos = my_df[my_df.polarity == 4][0:n_train]
        frames = [df_neg, df_neu, df_pos]
        result = pd.concat(frames, ignore_index=True)
        # THIS VARIABLE HAS ALL THE TRAINING DATA SEPARATED 
        result
        #result_clean = clean_text_for_training(df_content = result, column_name = 'text', stopword = 'french')
        #result_clean['text'] = result_clean['comment']
        #result = result_clean
        # This will give you the number of feature names

        df.head()

        x = result.text
        y = result.polarity

        #x_train_clean = clean_text_for_training(df_content = result, column_name = 'text', stopword = 'french')
        #x_train_clean['text'] = x_train_clean['comment']
        #x = x_train_clean['text'] 
        
        SEED = 2000

        # assigning x and y directly as the train set
        x_train = x
        y_train = y 


        # setting the x_test and y_test 
        # Testing Sample
        df_neg_1 = my_df[my_df.polarity == 0][len(df_neg) - (n_validate * 2):len(df_neg) - n_validate]
        df_neu_1 = my_df[my_df.polarity == 2][len(df_neg) - (n_validate * 2):len(df_neg) - n_validate]
        df_pos_1 = my_df[my_df.polarity == 4][len(df_neg) - (n_validate * 2):len(df_neg) - n_validate]
        frames_1 = [df_neg_1, df_neu_1, df_pos_1]
        result_1 = pd.concat(frames_1, ignore_index = True)
        x_test = result_1.text
        y_test = result_1.polarity

        result_1.to_excel(target + "/testing_data_sonelgaz.xlsx")

        #x_test_clean = clean_text_for_training(df_content = result_1, column_name = 'text', stopword = 'french')
        #x_test_clean['text'] = x_test_clean['comment']
        #x_test = x_test_clean['text'] 

        

        #Validation sample
        df_neg_2 = my_df[my_df.polarity == 0][len(df_neg) - n_validate:len(df_neg)]
        df_neu_2 = my_df[my_df.polarity == 2][len(df_neg) - n_validate:len(df_neg)]
        df_pos_2 = my_df[my_df.polarity == 4][len(df_neg) - n_validate:len(df_neg)]
        frames_2 = [df_neg_2, df_neu_2, df_pos_2]
        result_2 = pd.concat(frames_2, ignore_index = True)
        x_validation = result_2.text
        y_validation = result_2.polarity 

        result_2.to_excel(target + "/validation_data_sonelgaz.xlsx")

    #d = {'text':x_validation,'polarity':y_validation}
    #result_2 = pd.DataFrame(d)

    #x_validation_clean = clean_text_for_training(df_content = result_2, column_name = 'text', stopword = 'french')
    #x_validation_clean['text'] = x_validation_clean['comment']
    #x_validation = x_validation_clean['text'] 



    print ("Train set has total {0} entries with {1:.2f}% negative, {2:.2f}% Good, {3:.2f}% positive".format(len(x_train),
        (len(x_train[y_train == 0]) / (len(x_train)*1.))*100,
        (len(x_train[y_train == 2]) / (len(x_train)*1.))*100,
        (len(x_train[y_train == 4]) / (len(x_train)*1.))*100) )


    print ("Validation set has total {0} entries with {1:.2f}% negative, {2:.2f}% Good, {3:.2f}% positive".format(len(x_validation),
        (len(x_validation[y_validation == 0]) / (len(x_validation)*1.))*100,
        (len(x_validation[y_validation == 2]) / (len(x_validation)*1.))*100,
        (len(x_validation[y_validation == 4]) / (len(x_validation)*1.))*100) )

    print ("Test set has total {0} entries with {1:.2f}% negative, {2:.2f}% Good, {3:.2f}% positive".format(len(x_test),
        (len(x_test[y_test == 0]) / (len(x_test)*1.))*100,
        (len(x_test[y_test == 2]) / (len(x_test)*1.))*100,
        (len(x_test[y_test == 4]) / (len(x_test)*1.))*100) )


    from textblob import TextBlob
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import classification_report, confusion_matrix
    # Feature extraction
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from time import time




    def accuracy_summary(pipeline, x_train, y_train, x_test, y_test):
        if len(x_test[y_test == 0]) / (len(x_test)*1.) > 0.33:
            null_accuracy = len(x_test[y_test == 0]) / (len(x_test)*1.)
        else:
            null_accuracy = 1. - (len(x_test[y_test == 0]) / (len(x_test)*1.))
        t0 = time()
        sentiment_fit = pipeline.fit(x_train, y_train)
        y_pred = sentiment_fit.predict(x_test)
        train_test_time = time() - t0
        accuracy = accuracy_score(y_test, y_pred)
        print ("null accuracy: {0:.2f}%".format(null_accuracy*100) )
        print ("accuracy score: {0:.2f}%".format(accuracy*100) )
        if accuracy > null_accuracy:
            print ("model is {0:.2f}% more accurate than null accuracy".format((accuracy-null_accuracy)*100) )
        elif accuracy == null_accuracy:
            print ("model has the same accuracy with the null accuracy")
        else:
            print ("model is {0:.2f}% less accurate than null accuracy".format((null_accuracy-accuracy)*100) )
        print ("train and test time: {0:.2f}s".format(train_test_time) )
        print ("-"*80 )
        return accuracy, train_test_time


    cvec = CountVectorizer()
    # get the number of feature names , but first we fit the text
    cvec.fit(x)
    len(cvec.get_feature_names())
    num_features = len(cvec.get_feature_names())
    cv_steps = round(num_features / 8)

    lr = LogisticRegression(solver='newton-cg', multi_class='multinomial')
    n_features = np.arange(cv_steps, num_features, cv_steps)


    # get the stemmer function, this is optional 
    # with stemming words
    import nltk
    from nltk.stem.isri import ISRIStemmer
    stemmer = ISRIStemmer()
    analyzer = CountVectorizer().build_analyzer()

    def stemmed_words(doc):
        return (stemmer.stem(w) for w in analyzer(doc))
    # remove the above code as it is used in another function(if any errors)


    def nfeature_accuracy_checker(vectorizer=cvec, n_features=n_features, stop_words=None, ngram_range=(1, 1), classifier=lr):
        result = []
        print (classifier)
        print ("\n")
        for n in n_features:
            vectorizer.set_params(stop_words=stop_words, max_features=n, ngram_range=ngram_range)
            checker_pipeline = Pipeline([
                ('vectorizer', vectorizer),
                ('classifier', classifier)
            ])
            print ("Validation result for {} features".format(n) )
            nfeature_accuracy,tt_time = accuracy_summary(checker_pipeline, x_train, y_train, x_validation, y_validation)
            result.append((n,nfeature_accuracy,tt_time))
        return result


    my_stop_words = ['selmane', 'aissa', 'brahim']

    print ("RESULT FOR UNIGRAM WITHOUT STOP WORDS\n")
    feature_result_wosw = nfeature_accuracy_checker(stop_words = 'english') 

    print ("RESULT FOR UNIGRAM WITH STOP WORDS\n")
    feature_result_ug = nfeature_accuracy_checker() 

    print ("RESULT FOR UNIGRAM WITHOUT CUSTOM STOP WORDS (Top 10 frequent words)\n" )
    feature_result_wocsw = nfeature_accuracy_checker(stop_words=my_stop_words) 


    nfeatures_plot_ug = pd.DataFrame(feature_result_ug,columns=['nfeatures','validation_accuracy','train_test_time'])
    nfeatures_plot_ug_wocsw = pd.DataFrame(feature_result_wocsw,columns=['nfeatures','validation_accuracy','train_test_time'])
    nfeatures_plot_ug_wosw = pd.DataFrame(feature_result_wosw,columns=['nfeatures','validation_accuracy','train_test_time'])

    plt.figure(figsize=(8,6))
    plt.plot(nfeatures_plot_ug.nfeatures, nfeatures_plot_ug.validation_accuracy, label='with stop words')
    plt.plot(nfeatures_plot_ug_wosw.nfeatures, nfeatures_plot_ug_wosw.validation_accuracy,label='without stop words')
    plt.plot(nfeatures_plot_ug_wocsw.nfeatures, nfeatures_plot_ug_wocsw.validation_accuracy,label='without custom stop words')
    plt.title("Without stop words VS With stop words (Unigram): Accuracy")
    plt.xlabel("Number of features")
    plt.ylabel("Validation set accuracy")
    #plt.legend()

    ug_labels = nfeatures_plot_ug.nfeatures
    ug_values = nfeatures_plot_ug.validation_accuracy

    ug_wocsw_labels = nfeatures_plot_ug_wocsw.nfeatures
    ug_wocsw_values = nfeatures_plot_ug_wocsw.validation_accuracy

    ug_wosw_labels = nfeatures_plot_ug_wosw.nfeatures
    ug_wosw_values = nfeatures_plot_ug_wosw.validation_accuracy


    # Showing the tests for bigrams and trigrams 
    print ("RESULT FOR BIGRAM WITH STOP WORDS\n")
    feature_result_bg = nfeature_accuracy_checker(ngram_range=(1, 2))

    print ("RESULT FOR TRIGRAM WITH STOP WORDS\n")
    feature_result_tg = nfeature_accuracy_checker(ngram_range=(1, 3))


    nfeatures_plot_tg = pd.DataFrame(feature_result_tg,columns=['nfeatures','validation_accuracy','train_test_time'])
    nfeatures_plot_bg = pd.DataFrame(feature_result_bg,columns=['nfeatures','validation_accuracy','train_test_time'])
    nfeatures_plot_ug = pd.DataFrame(feature_result_ug,columns=['nfeatures','validation_accuracy','train_test_time'])

    plt.figure(figsize=(8,6))
    plt.plot(nfeatures_plot_tg.nfeatures, nfeatures_plot_tg.validation_accuracy,label='trigram')
    plt.plot(nfeatures_plot_bg.nfeatures, nfeatures_plot_bg.validation_accuracy,label='bigram')
    plt.plot(nfeatures_plot_ug.nfeatures, nfeatures_plot_ug.validation_accuracy, label='unigram')
    plt.title("N-gram(1~3) test result : Accuracy")
    plt.xlabel("Number of features")
    plt.ylabel("Validation set accuracy")
    plt.legend()
    #plt.show()

    bg_values = nfeatures_plot_bg.validation_accuracy
    tg_values = nfeatures_plot_tg.validation_accuracy

    def train_test_and_evaluate(pipeline, x_train, y_train, x_test, y_test):
        if len(x_test[y_test == 0]) / (len(x_test)*1.) > 0.33:
            null_accuracy = len(x_test[y_test == 0]) / (len(x_test)*1.)
        else:
            null_accuracy = 1. - (len(x_test[y_test == 0]) / (len(x_test)*1.))
        sentiment_fit = pipeline.fit(x_train, y_train)
        y_pred = sentiment_fit.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        conmat = np.array(confusion_matrix(y_test, y_pred, labels=[0,2,4]))
        confusion = pd.DataFrame(conmat, index=['negative', 'neutral', 'positive'],
                             columns=['predicted_negative','predicted_neutral','predicted_positive'])
        class_report = classification_report(y_test, y_pred, output_dict=True)
        class_report_df = pd.DataFrame(class_report).transpose()
        print ("null accuracy: {0:.2f}%".format(null_accuracy*100) )
        print ("accuracy score: {0:.2f}%".format(accuracy*100) )
        if accuracy > null_accuracy:
            print ("model is {0:.2f}% more accurate than null accuracy".format((accuracy-null_accuracy)*100) )
        elif accuracy == null_accuracy:
            print ("model has the same accuracy with the null accuracy" )
        else:
            print ("model is {0:.2f}% less accurate than null accuracy".format((null_accuracy-accuracy)*100) )
        print ("-"*80 )
        print ("Confusion Matrix\n" )
        print (confusion)
        print ("-"*80 )
        print ("Classification Report\n" )
        print (classification_report(y_test, y_pred, target_names=['negative', 'neutral', 'positive']) )
        return confusion , class_report_df


    tg_cvec = CountVectorizer(max_features=num_features,ngram_range=(1, 3))
    tg_pipeline = Pipeline([
            ('vectorizer', tg_cvec),
            ('classifier', lr)
        ])
    r_t_confusion, r_t_class_report = train_test_and_evaluate(tg_pipeline, x_train, y_train, x_test, y_test)

    r_v_confusion, r_v_class_report = train_test_and_evaluate(tg_pipeline, x_train, y_train,x_validation, y_validation )
    # getting the number of true positives and negatives to show on the website 

    true_negative = r_t_confusion["predicted_negative"][0]
    false_negative1 = r_t_confusion["predicted_negative"][1]
    false_negative2 = r_t_confusion["predicted_negative"][2]

    false_neutral1 = r_t_confusion["predicted_neutral"][0]
    true_neutral = r_t_confusion["predicted_neutral"][1]
    false_neutral2 = r_t_confusion["predicted_neutral"][2]

    false_positive1 = r_t_confusion["predicted_positive"][0]
    false_positive2 = r_t_confusion["predicted_positive"][1]
    true_positive = r_t_confusion["predicted_positive"][2]

    count_pos = true_positive + false_positive1 + false_positive2
    count_neu = true_neutral + false_neutral1 + false_neutral2
    count_neg = true_negative + false_negative1 + false_negative2

    count_all = count_pos + count_neu + count_neg
    accuracy_avg = r_t_class_report.precision[3]
    accuracy_avg = float("{0:.3f}".format(accuracy_avg))
    # there are some functions that have been used before, and 
    # i am testing on tf-idf , unigrams, bigrams, trigrams 
    # and printing the results on a graph
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from time import time


    def accuracy_summary(pipeline, x_train, y_train, x_test, y_test):
        if len(x_test[y_test == 0]) / (len(x_test)*1.) > 0.33:
            null_accuracy = len(x_test[y_test == 0]) / (len(x_test)*1.)
        else:
            null_accuracy = 1. - (len(x_test[y_test == 0]) / (len(x_test)*1.))
        t0 = time()
        sentiment_fit = pipeline.fit(x_train, y_train)
        y_pred = sentiment_fit.predict(x_test)
        train_test_time = time() - t0
        accuracy = accuracy_score(y_test, y_pred)
        print ("null accuracy: {0:.2f}%".format(null_accuracy*100) )
        print ("accuracy score: {0:.2f}%".format(accuracy*100) )
        if accuracy > null_accuracy:
            print ("model is {0:.2f}% more accurate than null accuracy".format((accuracy-null_accuracy)*100) )
        elif accuracy == null_accuracy:
            print ("model has the same accuracy with the null accuracy")
        else:
            print ("model is {0:.2f}% less accurate than null accuracy".format((null_accuracy-accuracy)*100) )
        print ("train and test time: {0:.2f}s".format(train_test_time) )
        print ("-"*80 )
        return (accuracy, train_test_time )

    cvec = CountVectorizer()
    lr = LogisticRegression(solver='newton-cg', multi_class='multinomial')
    n_features = np.arange(5000,num_features,5000)

    def nfeature_accuracy_checker(vectorizer=cvec, n_features=n_features, stop_words=None, ngram_range=(1, 1), classifier=lr):
        result = []
        print (classifier)
        print ("\n")
        for n in n_features:
            vectorizer.set_params(stop_words=stop_words, max_features=n, ngram_range=ngram_range)
            checker_pipeline = Pipeline([
                ('vectorizer', vectorizer),
                ('classifier', classifier)
            ])
            print ("Validation result for {} features".format(n) )
            nfeature_accuracy,tt_time = accuracy_summary(checker_pipeline, x_train, y_train, x_validation, y_validation)
            result.append((n,nfeature_accuracy,tt_time))
        return result

    from sklearn.feature_extraction.text import TfidfVectorizer
    tvec = TfidfVectorizer()

    feature_result_ugt = nfeature_accuracy_checker(vectorizer=tvec)
    feature_result_bgt = nfeature_accuracy_checker(vectorizer=tvec,ngram_range=(1, 2))
    feature_result_tgt = nfeature_accuracy_checker(vectorizer=tvec,ngram_range=(1, 3))


    nfeatures_plot_tgt = pd.DataFrame(feature_result_tgt,columns=['nfeatures','validation_accuracy','train_test_time'])
    nfeatures_plot_bgt = pd.DataFrame(feature_result_bgt,columns=['nfeatures','validation_accuracy','train_test_time'])
    nfeatures_plot_ugt = pd.DataFrame(feature_result_ugt,columns=['nfeatures','validation_accuracy','train_test_time'])
    plt.figure(figsize=(8,6))
    plt.plot(nfeatures_plot_tgt.nfeatures, nfeatures_plot_tgt.validation_accuracy,label='trigram tfidf vectorizer',color='royalblue')
    plt.plot(nfeatures_plot_tg.nfeatures, nfeatures_plot_tg.validation_accuracy,label='trigram count vectorizer',linestyle=':', color='royalblue')
    plt.plot(nfeatures_plot_bgt.nfeatures, nfeatures_plot_bgt.validation_accuracy,label='bigram tfidf vectorizer',color='orangered')
    plt.plot(nfeatures_plot_bg.nfeatures, nfeatures_plot_bg.validation_accuracy,label='bigram count vectorizer',linestyle=':',color='orangered')
    plt.plot(nfeatures_plot_ugt.nfeatures, nfeatures_plot_ugt.validation_accuracy, label='unigram tfidf vectorizer',color='gold')
    plt.plot(nfeatures_plot_ug.nfeatures, nfeatures_plot_ug.validation_accuracy, label='unigram count vectorizer',linestyle=':',color='gold')
    plt.title("N-gram(1~3) test result : Accuracy")
    plt.xlabel("Number of features")
    plt.ylabel("Validation set accuracy")
    plt.legend()
    #plt.show()

    ugt_values = nfeatures_plot_bgt.validation_accuracy
    bgt_values = nfeatures_plot_bgt.validation_accuracy
    tgt_values = nfeatures_plot_tgt.validation_accuracy



    # Comparing with other classifiers
    # Now , we compare using different algorithms:
    from sklearn.svm import LinearSVC
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.naive_bayes import MultinomialNB, BernoulliNB
    from sklearn.linear_model import RidgeClassifier
    from sklearn.linear_model import PassiveAggressiveClassifier
    from sklearn.linear_model import Perceptron
    from sklearn.neighbors import NearestCentroid
    from sklearn.feature_selection import SelectFromModel
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer

    names = ["Logistic Regression", "Linear SVC", "LinearSVC with L1-based feature selection","Multinomial NB", 
             "Bernoulli NB", "Ridge Classifier", "AdaBoost", "Perceptron","Passive-Aggresive", "Nearest Centroid"]
    classifiers = [
        LogisticRegression(solver='newton-cg', multi_class='multinomial'),
        LinearSVC(),
        Pipeline([
      ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False))),
      ('classification', LinearSVC(penalty="l2"))]),
        MultinomialNB(),
        BernoulliNB(),
        RidgeClassifier(),
        AdaBoostClassifier(),
        Perceptron(),
        PassiveAggressiveClassifier(),
        NearestCentroid()
        ]
    zipped_clf = zip(names,classifiers)

    tvec = TfidfVectorizer()

    def classifier_comparator(vectorizer=tvec, n_features= num_features, stop_words=None, ngram_range=(1, 1), classifier=zipped_clf):
        result = []
        vectorizer.set_params(stop_words=stop_words, max_features=n_features, ngram_range=ngram_range)
        for n,c in classifier:
            checker_pipeline = Pipeline([
                ('vectorizer', vectorizer),
                ('classifier', c)
            ])
            print ("Validation result for {}".format(n) )
            print (c)
            clf_accuracy,tt_time = accuracy_summary(checker_pipeline, x_train, y_train, x_validation, y_validation)
            result.append((n,clf_accuracy,tt_time))
        return result

    comparing_algorithms = classifier_comparator(n_features=num_features,ngram_range=(1,3))



    comparing_algorithms = pd.DataFrame(comparing_algorithms, columns=['Model','Validation set accuracy', 'Train and test time'])









    # Getting The positives and Negatives 
    my_df = result


    # Creating the count vectorizer for later use 
    from sklearn.feature_extraction.text import CountVectorizer
    cvec = CountVectorizer()
    my_df['text'] = my_df.text.astype(str)
    cvec.fit(my_df.text)

    # This will give you the number of feature names
    len(cvec.get_feature_names())
    # this above variable is important 


    # This will generate the terpm frequency for pos and neg and then combined 
    neg_doc_matrix = cvec.transform(my_df[my_df.polarity == 0].text)
    neu_doc_matrix = cvec.transform(my_df[my_df.polarity == 2].text)
    pos_doc_matrix = cvec.transform(my_df[my_df.polarity == 4].text)
    neg_tf = np.sum(neg_doc_matrix,axis=0)
    neu_tf = np.sum(neu_doc_matrix,axis=0)
    pos_tf = np.sum(pos_doc_matrix,axis=0)
    neg = np.squeeze(np.asarray(neg_tf))
    neu = np.squeeze(np.asarray(neu_tf))
    pos = np.squeeze(np.asarray(pos_tf))
    words = cvec.get_feature_names()
    term_freq_df = pd.DataFrame([neg,neu,pos], columns = cvec.get_feature_names()).transpose()



    document_matrix = cvec.transform(my_df.text)
    my_df[my_df.polarity == 0].tail()

    neg_batches = np.linspace(0,4500,100).astype(int)
    i=0
    neg_tf = []
    while i < len(neg_batches)-1:
        batch_result = np.sum(document_matrix[neg_batches[i]:neg_batches[i+1]].toarray(),axis=0)
        neg_tf.append(batch_result)
        if (i % 10 == 0) | (i == len(neg_batches)-2):
            print(neg_batches[i+1],"entries' term freuquency calculated")
        i += 1


    neu_batches = np.linspace(4501,9000,100).astype(int)
    i=0
    neu_tf = []
    while i < len(neu_batches)-1:
        batch_result = np.sum(document_matrix[neu_batches[i]:neu_batches[i+1]].toarray(),axis=0)
        neu_tf.append(batch_result)
        if (i % 10 == 0) | (i == len(neu_batches)-2):
            print(neu_batches[i+1],"entries' term freuquency calculated")
        i += 1


    pos_batches = np.linspace(9001,13500,100).astype(int)
    i=0
    pos_tf = []
    while i < len(pos_batches)-1:
        batch_result = np.sum(document_matrix[pos_batches[i]:pos_batches[i+1]].toarray(),axis=0)
        pos_tf.append(batch_result)
        if (i % 10 == 0) | (i == len(pos_batches)-2):
            print(pos_batches[i+1],"entries' term freuquency calculated")
        i += 1


    # Now we have positives and negatives 
    term_freq_df.columns = ['negative', 'neutral', 'positive']
    term_freq_df['total'] = term_freq_df['negative']+ term_freq_df['neutral'] + term_freq_df['positive']
    term_freq_df.sort_values(by='total', ascending=False).iloc[:10]



    #term_freq_df2 = term_freq_df
    term_freq_df2 =  term_freq_df

    term_freq_df2['pos_rate'] = term_freq_df2['positive'] * 1./term_freq_df2['total']
    term_freq_df2.sort_values(by='pos_rate', ascending=False).iloc[:10]


    term_freq_df2['pos_freq_pct'] = term_freq_df2['positive'] * 1./term_freq_df2['positive'].sum()
    term_freq_df2.sort_values(by='pos_freq_pct', ascending=False).iloc[:10]

    from scipy.stats import hmean

    term_freq_df2['pos_hmean'] = term_freq_df2.apply(lambda x: (hmean([x['pos_rate'], x['pos_freq_pct']])                                                               if x['pos_rate'] > 0 and x['pos_freq_pct'] > 0 else 0), axis=1)
                                                           
    term_freq_df2.sort_values(by='pos_hmean', ascending=False).iloc[:10]



    from scipy.stats import norm
    from scipy.stats import gmean

    #from scipy.stats import *



    def normcdf(x):
        return norm.cdf(x, x.mean(), x.std())


    term_freq_df2['pos_rate_normcdf'] = normcdf(term_freq_df2['pos_rate'])

    term_freq_df2['pos_freq_pct_normcdf'] = normcdf(term_freq_df2['pos_freq_pct'])

    term_freq_df2['pos_normcdf_hmean'] = gmean([term_freq_df2['pos_rate_normcdf'], term_freq_df2['pos_freq_pct_normcdf']])


    # Now the pos_rate_normcdf is the column to be shown for Super words



    term_freq_df2['neg_rate'] = term_freq_df2['negative'] * 1./term_freq_df2['total']

    term_freq_df2['neg_freq_pct'] = term_freq_df2['negative'] * 1./term_freq_df2['negative'].sum()

    term_freq_df2['neg_hmean'] = term_freq_df2.apply(lambda x: (hmean([x['neg_rate'], x['neg_freq_pct']]) if x['neg_rate'] > 0 and x['neg_freq_pct'] > 0 else 0), axis=1)
                                                           
    term_freq_df2['neg_rate_normcdf'] = normcdf(term_freq_df2['neg_rate'])

    term_freq_df2['neg_freq_pct_normcdf'] = normcdf(term_freq_df2['neg_freq_pct'])

    term_freq_df2['neg_normcdf_hmean'] = gmean([term_freq_df2['neg_rate_normcdf'], term_freq_df2['neg_freq_pct_normcdf']])



    # Good 
    term_freq_df2['neu_rate'] = term_freq_df2['neutral'] * 1./term_freq_df2['total']

    term_freq_df2['neu_freq_pct'] = term_freq_df2['neutral'] * 1./term_freq_df2['neutral'].sum()

    term_freq_df2['neu_hmean'] = term_freq_df2.apply(lambda x: (hmean([x['neu_rate'], x['neu_freq_pct']]) if x['neu_rate'] > 0 and x['neu_freq_pct'] > 0 else 0), axis=1)

    term_freq_df2['neu_rate_normcdf'] = normcdf(term_freq_df2['neu_rate'])

    term_freq_df2['neu_freq_pct_normcdf'] = normcdf(term_freq_df2['neu_freq_pct'])

    term_freq_df2['neu_normcdf_hmean'] = gmean([term_freq_df2['neu_rate_normcdf'], term_freq_df2['neu_freq_pct_normcdf']])





    term_freq_df2['word'] = words

    top_50_pos = term_freq_df2.sort_values(by='pos_normcdf_hmean',ascending=False).iloc[:50]
    top_50_neg = term_freq_df2.sort_values(by='neg_normcdf_hmean', ascending=False).iloc[:50]
    top_50_neu = term_freq_df2.sort_values(by='neu_normcdf_hmean', ascending=False).iloc[:50]

    pos_labels = top_50_pos['word']
    pos_values = top_50_pos['positive']

    neu_labels = top_50_neu['word']
    neu_values = top_50_neu['neutral']

    neg_labels = top_50_neg['word']
    neg_values = top_50_neg['negative']


    top_300_pos = term_freq_df2.sort_values(by='pos_normcdf_hmean',ascending=False).iloc[:300]
    top_300_neg = term_freq_df2.sort_values(by='neg_normcdf_hmean', ascending=False).iloc[:300]
    top_300_neu = term_freq_df2.sort_values(by='neu_normcdf_hmean', ascending=False).iloc[:300]

    top_300_pos.to_excel(target + "/top_300_pos.xlsx")
    top_300_neg.to_excel(target + "/top_300_neg.xlsx")
    top_300_neu.to_excel(target + "/top_300_neu.xlsx")


    sentiment_fit2 = tg_pipeline.fit(x_train, y_train)
    y_pred_test = sentiment_fit2.predict(x_test)

    # end of zipfs law tutorial
    # Showing the data as red when they are wrong predictions
    indexes = []
    for i in range(0, len(result_1.polarity)):
        indexes.append(i)

    result_1.index = indexes

    test_data_len = len(result_1)

    wrong_pred = []
    for i in result_1.index:
        if y_pred_test[i] != result_1.polarity.iloc[i]:
            wrong_pred.append(i)

    wrong_class = 'WrongPred alert alert-danger'




    #target = os.path.join(APP_ROOT, 'models/' + current_user.username + '/')
    #print(target)


    #if not os.path.isdir(target):
    #    os.mkdir(target)
    #with open(target + 'sentiment_fit.pkl', 'wb') as f:
    #    pickle.dump(sentiment_fit2, f)
    #with open(target + 'tg_cvec.pkl', 'wb') as f:
    #    pickle.dump(tg_cvec, f)



    return render_template('results_dareja2.html', 
        df = df, 
        accuracy_avg = accuracy_avg, 
        count_all = count_all, 
        count_pos = count_pos,
        count_neu = count_neu,  
        count_neg = count_neg,
        feature_result_wosw = feature_result_wosw, 
        ug_labels = ug_labels, ug_values = ug_values, 
        ug_wocsw_labels = ug_wocsw_labels, 
        ug_wocsw_values = ug_wocsw_values, 
        ug_wosw_labels = ug_wosw_labels, 
        ug_wosw_values = ug_wosw_values, 
        bg_values = bg_values, tg_values = tg_values, 
        r_t_confusion = r_t_confusion, 
        r_t_class_report = r_t_class_report, 
        r_v_confusion = r_v_confusion, 
        r_v_class_report = r_v_class_report, 
        ugt_values = ugt_values, 
        bgt_values = bgt_values, 
        tgt_values = tgt_values, 
        comparing_algorithms = comparing_algorithms, 
        top_50_neg = top_50_neg, 
        top_50_neu = top_50_neu, 
        top_50_pos = top_50_pos, 
        result_1 = result_1, 
        result_2 = result_2    , 
        pos_labels = pos_labels,
        pos_values = pos_values,
        neg_labels = neg_labels,
        neg_values = neg_values, 
        neu_labels = neu_labels,
        neu_values = neu_values, 
        wrong_pred = wrong_pred, 
        wrong_class = wrong_class, 
        test_data_len = test_data_len, 
        y_pred_test = y_pred_test, 
        sentiment_fit2 = sentiment_fit2, 
        tg_cvec = tg_cvec, 
        form = form
        )






@analyse.route('/results_test_dareja', methods = ['POST'])
def results_test_dareja():
    if not current_user.is_authenticated:
        flash('You should be logged in to acced this page', 'info')
        next_page =  request.endpoint
        return redirect(url_for('users.login', next = next_page))
    target = os.path.join(APP_ROOT, 'testxlsx/')
    print(target)

    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist("file"):
        print(file)
        filename = file.filename
        destination = "/".join([target, filename])
        print(destination)
        file.save(destination)
        df2 = pd.read_excel(os.path.join(APP_ROOT, 'testxlsx/'+filename))
        df_data2 = df2[["CONTENT", "CLASS"]]
        df_x2 = df_data2['CONTENT']
        test_x = df_x2
        df_y2 = df_data2.CLASS
        y_test = df_data2['CLASS']
        df_test = df_data2

    #df2 = pd.read_excel(os.path.join(APP_ROOT, 'testxlsx/HTL_test.xlsx'))
    #df_data2 = df2[["CONTENT", "CLASS"]]
    #df_x2 = df_data2['CONTENT']
    #test_x = df_x2
    #df_y2 = df_data2.CLASS
    #df_test = df_data2


    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import CountVectorizer
    import pickle
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    lr = LogisticRegression()



    from sklearn.externals import joblib 


    # Naive Bayes Classifier 
    from sklearn.naive_bayes import MultinomialNB

    ### Load and Test
    ytb_model = open("models/dareja/tg_cvec.pkl", "rb")
    tg_cvec = joblib.load(ytb_model)

    ### Load and Test
    ytb_model = open("models/dareja/sentiment_fit.pkl", "rb")
    sentiment_fit = joblib.load(ytb_model)

    tg_pipeline = Pipeline([
        ('vectorizer', tg_cvec),
        ('classifier', lr)
    ])

    s2 = pd.Series(df_x2)
    my_prediction = sentiment_fit.predict(s2)
    

    accuracy = accuracy_score(y_test, my_prediction)

    conmat = np.array(confusion_matrix(y_test, my_prediction, labels=[0,4]))
    confusion = pd.DataFrame(conmat, index=['negative', 'positive'],
                         columns=['predicted_negative','predicted_positive'])


    my_comments = []
    test_preds = []
    for i in range(len(test_x)):
        if my_prediction[i] == 4: 
            my_prediction_cat = 'Super'
        else:
            my_prediction_cat = 'Bad'
        test_preds.append(my_prediction_cat)
        my_comments.append(test_x[i])
        result_list = pd.DataFrame(
            {'Commentaire': my_comments,
             'Sentiment': test_preds
            })
        n_pos_neg = result_list['Sentiment'].value_counts()
        n_neg = len(result_list[result_list['Sentiment'] == 'Bad'])
        n_pos = len(result_list[result_list['Sentiment'] == 'Super'])
        n_total = n_pos + n_neg



    return render_template('results_test_dareja.html', 
        prediction = my_prediction, 
        result_list = result_list, 
        n_total = n_total, 
        n_neg = n_neg, 
        n_pos = n_pos, 
        my_comments = my_comments, 
        accuracy = accuracy,
        confusion = confusion
        
        )






def get_models():
    test_path = os.path.join(APP_ROOT, 'models\\')
    dirss = os.listdir( test_path )
    folders = []

    for dirs in dirss:
        if os.path.isdir(test_path + '\\' + dirs):
            folders.append(dirs)
    return folders



def get_models2():
    test_path = os.path.join(APP_ROOT, 'models\\' + current_user.username)
    dirss = os.listdir( test_path )
    folders = []

    for dirs in dirss:
        if os.path.isdir(test_path + '\\' + dirs):
            folders.append(dirs)
    return folders


#endpoint for search
@analyse.route('/search', methods=['GET', 'POST'])
def search():
    # redirect user if he is not logged in
    if not current_user.is_authenticated:
        flash('You should be logged in to acced this page', 'info')
        next_page =  request.endpoint
        return redirect(url_for('users.login', next = next_page))
    # get user id to be used in retrieving models creatd by the user.
    user_id = current_user.id
    # this line can be removed
    folders = get_models()
    # retrive models from the database
    cursor.execute("""SELECT name, type FROM `models` WHERE user_id = %s OR user_id = 1""", (user_id,))
    my_models = cursor.fetchall()
    models = []
    for i in my_models:
        models.append(i[0])
    return render_template('search.html', title='Search Comments', 
        folders = folders, 
        models = models, 
        user_id = user_id
        )

#endpoint for search
@analyse.route('/search_results', methods=['GET', 'POST'])
def search_results():
    # redirect user if not logged in, and if no post method was detected prior to accessing this page 
    if not current_user.is_authenticated:
        flash('You should be logged in to acced this page', 'info')
        next_page =  request.endpoint
        return redirect(url_for('users.login', next = next_page))
    if request.method != "POST":
        flash("You don't have permission accessing this page", 'info')
        return redirect(url_for('main.dashboard'))

    # declare two global variables
    global new_df, results_df
    #test_path = os.path.join(APP_ROOT, 'models\\')
    #dirss = os.listdir( test_path )
    # import important libraries for this page 
    from nltk.tokenize import sent_tokenize, word_tokenize
    from collections import Counter
    import re
    
    #folders = get_models()

    # retrieve models created by the user 
    user_id = current_user.id
    cursor.execute("""SELECT name, type FROM `models` WHERE user_id = %s OR user_id = 1""", (user_id,))
    my_models = cursor.fetchall()
    models = []
    for i in my_models:
        models.append(i[0])

    # get the user inputs
    if request.method == "POST":
        commenter = request.form['commenter']
        comment = request.form['comment']
        startdate = request.form['startdate']
        enddate = request.form['enddate']
        operator = request.form['operator']
        model = request.form['model']
        #start = datetime.datetime(start)
        #end = datetime.datetime(end)
        startdate = startdate + ' 00:00:00'
        enddate = enddate + ' 00:00:00'
        date_str = type(startdate)
        # select all commnets by the user input details (keyword, sector, startdate, enddate, )
        cursor.execute("SELECT nomCom, commentaire from commentaires WHERE nomCom LIKE %s AND commentaire LIKE %s AND idSocialPage = %s AND cast(dateCom as date) BETWEEN %s AND %s", ("%" + commenter + "%", "%" + comment + "%", operator, startdate, enddate))
        conn.commit()
        data = cursor.fetchall()
        type_data = type(data)
        new_df = pd.DataFrame(list(data), columns = ['commenter', 'comment'])
        
        # clean the retrieved data
        new_clean = clean_comments(df_content = new_df, column_name = 'comment', stopword = stopwords)


        # get the sentiment analysis of the retrived data .
        results_df, results_percents, results_df_pos, results_df_neu, results_df_neg, my_prediction = get_sentiment_analysis2(table = new_clean, model = model)
        new_df = results_df
        new_df_length = len(results_df)

        # (word counts graph) get the wordds with their counts for negative, neutral and positive
        # (wordcloud graph) get the corpus for negative, neutral, and positive comments.
        neg_results = results_df[results_df.Polarity == 'Bad']
        results_count_neg = len(neg_results)
        for_wordcloud_neg = ' '.join(map(str, neg_results['Comment']))
        all_words = word_tokenize(for_wordcloud_neg)
        counts = Counter(all_words)
        neg_counts = pd.DataFrame(list(counts.items()), columns = ['word', 'freq'] )  
        neg_counts = neg_counts.sort_values(by='freq', ascending=False).iloc[:20]

        neu_results = results_df[results_df.Polarity == 'Good']
        results_count_neu = len(neu_results)
        for_wordcloud_neu = ' '.join(map(str, neu_results['Comment']))
        all_words = word_tokenize(for_wordcloud_neu)
        counts = Counter(all_words)
        neu_counts = pd.DataFrame(list(counts.items()), columns = ['word', 'freq'] )     
        neu_counts = neu_counts.sort_values(by='freq', ascending=False).iloc[:20]

        pos_results = results_df[results_df.Polarity == 'Super']
        results_count_pos = len(pos_results)
        for_wordcloud_pos = ' '.join(map(str, pos_results['Comment']))
        all_words = word_tokenize(for_wordcloud_pos)
        counts = Counter(all_words)
        pos_counts = pd.DataFrame(list(counts.items()), columns = ['word', 'freq'] )  
        pos_counts = pos_counts.sort_values(by='freq', ascending=False).iloc[:20]

        # get telecom topics (as defined in get_telecom_topics function)
        telecom_counts, all_cats = get_telecom_topics(results_df = results_df)


        all_cats_str = []
        for elem in all_cats:
            all_cats_str.append(', '.join(elem))

        cats_length = len(all_cats_str)

        type_excel = type(results_df)

        # get number of comments in each polarity (for sentiment Polarity Numbers graph) 
        results_info = {
            'results_count_neg': results_count_neg, 
            'results_count_neu': results_count_neu,
            'results_count_pos': results_count_pos
        }
        return render_template('search_results.html', title='Search Results', 
            new_df = new_df,
            commenter = commenter, 
            comment = comment,
            type_data = type_data,
            my_prediction = my_prediction,
            new_df_length = new_df_length,
            startdate = startdate, 
            operator = operator, 
            enddate = enddate, 
            date_str = date_str, 
            models = models,
            results_df = results_df, 
            for_wordcloud_neg = for_wordcloud_neg,
            for_wordcloud_pos = for_wordcloud_pos, 
            for_wordcloud_neu = for_wordcloud_neu, 
            neg_counts = neg_counts,
            neu_counts = neu_counts, 
            pos_counts = pos_counts,
            pos_percentage = results_percents['pos_percentage'], 
            neu_percentage = results_percents['neu_percentage'], 
            neg_percentage = results_percents['neg_percentage'], 
            cats_length = cats_length, 
            telecom_counts = telecom_counts,
            all_cats_str = all_cats_str, 
            neg_count_internet = telecom_counts['neg_count_internet'],
            neg_count_reseau = telecom_counts['neg_count_reseau'],
            neg_count_offre = telecom_counts['neg_count_offre'],
            neg_count_appelle = telecom_counts['neg_count_appelle'],
            neg_count_other = telecom_counts['neg_count_other'],
            neu_count_internet = telecom_counts['neu_count_internet'],
            neu_count_reseau = telecom_counts['neu_count_reseau'],
            neu_count_offre = telecom_counts['neu_count_offre'],
            neu_count_appelle = telecom_counts['neu_count_appelle'],
            neu_count_other = telecom_counts['neu_count_other'],
            pos_count_internet = telecom_counts['pos_count_internet'],
            pos_count_reseau = telecom_counts['pos_count_reseau'],
            pos_count_offre = telecom_counts['pos_count_offre'],
            pos_count_appelle = telecom_counts['pos_count_appelle'],
            pos_count_other = telecom_counts['pos_count_other'],
            count_internet = telecom_counts['count_internet'],
            count_reseau = telecom_counts['count_reseau'],
            count_offre = telecom_counts['count_offre'],
            count_appelle = telecom_counts['count_appelle'],
            count_other = telecom_counts['count_other'],
            results_df_pos = results_df_pos, 
            type_excel = type_excel, 
            new_clean = new_clean, 
            stopwords = stopwords, 
            model = model, 
            results_info = results_info )

    else: 
        flash("There were an error processing your data, please try again.", 'danger')

    return render_template('search_results.html')







#endpoint for search_articles
@analyse.route('/search_articles', methods=['GET', 'POST'])
def search_articles():
    if not current_user.is_authenticated:
        flash('You should be logged in to acced this page', 'info')
        next_page =  request.endpoint
        return redirect(url_for('users.login', next = next_page))
    folders = get_models()
    return render_template('search_articles.html', title='Search Comments', 
        folders = folders)

#endpoint for search
@analyse.route('/search_articles_results', methods=['GET', 'POST'])
def search_articles_results():
    if not current_user.is_authenticated:
        flash('You should be logged in to acced this page', 'info')
        next_page =  request.endpoint
        return redirect(url_for('users.login', next = next_page))
    if request.method != "POST":
        flash("You don't have permission accessing this page", 'info')
        return redirect(url_for('main.dashboard'))

    global new_df, results_df
    test_path = os.path.join(APP_ROOT, 'models\\' + current_user.username)
    dirss = os.listdir( test_path )
    folders = []

    for dirs in dirss:
        if os.path.isdir(test_path + '\\' + dirs):
            folders.append(dirs)
    if request.method == "POST":
        Titrearticle = request.form['Titrearticle']
        resume = request.form['resume']
        startdate = request.form['startdate']
        enddate = request.form['enddate']
        langue = request.form['langue']
        model = request.form['model']
        #start = datetime.datetime(start)
        #end = datetime.datetime(end)
        startdate = startdate + ' 00:00:00'
        enddate = enddate + ' 00:00:00'
        date_str = type(startdate)
        # search by author or book
        cursor.execute("SELECT Titrearticle, resume from article WHERE Titrearticle LIKE %s AND resume LIKE %s AND cast(datesaisie as date) BETWEEN %s AND %s", ("%" + Titrearticle + "%", "%" + resume + "%", startdate, enddate))
        conn.commit()
        data = cursor.fetchall()
        type_data = type(data)
        new_df = pd.DataFrame(list(data), columns = ['Titrearticle', 'resume'])
        
 
        #all_clean = []
        import re
        #clean = re.compile('<.*?>')
        #for i in range(0, len(all_comments)):
        #    all_comments[i] = re.sub(clean, ' ', all_comments[i])
        #    all_comments[i] = all_comments[i].replace('\n',' ')
        #    all_comments[i] = all_comments[i].replace('\\n',' ')
        #    all_comments[i] = all_comments[i].replace('\t',' ')
        #    all_comments[i] = all_comments[i].replace('\r',' ')
        #    all_comments[i] = all_comments[i].replace('\'',' ')


        
        return render_template('search_articles_results.html', title='Search Results', 
            new_df = new_df)

    else: 
        flash("There were an error processing your data, please try again.", 'danger')

    return render_template('search_articles_results.html')










@analyse.route('/download_excel', methods = ['GET', 'POST'])
def download_excel():
    if not current_user.is_authenticated:
        flash('You should be logged in to acced this page', 'info')
        next_page =  request.endpoint
        return redirect(url_for('users.login', next = next_page))
    #Create DF
    #df = pd.concat([results_df['Comment']], axis=1)
    df = results_df
    #Convert DF
    strIO = io.BytesIO()
    excel_writer = pd.ExcelWriter(strIO)
    df.to_excel(excel_writer, sheet_name="sheet2")
    excel_writer.save()
    excel_data = strIO.getvalue()
    strIO.seek(0)

    return send_file(strIO,
                     attachment_filename='polarity-fb-results.xlsx',
                     as_attachment=True)
    


















































































@analyse.route('/results_dareja', methods = ['POST'])
def results_dareja():
    if not current_user.is_authenticated:
        flash('You should be logged in to acced this page', 'info')
        next_page =  request.endpoint
        return redirect(url_for('users.login', next = next_page))

    if request.method != "POST":
        flash("You don't have permission accessing this page", 'info')
        return redirect(url_for('main.dashboard'))
    global tg_cvec, sentiment_fit2
    form = SaveModel()
    #df = pd.read_csv('YoutubeSpamMergedData.csv')
    target = os.path.join(APP_ROOT, 'trainingcsv/')
    print(target)

    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist("file"):
        print(file)
        filename = file.filename
        destination = "/".join([target, filename])
        print(destination)
        file.save(destination)
        df = pd.read_excel(os.path.join(APP_ROOT, 'trainingcsv/'+filename))
        df['text'] = df['text'].astype(str)
        df_data = df[["text", "polarity"]]
        df_x = df_data['text']
        df_y = df_data.polarity

    df_n = df[df.polarity == -1]
    df_p = df[df.polarity == 1]
    df_n = df_n.assign(polarity= 0)
    df_p = df_p.assign(polarity= 4)
    df3 = df_n.append(df_p, ignore_index=True)
    # NOW , it has only POSITIVES AND NEGATIVES 
    df = df3

    my_df = df
    df_neg = my_df[my_df.polarity == 0][0:4500]
    df_pos = my_df[my_df.polarity == 4][0:4500]
    frames = [df_neg, df_pos]
    result = pd.concat(frames, ignore_index=True)
    # THIS VARIABLE HAS ALL THE TRAINING DATA SEPARATED 
    result

    df.head()

    x = result.text
    y = result.polarity
    from sklearn.model_selection import train_test_split

    SEED = 2000

    # assigning x and y directly as the train set
    x_train = x
    y_train = y 

    # setting the x_test and y_test 
    # Testing Sample
    df_neg_1 = df[df.polarity == 0][4501:5300]
    df_pos_1 = df[df.polarity == 4][4501:5300]
    frames_1 = [df_neg_1, df_pos_1]
    result_1 = pd.concat(frames_1)
    x_test = result_1.text
    y_test = result_1.polarity

    #Validation sample
    df_neg_2 = df[df.polarity == 0][5301:5987]
    df_pos_2 = df[df.polarity == 4][5301:5987]
    frames_2 = [df_neg_2, df_pos_2]
    result_2 = pd.concat(frames_2)
    x_validation = result_2.text
    y_validation = result_2.polarity



    print ("Train set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".format(len(x_train),
        (len(x_train[y_train == 0]) / (len(x_train)*1.))*100,
        (len(x_train[y_train == 4]) / (len(x_train)*1.))*100) )


    print ("Validation set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".format(len(x_validation),
        (len(x_validation[y_validation == 0]) / (len(x_validation)*1.))*100,
        (len(x_validation[y_validation == 4]) / (len(x_validation)*1.))*100) )

    print ("Test set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".format(len(x_test),
        (len(x_test[y_test == 0]) / (len(x_test)*1.))*100,
        (len(x_test[y_test == 4]) / (len(x_test)*1.))*100) )


    from textblob import TextBlob
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import classification_report, confusion_matrix
    # Feature extraction
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from time import time




    def accuracy_summary(pipeline, x_train, y_train, x_test, y_test):
        if len(x_test[y_test == 0]) / (len(x_test)*1.) > 0.5:
            null_accuracy = len(x_test[y_test == 0]) / (len(x_test)*1.)
        else:
            null_accuracy = 1. - (len(x_test[y_test == 0]) / (len(x_test)*1.))
        t0 = time()
        sentiment_fit = pipeline.fit(x_train, y_train)
        y_pred = sentiment_fit.predict(x_test)
        train_test_time = time() - t0
        accuracy = accuracy_score(y_test, y_pred)
        print ("null accuracy: {0:.2f}%".format(null_accuracy*100) )
        print ("accuracy score: {0:.2f}%".format(accuracy*100) )
        if accuracy > null_accuracy:
            print ("model is {0:.2f}% more accurate than null accuracy".format((accuracy-null_accuracy)*100) )
        elif accuracy == null_accuracy:
            print ("model has the same accuracy with the null accuracy")
        else:
            print ("model is {0:.2f}% less accurate than null accuracy".format((null_accuracy-accuracy)*100) )
        print ("train and test time: {0:.2f}s".format(train_test_time) )
        print ("-"*80 )
        return accuracy, train_test_time


    cvec = CountVectorizer()
    # get the number of feature names , but first we fit the text
    cvec.fit(x)
    len(cvec.get_feature_names())
    num_features = len(cvec.get_feature_names())
    lr = LogisticRegression()
    n_features = np.arange(1000,num_features,1000)


    # get the stemmer function, this is optional 
    # with stemming words
    import nltk
    from nltk.stem.isri import ISRIStemmer
    stemmer = ISRIStemmer()
    analyzer = CountVectorizer().build_analyzer()

    def stemmed_words(doc):
        return (stemmer.stem(w) for w in analyzer(doc))
    # remove the above code as it is used in another function(if any errors)


    def nfeature_accuracy_checker(vectorizer=cvec, n_features=n_features, stop_words=None, ngram_range=(1, 1), classifier=lr):
        result = []
        print (classifier)
        print ("\n")
        for n in n_features:
            vectorizer.set_params(stop_words=stop_words, max_features=n, ngram_range=ngram_range)
            checker_pipeline = Pipeline([
                ('vectorizer', vectorizer),
                ('classifier', classifier)
            ])
            print ("Validation result for {} features".format(n) )
            nfeature_accuracy,tt_time = accuracy_summary(checker_pipeline, x_train, y_train, x_validation, y_validation)
            result.append((n,nfeature_accuracy,tt_time))
        return result


    my_stop_words = ['selmane', 'aissa', 'brahim']

    print ("RESULT FOR UNIGRAM WITHOUT STOP WORDS\n")
    feature_result_wosw = nfeature_accuracy_checker(stop_words = 'english') 

    print ("RESULT FOR UNIGRAM WITH STOP WORDS\n")
    feature_result_ug = nfeature_accuracy_checker() 

    print ("RESULT FOR UNIGRAM WITHOUT CUSTOM STOP WORDS (Top 10 frequent words)\n" )
    feature_result_wocsw = nfeature_accuracy_checker(stop_words=my_stop_words) 


    nfeatures_plot_ug = pd.DataFrame(feature_result_ug,columns=['nfeatures','validation_accuracy','train_test_time'])
    nfeatures_plot_ug_wocsw = pd.DataFrame(feature_result_wocsw,columns=['nfeatures','validation_accuracy','train_test_time'])
    nfeatures_plot_ug_wosw = pd.DataFrame(feature_result_wosw,columns=['nfeatures','validation_accuracy','train_test_time'])

    plt.figure(figsize=(8,6))
    plt.plot(nfeatures_plot_ug.nfeatures, nfeatures_plot_ug.validation_accuracy, label='with stop words')
    plt.plot(nfeatures_plot_ug_wosw.nfeatures, nfeatures_plot_ug_wosw.validation_accuracy,label='without stop words')
    plt.plot(nfeatures_plot_ug_wocsw.nfeatures, nfeatures_plot_ug_wocsw.validation_accuracy,label='without custom stop words')
    plt.title("Without stop words VS With stop words (Unigram): Accuracy")
    plt.xlabel("Number of features")
    plt.ylabel("Validation set accuracy")
    plt.legend()

    ug_labels = nfeatures_plot_ug.nfeatures
    ug_values = nfeatures_plot_ug.validation_accuracy

    ug_wocsw_labels = nfeatures_plot_ug_wocsw.nfeatures
    ug_wocsw_values = nfeatures_plot_ug_wocsw.validation_accuracy

    ug_wosw_labels = nfeatures_plot_ug_wosw.nfeatures
    ug_wosw_values = nfeatures_plot_ug_wosw.validation_accuracy


    # Showing the tests for bigrams and trigrams 
    print ("RESULT FOR BIGRAM WITH STOP WORDS\n")
    feature_result_bg = nfeature_accuracy_checker(ngram_range=(1, 2))

    print ("RESULT FOR TRIGRAM WITH STOP WORDS\n")
    feature_result_tg = nfeature_accuracy_checker(ngram_range=(1, 3))


    nfeatures_plot_tg = pd.DataFrame(feature_result_tg,columns=['nfeatures','validation_accuracy','train_test_time'])
    nfeatures_plot_bg = pd.DataFrame(feature_result_bg,columns=['nfeatures','validation_accuracy','train_test_time'])
    nfeatures_plot_ug = pd.DataFrame(feature_result_ug,columns=['nfeatures','validation_accuracy','train_test_time'])

    plt.figure(figsize=(8,6))
    plt.plot(nfeatures_plot_tg.nfeatures, nfeatures_plot_tg.validation_accuracy,label='trigram')
    plt.plot(nfeatures_plot_bg.nfeatures, nfeatures_plot_bg.validation_accuracy,label='bigram')
    plt.plot(nfeatures_plot_ug.nfeatures, nfeatures_plot_ug.validation_accuracy, label='unigram')
    plt.title("N-gram(1~3) test result : Accuracy")
    plt.xlabel("Number of features")
    plt.ylabel("Validation set accuracy")
    plt.legend()
    plt.show()

    bg_values = nfeatures_plot_bg.validation_accuracy
    tg_values = nfeatures_plot_tg.validation_accuracy

    def train_test_and_evaluate(pipeline, x_train, y_train, x_test, y_test):
        if len(x_test[y_test == 0]) / (len(x_test)*1.) > 0.5:
            null_accuracy = len(x_test[y_test == 0]) / (len(x_test)*1.)
        else:
            null_accuracy = 1. - (len(x_test[y_test == 0]) / (len(x_test)*1.))
        sentiment_fit = pipeline.fit(x_train, y_train)
        y_pred = sentiment_fit.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        conmat = np.array(confusion_matrix(y_test, y_pred, labels=[0,4]))
        confusion = pd.DataFrame(conmat, index=['negative', 'positive'],
                             columns=['predicted_negative','predicted_positive'])
        class_report = classification_report(y_test, y_pred, target_names=['negative','positive'], output_dict=True)
        class_report_df = pd.DataFrame(class_report).transpose()
        print ("null accuracy: {0:.2f}%".format(null_accuracy*100) )
        print ("accuracy score: {0:.2f}%".format(accuracy*100) )
        if accuracy > null_accuracy:
            print ("model is {0:.2f}% more accurate than null accuracy".format((accuracy-null_accuracy)*100) )
        elif accuracy == null_accuracy:
            print ("model has the same accuracy with the null accuracy" )
        else:
            print ("model is {0:.2f}% less accurate than null accuracy".format((null_accuracy-accuracy)*100) )
        print ("-"*80 )
        print ("Confusion Matrix\n" )
        print (confusion)
        print ("-"*80 )
        print ("Classification Report\n" )
        print (classification_report(y_test, y_pred, target_names=['negative','positive'],  output_dict=True) )
        return confusion , class_report_df


    tg_cvec = CountVectorizer(max_features=num_features,ngram_range=(1, 3))
    tg_pipeline = Pipeline([
            ('vectorizer', tg_cvec),
            ('classifier', lr)
        ])
    r_t_confusion, r_t_class_report = train_test_and_evaluate(tg_pipeline, x_train, y_train, x_test, y_test)

    r_v_confusion, r_v_class_report = train_test_and_evaluate(tg_pipeline, x_train, y_train,x_validation, y_validation )

    # getting the number of true positives and negatives to show on the website 

    true_negative = r_t_confusion["predicted_negative"][0]
    false_negative = r_t_confusion["predicted_negative"][1]

    false_positive = r_t_confusion["predicted_positive"][0]
    true_positive = r_t_confusion["predicted_positive"][1]
    count_pos = true_positive + false_positive
    count_neg = true_negative + false_negative
    count_all = count_pos + count_neg    
    accuracy_avg = r_t_class_report.precision[2]
    accuracy_avg = float("{0:.3f}".format(accuracy_avg))
    # there are some functions that have been used before, and 
    # i am testing on tf-idf , unigrams, bigrams, trigrams 
    # and printing the results on a graph
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from time import time


    def accuracy_summary(pipeline, x_train, y_train, x_test, y_test):
        if len(x_test[y_test == 0]) / (len(x_test)*1.) > 0.5:
            null_accuracy = len(x_test[y_test == 0]) / (len(x_test)*1.)
        else:
            null_accuracy = 1. - (len(x_test[y_test == 0]) / (len(x_test)*1.))
        t0 = time()
        sentiment_fit = pipeline.fit(x_train, y_train)
        y_pred = sentiment_fit.predict(x_test)
        train_test_time = time() - t0
        accuracy = accuracy_score(y_test, y_pred)
        print ("null accuracy: {0:.2f}%".format(null_accuracy*100) )
        print ("accuracy score: {0:.2f}%".format(accuracy*100) )
        if accuracy > null_accuracy:
            print ("model is {0:.2f}% more accurate than null accuracy".format((accuracy-null_accuracy)*100) )
        elif accuracy == null_accuracy:
            print ("model has the same accuracy with the null accuracy")
        else:
            print ("model is {0:.2f}% less accurate than null accuracy".format((null_accuracy-accuracy)*100) )
        print ("train and test time: {0:.2f}s".format(train_test_time) )
        print ("-"*80 )
        return (accuracy, train_test_time )

    cvec = CountVectorizer()
    lr = LogisticRegression()
    n_features = np.arange(1000,num_features,1000)

    def nfeature_accuracy_checker(vectorizer=cvec, n_features=n_features, stop_words=None, ngram_range=(1, 1), classifier=lr):
        result = []
        print (classifier)
        print ("\n")
        for n in n_features:
            vectorizer.set_params(stop_words=stop_words, max_features=n, ngram_range=ngram_range)
            checker_pipeline = Pipeline([
                ('vectorizer', vectorizer),
                ('classifier', classifier)
            ])
            print ("Validation result for {} features".format(n) )
            nfeature_accuracy,tt_time = accuracy_summary(checker_pipeline, x_train, y_train, x_validation, y_validation)
            result.append((n,nfeature_accuracy,tt_time))
        return result

    from sklearn.feature_extraction.text import TfidfVectorizer
    tvec = TfidfVectorizer()

    feature_result_ugt = nfeature_accuracy_checker(vectorizer=tvec)
    feature_result_bgt = nfeature_accuracy_checker(vectorizer=tvec,ngram_range=(1, 2))
    feature_result_tgt = nfeature_accuracy_checker(vectorizer=tvec,ngram_range=(1, 3))


    nfeatures_plot_tgt = pd.DataFrame(feature_result_tgt,columns=['nfeatures','validation_accuracy','train_test_time'])
    nfeatures_plot_bgt = pd.DataFrame(feature_result_bgt,columns=['nfeatures','validation_accuracy','train_test_time'])
    nfeatures_plot_ugt = pd.DataFrame(feature_result_ugt,columns=['nfeatures','validation_accuracy','train_test_time'])
    plt.figure(figsize=(8,6))
    plt.plot(nfeatures_plot_tgt.nfeatures, nfeatures_plot_tgt.validation_accuracy,label='trigram tfidf vectorizer',color='royalblue')
    plt.plot(nfeatures_plot_tg.nfeatures, nfeatures_plot_tg.validation_accuracy,label='trigram count vectorizer',linestyle=':', color='royalblue')
    plt.plot(nfeatures_plot_bgt.nfeatures, nfeatures_plot_bgt.validation_accuracy,label='bigram tfidf vectorizer',color='orangered')
    plt.plot(nfeatures_plot_bg.nfeatures, nfeatures_plot_bg.validation_accuracy,label='bigram count vectorizer',linestyle=':',color='orangered')
    plt.plot(nfeatures_plot_ugt.nfeatures, nfeatures_plot_ugt.validation_accuracy, label='unigram tfidf vectorizer',color='gold')
    plt.plot(nfeatures_plot_ug.nfeatures, nfeatures_plot_ug.validation_accuracy, label='unigram count vectorizer',linestyle=':',color='gold')
    plt.title("N-gram(1~3) test result : Accuracy")
    plt.xlabel("Number of features")
    plt.ylabel("Validation set accuracy")
    plt.legend()
    plt.show()

    ugt_values = nfeatures_plot_bgt.validation_accuracy
    bgt_values = nfeatures_plot_bgt.validation_accuracy
    tgt_values = nfeatures_plot_tgt.validation_accuracy



    # Comparing with other classifiers
    # Now , we compare using different algorithms:
    from sklearn.svm import LinearSVC
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.naive_bayes import MultinomialNB, BernoulliNB
    from sklearn.linear_model import RidgeClassifier
    from sklearn.linear_model import PassiveAggressiveClassifier
    from sklearn.linear_model import Perceptron
    from sklearn.neighbors import NearestCentroid
    from sklearn.feature_selection import SelectFromModel
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer

    names = ["Logistic Regression", "Linear SVC", "LinearSVC with L1-based feature selection","Multinomial NB", 
             "Bernoulli NB", "Ridge Classifier", "AdaBoost", "Perceptron","Passive-Aggresive", "Nearest Centroid"]
    classifiers = [
        LogisticRegression(),
        LinearSVC(),
        Pipeline([
      ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False))),
      ('classification', LinearSVC(penalty="l2"))]),
        MultinomialNB(),
        BernoulliNB(),
        RidgeClassifier(),
        AdaBoostClassifier(),
        Perceptron(),
        PassiveAggressiveClassifier(),
        NearestCentroid()
        ]
    zipped_clf = zip(names,classifiers)

    tvec = TfidfVectorizer()

    def classifier_comparator(vectorizer=tvec, n_features= num_features, stop_words=None, ngram_range=(1, 1), classifier=zipped_clf):
        result = []
        vectorizer.set_params(stop_words=stop_words, max_features=n_features, ngram_range=ngram_range)
        for n,c in classifier:
            checker_pipeline = Pipeline([
                ('vectorizer', vectorizer),
                ('classifier', c)
            ])
            print ("Validation result for {}".format(n) )
            print (c)
            clf_accuracy,tt_time = accuracy_summary(checker_pipeline, x_train, y_train, x_validation, y_validation)
            result.append((n,clf_accuracy,tt_time))
        return result

    comparing_algorithms = classifier_comparator(n_features=num_features,ngram_range=(1,3))



    comparing_algorithms = pd.DataFrame(comparing_algorithms, columns=['Model','Validation set accuracy', 'Train and test time'])









    # Getting The positives and Negatives 
    my_df = result


    # Creating the count vectorizer for later use 
    from sklearn.feature_extraction.text import CountVectorizer
    cvec = CountVectorizer()
    cvec.fit(my_df.text)

    # This will give you the number of feature names
    len(cvec.get_feature_names())
    # this above variable is important 


    # This will generate the terpm frequency for pos and neg and then combined 
    neg_doc_matrix = cvec.transform(my_df[my_df.polarity == 0].text)
    pos_doc_matrix = cvec.transform(my_df[my_df.polarity == 4].text)
    neg_tf = np.sum(neg_doc_matrix,axis=0)
    pos_tf = np.sum(pos_doc_matrix,axis=0)
    neg = np.squeeze(np.asarray(neg_tf))
    pos = np.squeeze(np.asarray(pos_tf))
    words = cvec.get_feature_names()
    term_freq_df = pd.DataFrame([neg,pos], columns = cvec.get_feature_names()).transpose()



    document_matrix = cvec.transform(my_df.text)
    my_df[my_df.polarity == 0].tail()

    neg_batches = np.linspace(0,4500,100).astype(int)
    i=0
    neg_tf = []
    while i < len(neg_batches)-1:
        batch_result = np.sum(document_matrix[neg_batches[i]:neg_batches[i+1]].toarray(),axis=0)
        neg_tf.append(batch_result)
        if (i % 10 == 0) | (i == len(neg_batches)-2):
            print(neg_batches[i+1],"entries' term freuquency calculated")
        i += 1


    pos_batches = np.linspace(4501,9000,100).astype(int)
    i=0
    pos_tf = []
    while i < len(pos_batches)-1:
        batch_result = np.sum(document_matrix[pos_batches[i]:pos_batches[i+1]].toarray(),axis=0)
        pos_tf.append(batch_result)
        if (i % 10 == 0) | (i == len(pos_batches)-2):
            print(pos_batches[i+1],"entries' term freuquency calculated")
        i += 1


    # Now we have positives and negatives 
    term_freq_df.columns = ['negative', 'positive']
    term_freq_df['total'] = term_freq_df['negative'] + term_freq_df['positive']
    term_freq_df.sort_values(by='total', ascending=False).iloc[:10]



    #term_freq_df2 = term_freq_df
    term_freq_df2 =  term_freq_df

    term_freq_df2['pos_rate'] = term_freq_df2['positive'] * 1./term_freq_df2['total']
    term_freq_df2.sort_values(by='pos_rate', ascending=False).iloc[:10]


    term_freq_df2['pos_freq_pct'] = term_freq_df2['positive'] * 1./term_freq_df2['positive'].sum()
    term_freq_df2.sort_values(by='pos_freq_pct', ascending=False).iloc[:10]

    from scipy.stats import hmean

    term_freq_df2['pos_hmean'] = term_freq_df2.apply(lambda x: (hmean([x['pos_rate'], x['pos_freq_pct']])                                                               if x['pos_rate'] > 0 and x['pos_freq_pct'] > 0 else 0), axis=1)
                                                           
    term_freq_df2.sort_values(by='pos_hmean', ascending=False).iloc[:10]



    from scipy.stats import norm
    from scipy.stats import gmean

    #from scipy.stats import *



    def normcdf(x):
        return norm.cdf(x, x.mean(), x.std())


    term_freq_df2['pos_rate_normcdf'] = normcdf(term_freq_df2['pos_rate'])

    term_freq_df2['pos_freq_pct_normcdf'] = normcdf(term_freq_df2['pos_freq_pct'])

    term_freq_df2['pos_normcdf_hmean'] = gmean([term_freq_df2['pos_rate_normcdf'], term_freq_df2['pos_freq_pct_normcdf']])


    # Now the pos_rate_normcdf is the column to be shown for Super words



    term_freq_df2['neg_rate'] = term_freq_df2['negative'] * 1./term_freq_df2['total']

    term_freq_df2['neg_freq_pct'] = term_freq_df2['negative'] * 1./term_freq_df2['negative'].sum()

    term_freq_df2['neg_hmean'] = term_freq_df2.apply(lambda x: (hmean([x['neg_rate'], x['neg_freq_pct']])                                                                if x['neg_rate'] > 0 and x['neg_freq_pct'] > 0                                                                else 0), axis=1)
                                                           
    term_freq_df2['neg_rate_normcdf'] = normcdf(term_freq_df2['neg_rate'])

    term_freq_df2['neg_freq_pct_normcdf'] = normcdf(term_freq_df2['neg_freq_pct'])

    term_freq_df2['neg_normcdf_hmean'] = gmean([term_freq_df2['neg_rate_normcdf'], term_freq_df2['neg_freq_pct_normcdf']])


    term_freq_df2['word'] = words

    top_50_pos = term_freq_df2.sort_values(by='pos_normcdf_hmean',ascending=False).iloc[:50]
    top_50_neg = term_freq_df2.sort_values(by='neg_normcdf_hmean', ascending=False).iloc[:50]

    pos_labels = top_50_pos['word']
    pos_values = top_50_pos['positive']

    neg_labels = top_50_neg['word']
    neg_values = top_50_neg['negative']



    sentiment_fit2 = tg_pipeline.fit(x_train, y_train)
    y_pred_test = sentiment_fit2.predict(x_test)

    # end of zipfs law tutorial
    # Showing the data as red when they are wrong predictions
    indexes = []
    for i in range(0, len(result_1.polarity)):
        indexes.append(i)

    result_1.index = indexes

    test_data_len = len(result_1)

    wrong_pred = []
    for i in result_1.index:
        if y_pred_test[i] != result_1.polarity.iloc[i]:
            wrong_pred.append(i)

    wrong_class = 'WrongPred alert alert-danger'




    target = os.path.join(APP_ROOT, 'models/' + current_user.username + '/')
    print(target)


    if not os.path.isdir(target):
        os.mkdir(target)
    with open(target + 'sentiment_fit.pkl', 'wb') as f:
        pickle.dump(sentiment_fit2, f)
    with open(target + 'tg_cvec.pkl', 'wb') as f:
        pickle.dump(tg_cvec, f)



    return render_template('results_dareja.html',
        df = df, 
        accuracy_avg = accuracy_avg, 
        count_all = count_all, 
        count_pos = count_pos, 
        count_neg = count_neg,
        feature_result_wosw = feature_result_wosw, 
        ug_labels = ug_labels, ug_values = ug_values, 
        ug_wocsw_labels = ug_wocsw_labels, 
        ug_wocsw_values = ug_wocsw_values, 
        ug_wosw_labels = ug_wosw_labels, 
        ug_wosw_values = ug_wosw_values, 
        bg_values = bg_values, tg_values = tg_values, 
        r_t_confusion = r_t_confusion, 
        r_t_class_report = r_t_class_report, 
        r_v_confusion = r_v_confusion, 
        r_v_class_report = r_v_class_report, 
        ugt_values = ugt_values, 
        bgt_values = bgt_values, 
        tgt_values = tgt_values, 
        comparing_algorithms = comparing_algorithms, 
        top_50_neg = top_50_neg, 
        top_50_pos = top_50_pos, 
        result_1 = result_1, 
        result_2 = result_2    , 
        pos_labels = pos_labels,
        pos_values = pos_values,
        neg_labels = neg_labels,
        neg_values = neg_values, 
        wrong_pred = wrong_pred, 
        wrong_class = wrong_class, 
        test_data_len = test_data_len, 
        y_pred_test = y_pred_test, 
        sentiment_fit2 = sentiment_fit2, 
        tg_cvec = tg_cvec, 
        form = form
        )






































@analyse.route('/results_dareja_graphs', methods = ['POST', 'GET'])
def results_dareja_graphs():
    if not current_user.is_authenticated:
        flash('You should be logged in to acced this page', 'info')
        next_page =  request.endpoint
        return redirect(url_for('users.login', next = next_page))

    if request.method != "POST":
        flash("You don't have permission accessing this page", 'info')
        return redirect(url_for('main.dashboard'))
    global tg_cvec, sentiment_fit2
    form = SaveModel()
    #df = pd.read_csv('YoutubeSpamMergedData.csv')
    target = os.path.join(APP_ROOT, 'trainingcsv/')
    print(target)

    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist("file"):
        print(file)
        filename = file.filename
        destination = "/".join([target, filename])
        print(destination)
        file.save(destination)
        df = pd.read_excel(os.path.join(APP_ROOT, 'trainingcsv/'+filename))
        df['text'] = df['text'].astype(str)
        #df['text'] = df['text'].replace('<emoji>',' emoji_start', regex = True)
        #df['text'] = df['text'].replace('</emoji>','emoji_end ', regex = True)
        emoji_dict = { "<emoji>": " ", "</emoji>" : " ",
          "\n": " ", "\t": " ", "\'": " ",
          "1f600": "GRINNING_FACE",
          "1f601": "GRINNING_FACE_WITH_SMILING_EYES",
          "1f602": "FACE_WITH_TEARS_OF_JOY",
          "1f603": "SMILING_FACE_WITH_OPEN_MOUTH",
          "1f604": "SMILING_FACE_WITH_OPEN_MOUTH_AND_SMILING_EYES",
          "1f605": "SMILING_FACE_WITH_OPEN_MOUTH_AND_COLD_SWEAT",
          "1f606": "SMILING_FACE_WITH_OPEN_MOUTH_AND_TIGHTLY-CLOSED_EYES",
          "1f607": "SMILING_FACE_WITH_HALO",
          "1f608": "SMILING_FACE_WITH_HORNS",
          "1f609": "WINKING_FACE",
          "1f60a": "SMILING_FACE_WITH_SMILING_EYES",
          "1f60b": "FACE_SAVOURING_DELICIOUS_FOOD",
          "1f60c": "RELIEVED_FACE",
          "1f60d": "SMILING_FACE_WITH_HEART-SHAPED_EYES",
          "1f60e": "SMILING_FACE_WITH_SUNGLASSES",
          "1f60f": "SMIRKING_FACE",
          "1f610": "NEUTRAL_FACE",
          "1f611": "EXPRESSIONLESS_FACE",
          "1f612": "UNAMUSED_FACE",
          "1f613": "FACE_WITH_COLD_SWEAT",
          "1f614": "PENSIVE_FACE",
          "1f615": "CONFUSED_FACE",
          "1f616": "CONFOUNDED_FACE",
          "1f617": "KISSING_FACE",
          "1f618": "FACE_THROWING_A_KISS",
          "1f619": "KISSING_FACE_WITH_SMILING_EYES",
          "1f61a": "KISSING_FACE_WITH_CLOSED_EYES",
          "1f61b": "FACE_WITH_STUCK-OUT_TONGUE",
          "1f61c": "FACE_WITH_STUCK-OUT_TONGUE_AND_WINKING_EYE",
          "1f61d": "FACE_WITH_STUCK-OUT_TONGUE_AND_TIGHTLY-CLOSED_EYES",
          "1f61e": "DISAPPOINTED_FACE",
          "1f61f": "WORRIED_FACE",
          "1f620": "ANGRY_FACE",
          "1f621": "POUTING_FACE",
          "1f622": "CRYING_FACE",
          "1f623": "PERSEVERING_FACE",
          "1f624": "FACE_WITH_LOOK_OF_TRIUMPH",
          "1f625": "DISAPPOINTED_BUT_RELIEVED_FACE",
          "1f626": "FROWNING_FACE_WITH_OPEN_MOUTH",
          "1f627": "ANGUISHED_FACE",
          "1f628": "FEARFUL_FACE",
          "1f629": "WEARY_FACE",
          "1f62a": "SLEEPY_FACE",
          "1f62b": "TIRED_FACE",
          "1f62c": "GRIMACING_FACE",
          "1f62d": "LOUDLY_CRYING_FACE",
          "1f62e": "FACE_WITH_OPEN_MOUTH",
          "1f62f": "HUSHED_FACE",
          "1f630": "FACE_WITH_OPEN_MOUTH_AND_COLD_SWEAT",
          "1f631": "FACE_SCREAMING_IN_FEAR",
          "1f632": "ASTONISHED_FACE",
          "1f633": "FLUSHED_FACE",
          "1f634": "SLEEPING_FACE",
          "1f635": "DIZZY_FACE",
          "1f636": "FACE_WITHOUT_MOUTH",
          "1f637": "FACE_WITH_MEDICAL_MASK" }
        df['text'] = df['text'].str.lower()
        for word, initial in emoji_dict.items():
            df['text'] = df['text'].replace(word.lower() , initial.lower(), regex = True)

        Text_Clean = []
        for i in df['text']:
            a = re.sub(r'(.)\1+', r'\1', i)  
            Text_Clean.append(a)

        df['text'] = Text_Clean
        df_data = df[["text", "polarity"]]
        df_x = df_data['text']
        df_y = df_data.polarity


    df_neg = df[df.polarity == -1]
    df_pos = df[df.polarity == 1]
    df_neu = df[df.polarity == 0]
    df_neg = df_neg.assign(polarity= 0)
    df_neu = df_neu.assign(polarity= 2)
    df_pos = df_pos.assign(polarity= 4)
    df3 = df_neg.append(df_neu, ignore_index=True)
    df3 = df3.append(df_pos, ignore_index=True)
    # NOW , it has only POSITIVES AND NEGATIVES 
    df = df3

    my_df = df
    df_neg = my_df[my_df.polarity == 0][0:4500]
    df_neu = my_df[my_df.polarity == 2][0:4500]
    df_pos = my_df[my_df.polarity == 4][0:4500]
    frames = [df_neg, df_neu, df_pos]
    result = pd.concat(frames, ignore_index=True)
    # THIS VARIABLE HAS ALL THE TRAINING DATA SEPARATED 
    result

    df.head()

    x = result.text
    y = result.polarity
    from sklearn.model_selection import train_test_split

    SEED = 2000


    # Getting The positives and Negatives 
    my_df = result


    # Creating the count vectorizer for later use 
    from sklearn.feature_extraction.text import CountVectorizer
    cvec = CountVectorizer()
    my_df['text'] = my_df.text.astype(str)
    cvec.fit(my_df.text)

    # This will give you the number of feature names
    len(cvec.get_feature_names())
    # this above variable is important 


    # This will generate the terpm frequency for pos and neg and then combined 
    neg_doc_matrix = cvec.transform(my_df[my_df.polarity == 0].text)
    neu_doc_matrix = cvec.transform(my_df[my_df.polarity == 2].text)
    pos_doc_matrix = cvec.transform(my_df[my_df.polarity == 4].text)
    neg_tf = np.sum(neg_doc_matrix,axis=0)
    neu_tf = np.sum(neu_doc_matrix,axis=0)
    pos_tf = np.sum(pos_doc_matrix,axis=0)
    neg = np.squeeze(np.asarray(neg_tf))
    neu = np.squeeze(np.asarray(neu_tf))
    pos = np.squeeze(np.asarray(pos_tf))
    words = cvec.get_feature_names()
    term_freq_df = pd.DataFrame([neg,neu,pos], columns = cvec.get_feature_names()).transpose()



    document_matrix = cvec.transform(my_df.text)
    my_df[my_df.polarity == 0].tail()

    neg_batches = np.linspace(0,4500,100).astype(int)
    i=0
    neg_tf = []
    while i < len(neg_batches)-1:
        batch_result = np.sum(document_matrix[neg_batches[i]:neg_batches[i+1]].toarray(),axis=0)
        neg_tf.append(batch_result)
        if (i % 10 == 0) | (i == len(neg_batches)-2):
            print(neg_batches[i+1],"entries' term freuquency calculated")
        i += 1


    neu_batches = np.linspace(4501,9000,100).astype(int)
    i=0
    neu_tf = []
    while i < len(neu_batches)-1:
        batch_result = np.sum(document_matrix[neu_batches[i]:neu_batches[i+1]].toarray(),axis=0)
        neu_tf.append(batch_result)
        if (i % 10 == 0) | (i == len(neu_batches)-2):
            print(neu_batches[i+1],"entries' term freuquency calculated")
        i += 1


    pos_batches = np.linspace(9001,13500,100).astype(int)
    i=0
    pos_tf = []
    while i < len(pos_batches)-1:
        batch_result = np.sum(document_matrix[pos_batches[i]:pos_batches[i+1]].toarray(),axis=0)
        pos_tf.append(batch_result)
        if (i % 10 == 0) | (i == len(pos_batches)-2):
            print(pos_batches[i+1],"entries' term freuquency calculated")
        i += 1


    # Now we have positives and negatives 
    term_freq_df.columns = ['negative', 'neutral', 'positive']
    term_freq_df['total'] = term_freq_df['negative']+ term_freq_df['neutral'] + term_freq_df['positive']
    term_freq_df.sort_values(by='total', ascending=False).iloc[:10]



    #term_freq_df2 = term_freq_df
    term_freq_df2 =  term_freq_df

    term_freq_df2['pos_rate'] = term_freq_df2['positive'] * 1./term_freq_df2['total']
    term_freq_df2.sort_values(by='pos_rate', ascending=False).iloc[:10]


    term_freq_df2['pos_freq_pct'] = term_freq_df2['positive'] * 1./term_freq_df2['positive'].sum()
    term_freq_df2.sort_values(by='pos_freq_pct', ascending=False).iloc[:10]

    from scipy.stats import hmean

    term_freq_df2['pos_hmean'] = term_freq_df2.apply(lambda x: (hmean([x['pos_rate'], x['pos_freq_pct']])                                                               if x['pos_rate'] > 0 and x['pos_freq_pct'] > 0 else 0), axis=1)
                                                           
    term_freq_df2.sort_values(by='pos_hmean', ascending=False).iloc[:10]



    from scipy.stats import norm
    from scipy.stats import gmean

    #from scipy.stats import *



    def normcdf(x):
        return norm.cdf(x, x.mean(), x.std())


    term_freq_df2['pos_rate_normcdf'] = normcdf(term_freq_df2['pos_rate'])

    term_freq_df2['pos_freq_pct_normcdf'] = normcdf(term_freq_df2['pos_freq_pct'])

    term_freq_df2['pos_normcdf_hmean'] = gmean([term_freq_df2['pos_rate_normcdf'], term_freq_df2['pos_freq_pct_normcdf']])


    # Now the pos_rate_normcdf is the column to be shown for Super words



    term_freq_df2['neg_rate'] = term_freq_df2['negative'] * 1./term_freq_df2['total']

    term_freq_df2['neg_freq_pct'] = term_freq_df2['negative'] * 1./term_freq_df2['negative'].sum()

    term_freq_df2['neg_hmean'] = term_freq_df2.apply(lambda x: (hmean([x['neg_rate'], x['neg_freq_pct']]) if x['neg_rate'] > 0 and x['neg_freq_pct'] > 0 else 0), axis=1)
                                                           
    term_freq_df2['neg_rate_normcdf'] = normcdf(term_freq_df2['neg_rate'])

    term_freq_df2['neg_freq_pct_normcdf'] = normcdf(term_freq_df2['neg_freq_pct'])

    term_freq_df2['neg_normcdf_hmean'] = gmean([term_freq_df2['neg_rate_normcdf'], term_freq_df2['neg_freq_pct_normcdf']])



    # Good 
    term_freq_df2['neu_rate'] = term_freq_df2['neutral'] * 1./term_freq_df2['total']

    term_freq_df2['neu_freq_pct'] = term_freq_df2['neutral'] * 1./term_freq_df2['neutral'].sum()

    term_freq_df2['neu_hmean'] = term_freq_df2.apply(lambda x: (hmean([x['neu_rate'], x['neu_freq_pct']]) if x['neu_rate'] > 0 and x['neu_freq_pct'] > 0 else 0), axis=1)

    term_freq_df2['neu_rate_normcdf'] = normcdf(term_freq_df2['neu_rate'])

    term_freq_df2['neu_freq_pct_normcdf'] = normcdf(term_freq_df2['neu_freq_pct'])

    term_freq_df2['neu_normcdf_hmean'] = gmean([term_freq_df2['neu_rate_normcdf'], term_freq_df2['neu_freq_pct_normcdf']])





    term_freq_df2['word'] = words
    term_freq_df2['word'] = term_freq_df2['word'].replace('emoji_start',' \\U000', regex = True)
    term_freq_df2['word'] = term_freq_df2['word'].replace('emoji_end',' ', regex = True)

    top_50_pos = term_freq_df2.sort_values(by='pos_normcdf_hmean',ascending=False).iloc[:300]
    top_50_neg = term_freq_df2.sort_values(by='neg_normcdf_hmean', ascending=False).iloc[:300]
    top_50_neu = term_freq_df2.sort_values(by='neu_normcdf_hmean', ascending=False).iloc[:300]

    pos_labels = top_50_pos['word']
    pos_values = top_50_pos['positive']

    data = {'word':pos_labels, 'number':pos_values} 
    df_positive = pd.DataFrame(data) 
    target = os.path.join(APP_ROOT, 'words/')
    print(target)

    if not os.path.isdir(target):
        os.mkdir(target)

    df_positive.to_excel(target + "/df_positive.xlsx")  # doctest: +SKIP


    neu_labels = top_50_neu['word']
    neu_values = top_50_neu['neutral']

    data = {'word':neu_labels, 'number':neu_values} 
    df_neutral = pd.DataFrame(data) 
    df_neutral.to_excel(target + "/df_neutral.xlsx")

    neg_labels = top_50_neg['word']
    neg_values = top_50_neg['negative']

    data = {'word':neg_labels, 'number':neg_values} 
    df_negative = pd.DataFrame(data) 
    df_negative.to_excel(target + "/df_negative.xlsx")

    test_emoji = "\U00012854"

    return render_template('results_dareja_graphs.html',
        df = df,
        my_df = my_df,
        words = words,
        term_freq_df2 = term_freq_df2,
        top_50_neg = top_50_neg, 
        top_50_pos = top_50_pos,
        top_50_neu = top_50_neu,  
        pos_labels = pos_labels,
        pos_values = pos_values,
        neg_labels = neg_labels,
        neg_values = neg_values, 
        neu_labels = neu_labels,
        neu_values = neu_values, 
        test_emoji = test_emoji
        )






































@analyse.route('/topic', methods=['POST', 'GET'])
def topic():
    if not current_user.is_authenticated:
        flash('You should be logged in to acced this page', 'info')
        next_page =  request.endpoint
        return redirect(url_for('users.login', next = next_page))
    title = "Topic"
    return render_template('topic.html', title = title)

@analyse.route('/topic_results', methods=['POST'])
def topic_results():
    if not current_user.is_authenticated:
        flash('You should be logged in to acced this page', 'info')
        next_page =  request.endpoint
        return redirect(url_for('users.login', next = next_page))
    if request.method != "POST":
        flash("You don't have permission accessing this page", 'info')
        return redirect(url_for('main.dashboard'))
    title = "Topic Results"
    import spacy
    #spacy.load('en_core_web_sm')
    from spacy.lang.en import English
    parser = English()
    def tokenize(text):
        lda_tokens = []
        tokens = parser(text)
        for token in tokens:
            if token.orth_.isspace():
                continue
            elif token.like_url:
                lda_tokens.append('URL')
            elif token.orth_.startswith('@'):
                lda_tokens.append('SCREEN_NAME')
            else:
                lda_tokens.append(token.lower_)
        return lda_tokens

    import nltk
    from nltk.corpus import wordnet as wn
    def get_lemma(word):
        lemma = wn.morphy(word)
        if lemma is None:
            return word
        else:
            return lemma
        
    from nltk.stem.wordnet import WordNetLemmatizer
    def get_lemma2(word):
        return WordNetLemmatizer().lemmatize(word)

    en_stop = set(nltk.corpus.stopwords.words('english'))
    def prepare_text_for_lda(text):
        tokens = tokenize(text)
        tokens = [token for token in tokens if len(token) > 4]
        tokens = [token for token in tokens if token not in en_stop]
        tokens = [get_lemma(token) for token in tokens]
        return tokens

    target = os.path.join(APP_ROOT, 'topic_modeling/')
    print(target)
    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist("file"):
        print(file)
        filename = file.filename
        destination = "/".join([target, filename])
        print(destination)
        file.save(destination)
        import random
        text_data = []
        with open(os.path.join(APP_ROOT, 'topic_modeling/'+filename)) as f:
            for line in f:
                tokens = prepare_text_for_lda(line)
                if random.random() > .99:
                    print(tokens)
                    text_data.append(tokens)

    from gensim import corpora
    dictionary = corpora.Dictionary(text_data)
    corpus = [dictionary.doc2bow(text) for text in text_data]
    import pickle
    pickle.dump(corpus, open('corpus.pkl', 'wb'))
    dictionary.save('dictionary.gensim')


    import gensim
    NUM_TOPICS = 7
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)
    ldamodel.save('model5.gensim')
    topics = ldamodel.print_topics(num_words=10)
    for topic in topics:
        print(topic)


    new_doc = 'Practical Bayesian Optimization of Machine Learning Algorithms'
    new_doc = prepare_text_for_lda(new_doc)
    new_doc_bow = dictionary.doc2bow(new_doc)
    print(new_doc_bow)
    print(ldamodel.get_document_topics(new_doc_bow))


    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = 3, id2word=dictionary, passes=15)
    ldamodel.save('model_sel.gensim')
    topics = ldamodel.print_topics(num_words=4)
    for topic in topics:
        print(topic)



    topics = ldamodel.print_topics(num_words=7)
    for topic in topics:
        print(topic)



    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = 10, id2word=dictionary, passes=15)
    ldamodel.save('model_sel.gensim')
    topics = ldamodel.print_topics(num_words=4)
    for topic in topics:
        print(topic)


    dictionary = gensim.corpora.Dictionary.load('dictionary.gensim')
    corpus = pickle.load(open('corpus.pkl', 'rb'))
    lda = gensim.models.ldamodel.LdaModel.load('model_sel.gensim')
    import pyLDAvis.gensim
    lda_display = pyLDAvis.gensim.prepare(lda, corpus, dictionary, sort_topics=False)
    pyLDAvis.save_html(lda_display, 'lda.html')
    pyLDAvis.show(lda_display)

    return render_template('topic_results.html', title = title, 
        text_data = text_data)


















@analyse.route('/choose_model')
def choose_model():
    if not current_user.is_authenticated:
        flash('You should be logged in to acced this page', 'info')
        next_page =  request.endpoint
        return redirect(url_for('users.login', next = next_page))
    title = "choose model"
    folders = get_models()
    return render_template('choose_model.html', title = title)









@analyse.route('/save_model_reg', methods = ['POST', 'GET'])
def save_model_reg():
    if not current_user.is_authenticated:
        flash('You should be logged in to acced this page', 'info')
        next_page =  request.endpoint
        return redirect(url_for('users.login', next = next_page))
    if request.method != "POST":
        flash("You don't have permission accessing this page", 'info')
        return redirect(url_for('main.dashboard'))
    global tg_cvec, sentiment_fit
    import pandas as pd  
    import pickle
    import numpy as np
    import re


       
                  

    form = SaveModel()
    title = 'Sauvegarde de votre apprentissage'
    #df = pd.read_csv('YoutubeSpamMergedData.csv')


    df = df_save
    df['comment'] = df['comment'].astype(str)
    df_data = df[["comment", "polarity"]]
    df_x = df_data['comment']
    df_y = df_data.polarity

    if request.method == "POST":
        classifier = request.form['classifier']
        word_grams = request.form['word_grams']
        vectorizer = request.form['vectorizer']
        stopwords = 'dareja'
        save_path = request.form['modelName']


    df_neg = df[df.polarity == -1]
    df_pos = df[df.polarity == 1]
    df_neu = df[df.polarity == 0]
    df_neg = df_neg.assign(polarity= 0)
    df_neu = df_neu.assign(polarity= 2)
    df_pos = df_pos.assign(polarity= 4)
    df3 = df_neg.append(df_neu, ignore_index=True)
    df3 = df3.append(df_pos, ignore_index=True)
    # NOW , it has only POSITIVES AND NEGATIVES 
    df = df3
    my_df = df
    


    #cvec = CountVectorizer()


    # get the lengths of the three categories
    len_neg = len(df_neg)
    len_neu = len(df_neu)
    len_pos = len(df_pos)

    min_n_cats = min(len_neg, len_neu, len_pos)

    n_train =  int(min_n_cats * 90 // 100)
    n_test = int(min_n_cats * 5 // 100)
    n_validate = int(min_n_cats * 5 // 100)

    total = n_train + n_test + n_validate

    my_df['comment'] = my_df.comment.astype(str)

    df_neg = my_df[my_df.polarity == 0][0:n_train]
    df_neu = my_df[my_df.polarity == 2][0:n_train]
    df_pos = my_df[my_df.polarity == 4][0:n_train]
    frames = [df_neg, df_neu, df_pos]
    result = pd.concat(frames, ignore_index=True)
    # THIS VARIABLE HAS ALL THE TRAINING DATA SEPARATED 
    result

    # This will give you the number of feature names

    df.head()

    x = result.comment
    y = result.polarity
    
    SEED = 2000

    # assigning x and y directly as the train set
    x_train = x
    y_train = y 


    # setting the x_test and y_test 
    # Testing Sample
    df_neg_1 = my_df[my_df.polarity == 0][n_train + 1:n_train + n_test]
    df_neu_1 = my_df[my_df.polarity == 2][n_train + 1:n_train + n_test]
    df_pos_1 = my_df[my_df.polarity == 4][n_train + 1:n_train + n_test]
    frames_1 = [df_neg_1, df_neu_1, df_pos_1]
    result_1 = pd.concat(frames_1)
    x_test = result_1.comment
    y_test = result_1.polarity

    #Validation sample
    df_neg_2 = my_df[my_df.polarity == 0][n_train + n_test + 1:n_train + n_test + n_validate]
    df_neu_2 = my_df[my_df.polarity == 2][n_train + n_test + 1:n_train + n_test + n_validate]
    df_pos_2 = my_df[my_df.polarity == 4][n_train + n_test + 1:n_train + n_test + n_validate]
    frames_2 = [df_neg_2, df_neu_2, df_pos_2]
    result_2 = pd.concat(frames_2)
    x_validation = result_2.comment
    y_validation = result_2.polarity  


    def train_test_and_evaluate(pipeline, x_train, y_train, x_test, y_test):
        if len(x_test[y_test == 0]) / (len(x_test)*1.) > 0.33:
            null_accuracy = len(x_test[y_test == 0]) / (len(x_test)*1.)
        else:
            null_accuracy = 1. - (len(x_test[y_test == 0]) / (len(x_test)*1.))
        sentiment_fit = pipeline.fit(x_train, y_train)
        y_pred = sentiment_fit.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        conmat = np.array(confusion_matrix(y_test, y_pred, labels=[0,2,4]))
        confusion = pd.DataFrame(conmat, index=['negative', 'neutral', 'positive'],
                             columns=['predicted_negative','predicted_neutral','predicted_positive'])
        class_report = classification_report(y_test, y_pred, output_dict=True)
        class_report_df = pd.DataFrame(class_report).transpose()
        print ("null accuracy: {0:.2f}%".format(null_accuracy*100) )
        print ("accuracy score: {0:.2f}%".format(accuracy*100) )
        if accuracy > null_accuracy:
            print ("model is {0:.2f}% more accurate than null accuracy".format((accuracy-null_accuracy)*100) )
        elif accuracy == null_accuracy:
            print ("model has the same accuracy with the null accuracy" )
        else:
            print ("model is {0:.2f}% less accurate than null accuracy".format((null_accuracy-accuracy)*100) )
        print ("-"*80 )
        print ("Confusion Matrix\n" )
        print (confusion)
        print ("-"*80 )
        print ("Classification Report\n" )
        print (classification_report(y_test, y_pred, target_names=['negative', 'neutral', 'positive']) )
        return confusion , class_report_df



    stopwords_dareja = frozenset(['ooredoo', 'klina', 'merci',
    'mercii', 'merciii', 'ana', 'homa', 'w', 'sel', 'jit', 'lebareh', 'ﻭ'])

    def train_model(x_train = '', y_train = '', x_test = '', vectorizer = '', classifier = '', stop_words = None, ngram_range='unigrams'):
        if stop_words in ['dareja', 'darja']:
            stop_words = stopwords_dareja
        #elif stopwords in ['dareja', 'darja']:
        #    stop = stop_dareja
        #else:
        #    stop = None
        if ngram_range == 'unigrams':
            ngram_range = (1, 1)
        elif ngram_range == 'bigrams':
            ngram_range = (1, 2)
        elif ngram_range == 'trigrams':
            ngram_range = (1, 3)
        if vectorizer == 'cv':
            vectorizer = CountVectorizer()
            # get the number of feature names , but first we fit the comment
            vectorizer.fit(x)
            num_features = len(vectorizer.get_feature_names())
            vectorizer.set_params(stop_words = stop_words, analyzer = 'word', max_features=num_features,ngram_range=ngram_range)
        elif vectorizer == 'tfidf':
            vectorizer = TfidfVectorizer()
            # get the number of feature names , but first we fit the comment
            vectorizer.fit(x)
            num_features = len(vectorizer.get_feature_names())
            vectorizer.set_params(stop_words = stop_words, analyzer = 'word', max_features=num_features, ngram_range=ngram_range)
        else:
            vectorizer = CountVectorizer()
            # get the number of feature names , but first we fit the comment
            vectorizer.fit(x)
            num_features = len(vectorizer.get_feature_names())
            vectorizer.set_params(stop_words = stop_words, analyzer = 'word', max_features=num_features,ngram_range=ngram_range)
        if classifier == 'Logistic_Regression':
            classifier = Logistic_Regression
        elif classifier == 'Linear_SVC':
            classifier = Linear_SVC
        elif classifier == 'LinearSVC_L1':
            classifier = LinearSVC_L1
        elif classifier == 'Multinomial_NB':
            classifier = Multinomial_NB
        elif classifier == 'Bernoulli_NB':
            classifier = Bernoulli_NB
        elif classifier == 'Ridge_Classifier':
            classifier = Ridge_Classifier
        elif classifier == 'AdaBoost':
            classifier = AdaBoost
        elif classifier == 'Perceptron':
            classifier = Perceptron
        elif classifier == 'Passive_Aggresive':
            classifier = Passive_Aggresive
        elif classifier == 'Nearest_Centroid':
            classifier = Nearest_Centroid
        else:
            classifier = Logistic_Regression
        tg_pipeline = Pipeline([
                ('vectorizer', vectorizer),
                ('classifier', classifier)
            ])
        sentiment_fit = tg_pipeline.fit(x_train, y_train)
        y_pred_test = sentiment_fit.predict(x_test)
        tg_cvec = vectorizer
        r_t_confusion, r_t_class_report = train_test_and_evaluate(tg_pipeline, x_train, y_train, x_test, y_test)
        accuracy_avg = r_t_class_report.precision[3]
        accuracy_avg = float("{0:.3f}".format(accuracy_avg))
        return sentiment_fit, y_pred_test, tg_cvec, accuracy_avg
      


    sentiment_fit, y_pred_test, tg_cvec, accuracy_avg = train_model(x_train = x_train, 
        y_train = y_train, 
        x_test = x_test, 
        vectorizer = vectorizer, 
        classifier = classifier, 
        stop_words = stopwords,
        ngram_range = word_grams)





    target = os.path.join(APP_ROOT, 'models/' + save_path + '/')
    print(target)


    if not os.path.isdir(target):
        os.mkdir(target)

    
    with open(target + 'sentiment_fit.pkl', 'wb') as f:
        pickle.dump(sentiment_fit, f)

    with open(target + 'tg_cvec.pkl', 'wb') as f:
        pickle.dump(tg_cvec, f)

    model_type = "regression"
    seq_length = 0
    n_layers = 3
    user_id = current_user.id

    result = cursor.execute( """INSERT INTO models (name, type, seq_length, 
        n_layers, user_id, acc, val_acc ) VALUES ( %s , %s , %s, %s , %s, %s, %s)""" , 
        (save_path, model_type, seq_length, n_layers ,user_id, accuracy_avg, accuracy_avg)  )
    conn.commit()
    print("model details are saved now ")


    return render_template('save_model_reg.html', title = title,
        y_pred_test = y_pred_test, 
        tg_cvec = tg_cvec,
        sentiment_fit = sentiment_fit, 
        classifier = classifier,
        vectorizer = vectorizer,
        save_path = save_path, 
        accuracy_avg = accuracy_avg)

































































































@analyse.route('/save_model3', methods = ['POST', 'GET'])
def save_model3():
    if not current_user.is_authenticated:
        flash('You should be logged in to acced this page', 'info')
        next_page =  request.endpoint
        return redirect(url_for('users.login', next = next_page))

    if request.method != "POST":
        flash("You don't have permission accessing this page", 'info')
        return redirect(url_for('main.dashboard'))
    global tg_cvec, sentiment_fit
    import pandas as pd  
    import pickle
    import numpy as np
    import re


       
                  

    form = SaveModel()
    title = 'Saving model page'
    #df = pd.read_csv('YoutubeSpamMergedData.csv')
    target = os.path.join(APP_ROOT, 'trainingcsv/')
    print(target)

    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist("file"):
        print(file)
        filename = file.filename
        destination = "/".join([target, filename])
        print(destination)
        file.save(destination)
        df = pd.read_excel(os.path.join(APP_ROOT, 'trainingcsv/'+filename))
        df['text'] = df['text'].astype(str)
        df_data = df[["text", "polarity"]]
        df_x = df_data['text']
        df_y = df_data.polarity

    if request.method == "POST":
        classifier = request.form['classifier']
        word_grams = request.form['word_grams']
        vectorizer = request.form['vectorizer']
        stopwords = request.form['stopwords']
        save_path = request.form['modelName']


    df_neg = df[df.polarity == -1]
    df_pos = df[df.polarity == 1]
    df_neu = df[df.polarity == 0]
    df_neg = df_neg.assign(polarity= 0)
    df_neu = df_neu.assign(polarity= 2)
    df_pos = df_pos.assign(polarity= 4)
    df3 = df_neg.append(df_neu, ignore_index=True)
    df3 = df3.append(df_pos, ignore_index=True)
    # NOW , it has only POSITIVES AND NEGATIVES 
    df = df3
    my_df = df
    


    #cvec = CountVectorizer()


    # get the lengths of the three categories
    len_neg = len(df_neg)
    len_neu = len(df_neu)
    len_pos = len(df_pos)

    min_n_cats = min(len_neg, len_neu, len_pos)

    n_train =  int(min_n_cats * 80 // 100)
    n_test = int(min_n_cats * 10 // 100)
    n_validate = int(min_n_cats * 10 // 100)

    total = n_train + n_test + n_validate

    my_df['text'] = my_df.text.astype(str)

    df_neg = my_df[my_df.polarity == 0][0:n_train]
    df_neu = my_df[my_df.polarity == 2][0:n_train]
    df_pos = my_df[my_df.polarity == 4][0:n_train]
    frames = [df_neg, df_neu, df_pos]
    result = pd.concat(frames, ignore_index=True)
    # THIS VARIABLE HAS ALL THE TRAINING DATA SEPARATED 
    result

    # This will give you the number of feature names

    df.head()

    x = result.text
    y = result.polarity
    
    SEED = 2000

    # assigning x and y directly as the train set
    x_train = x
    y_train = y 


    # setting the x_test and y_test 
    # Testing Sample
    df_neg_1 = my_df[my_df.polarity == 0][n_train + 1:n_train + n_test]
    df_neu_1 = my_df[my_df.polarity == 2][n_train + 1:n_train + n_test]
    df_pos_1 = my_df[my_df.polarity == 4][n_train + 1:n_train + n_test]
    frames_1 = [df_neg_1, df_neu_1, df_pos_1]
    result_1 = pd.concat(frames_1)
    x_test = result_1.text
    y_test = result_1.polarity

    #Validation sample
    df_neg_2 = my_df[my_df.polarity == 0][n_train + n_test + 1:n_train + n_test + n_validate]
    df_neu_2 = my_df[my_df.polarity == 2][n_train + n_test + 1:n_train + n_test + n_validate]
    df_pos_2 = my_df[my_df.polarity == 4][n_train + n_test + 1:n_train + n_test + n_validate]
    frames_2 = [df_neg_2, df_neu_2, df_pos_2]
    result_2 = pd.concat(frames_2)
    x_validation = result_2.text
    y_validation = result_2.polarity  



    stopwords_dareja = frozenset(['ooredoo', 'klina', 'merci',
    'mercii', 'merciii', 'ana', 'homa', 'w', 'sel', 'jit', 'lebareh', 'ﻭ'])

    def train_model(x_train = '', y_train = '', x_test = '', vectorizer = '', classifier = '', stop_words = None, ngram_range='unigrams'):
        if stop_words in ['dareja', 'darja']:
            stop_words = stopwords_dareja
        #elif stopwords in ['dareja', 'darja']:
        #    stop = stop_dareja
        #else:
        #    stop = None
        if ngram_range == 'unigrams':
            ngram_range = (1, 1)
        elif ngram_range == 'bigrams':
            ngram_range = (1, 2)
        elif ngram_range == 'trigrams':
            ngram_range = (1, 3)
        if vectorizer == 'cv':
            vectorizer = CountVectorizer()
            # get the number of feature names , but first we fit the text
            vectorizer.fit(x)
            num_features = len(vectorizer.get_feature_names())
            vectorizer.set_params(stop_words = stop_words, analyzer = 'word', max_features=num_features,ngram_range=ngram_range)
        elif vectorizer == 'tfidf':
            vectorizer = TfidfVectorizer()
            # get the number of feature names , but first we fit the text
            vectorizer.fit(x)
            num_features = len(vectorizer.get_feature_names())
            vectorizer.set_params(stop_words = stop_words, analyzer = 'word', max_features=num_features, ngram_range=ngram_range)
        else:
            vectorizer = CountVectorizer()
            # get the number of feature names , but first we fit the text
            vectorizer.fit(x)
            num_features = len(vectorizer.get_feature_names())
            vectorizer.set_params(stop_words = stop_words, analyzer = 'word', max_features=num_features,ngram_range=ngram_range)
        if classifier == 'Logistic_Regression':
            classifier = Logistic_Regression
        elif classifier == 'Linear_SVC':
            classifier = Linear_SVC
        elif classifier == 'LinearSVC_L1':
            classifier = LinearSVC_L1
        elif classifier == 'Multinomial_NB':
            classifier = Multinomial_NB
        elif classifier == 'Bernoulli_NB':
            classifier = Bernoulli_NB
        elif classifier == 'Ridge_Classifier':
            classifier = Ridge_Classifier
        elif classifier == 'AdaBoost':
            classifier = AdaBoost
        elif classifier == 'Perceptron':
            classifier = Perceptron
        elif classifier == 'Passive_Aggresive':
            classifier = Passive_Aggresive
        elif classifier == 'Nearest_Centroid':
            classifier = Nearest_Centroid
        else:
            classifier = Logistic_Regression
        tg_pipeline = Pipeline([
                ('vectorizer', vectorizer),
                ('classifier', classifier)
            ])
        sentiment_fit = tg_pipeline.fit(x_train, y_train)
        y_pred_test = sentiment_fit.predict(x_test)
        tg_cvec = vectorizer
        return sentiment_fit, y_pred_test, tg_cvec
      


    sentiment_fit, y_pred_test, tg_cvec = train_model(x_train = x_train, 
        y_train = y_train, 
        x_test = x_test, 
        vectorizer = vectorizer, 
        classifier = classifier, 
        stop_words = stopwords,
        ngram_range = word_grams)

    target = os.path.join(APP_ROOT, 'models/' + current_user.username + '/' + save_path + '/')
    print(target)


    if not os.path.isdir(target):
        os.mkdir(target)

    
    with open(target + 'sentiment_fit.pkl', 'wb') as f:
        pickle.dump(sentiment_fit, f)

    with open(target + 'tg_cvec.pkl', 'wb') as f:
        pickle.dump(tg_cvec, f)



    return render_template('save_model2.html', title = title,
        y_pred_test = y_pred_test, 
        tg_cvec = tg_cvec,
        sentiment_fit = sentiment_fit)






















































#endpoint for search
@analyse.route('/train_database', methods=['GET', 'POST'])
def train_database():
    # redirect if user is not logged in 
    if not current_user.is_authenticated:
        flash('You should be logged in to acced this page', 'info')
        next_page =  request.endpoint
        return redirect(url_for('users.login', next = next_page))
    # render the template
    return render_template('train_database.html', title='Train From Database' 
        )




@analyse.route('/train_results_db', methods = ['POST', 'GET'])
def train_results_db():
    # redirect user if not logged in, and redirect if it is not a form submission
    if not current_user.is_authenticated:
        flash('You should be logged in to acced this page', 'info')
        next_page =  request.endpoint
        return redirect(url_for('users.login', next = next_page))
    if request.method != "POST":
        flash("You don't have permission accessing this page", 'info')
        return redirect(url_for('main.dashboard'))
    # declare global variables (these will be used in saving the model)
    global tg_cvec, sentiment_fit2, df_save
    # import the Savemodel Form
    form = SaveModel()
    #df = pd.read_csv('YoutubeSpamMergedData.csv')
    target = os.path.join(APP_ROOT, 'trainingcsv/')
    print(target)
    if not os.path.isdir(target):
        os.mkdir(target)

    # get the form inputs and select data from the database based on it.
    if request.method == "POST":
        startdate = request.form['startdate']
        enddate = request.form['enddate']
        sector = request.form['sector']
        ratio_train = int(request.form['ratio_train'])
        equalize_set = request.form['equalize_set']
        ratio_test = int(100 - int(ratio_train)) // 2
        #ratio_train = 95
        #start = datetime.datetime(start)
        #end = datetime.datetime(end)
        startdate = startdate + ' 00:00:00'
        enddate = enddate + ' 00:00:00'
        date_str = type(startdate)
        # search by author or book
        cursor.execute("""SELECT comment, polarity from training 
            WHERE date BETWEEN %s AND %s AND sector = %s""", (startdate, enddate, sector))
        conn.commit()
        data = cursor.fetchall()
        type_data = type(data)
        fetched_data = pd.DataFrame(list(data), columns = ['comment', 'polarity'])
        df = pd.DataFrame()
        df['comment'] = fetched_data['comment'].astype(str)
        df['polarity'] = fetched_data['polarity'].astype(int)
        df_data = df[["comment", "polarity"]]
        df_x = df_data['comment']
        df_y = df_data.polarity
        df_save = df
        df_save.to_excel( target + "Before_training.xlsx")
        
    # clean the selected data
    df2 = clean_comments(df_content = df, column_name = 'comment', stopword = stopwords)
    df = df2
    # this is optional, meant for testing purposes
    df2.to_excel(target + "after_training.xlsx")
        
    #form = SaveModel()

    # retrieve existing model names 
    cursor.execute("""SELECT name, type FROM `models` """)
    my_models = cursor.fetchall()
    models = []
    for i in my_models:
        models.append(i[0])

    #ratio_train = request.form['ratio_train']
    #ratio_test = int(100 - int(ratio_train)) // 2
    
    #df = pd.read_csv('YoutubeSpamMergedData.csv')

    if not os.path.isdir(target):
        os.mkdir(target)

    # get the comments by polarity, and assign new codes for polarity (0, 2, 4)
    # the  code 0 is for negative, 2 for neutral, 4 for positive
    df['comment'] = df.comment.astype(str)
    df['polarity'] = df.polarity.astype(int)
    df_neg = df[df.polarity == -1]
    df_pos = df[df.polarity == 1]
    df_neu = df[df.polarity == 0]
    df_neg = df_neg.assign(polarity= 0)
    df_neu = df_neu.assign(polarity= 2)
    df_pos = df_pos.assign(polarity= 4)
    df3 = df_neg.append(df_neu, ignore_index=True)
    df3 = df3.append(df_pos, ignore_index=True)
    # NOW , it has only POSITIVES AND NEGATIVES 
    df = df3

    my_df = df

    # get the lengths of the three categories
    len_neg = len(df_neg)
    len_neu = len(df_neu)
    len_pos = len(df_pos)

    # if equalize set is yes, we create duplicates of the smaller sets, to make our training data bigger
    if equalize_set == "yes":
        df_neg, df_neg_test, df_neu, df_neu_test, df_pos, df_pos_test = equilibrate_polarities(df_neg = df_neg, df_neu = df_neu, df_pos = df_pos, test_valid_percent = '')
        frames = [df_neg, df_neu, df_pos]
        result = pd.concat(frames, ignore_index=True)
        x = result.comment
        y = result.polarity
        x_train = x
        y_train = y
        frames = [df_neg_test, df_neu_test, df_pos_test]
        result_1 = pd.concat(frames, ignore_index=True)
        result_2 = result_1
        x_test = result_1.comment
        y_test = result_1.polarity
        x_validation = result_1.comment
        y_validation = result_1.polarity
        SEED = 2000
        result_1.to_excel(target + "/testing_data_sonelgaz.xlsx")
        result.to_excel(target + "/training_data_sonelgaz.xlsx")
    else:
        min_n_cats = min(len_neg, len_neu, len_pos)

        n_train =  int(min_n_cats * ratio_train // 100)
        n_test = int(min_n_cats * ratio_test // 100)
        n_validate = int(min_n_cats * ratio_test // 100)

        total = n_train + n_test + n_validate

        my_df['comment'] = my_df.comment.astype(str)

        df_neg = my_df[my_df.polarity == 0][0:n_train]
        df_neu = my_df[my_df.polarity == 2][0:n_train]
        df_pos = my_df[my_df.polarity == 4][0:n_train]
        frames = [df_neg, df_neu, df_pos]
        result = pd.concat(frames, ignore_index=True)
        # THIS VARIABLE HAS ALL THE TRAINING DATA SEPARATED 
        result
        #result_clean = clean_text_for_training(df_content = result, column_name = 'text', stopword = 'french')
        #result_clean['text'] = result_clean['comment']
        #result = result_clean
        # This will give you the number of feature names
        df.head()
        x = result.comment
        y = result.polarity
        SEED = 2000
        x_train = x
        y_train = y 
        # setting the x_test and y_test 
        # Testing Sample
        df_neg_1 = my_df[my_df.polarity == 0][len(df_neg) - (n_validate * 2):len(df_neg) - n_validate]
        df_neu_1 = my_df[my_df.polarity == 2][len(df_neg) - (n_validate * 2):len(df_neg) - n_validate]
        df_pos_1 = my_df[my_df.polarity == 4][len(df_neg) - (n_validate * 2):len(df_neg) - n_validate]
        frames_1 = [df_neg_1, df_neu_1, df_pos_1]
        result_1 = pd.concat(frames_1, ignore_index = True)
        x_test = result_1.comment
        y_test = result_1.polarity

        # for testing purposes, to see the testing data 
        result_1.to_excel(target + "/test1.xlsx")

        #x_test_clean = clean_text_for_training(df_content = result_1, column_name = 'text', stopword = 'french')
        #x_test_clean['text'] = x_test_clean['comment']
        #x_test = x_test_clean['text'] 

        

        #Validation sample
        df_neg_2 = my_df[my_df.polarity == 0][len(df_neg) - n_validate:len(df_neg)]
        df_neu_2 = my_df[my_df.polarity == 2][len(df_neg) - n_validate:len(df_neg)]
        df_pos_2 = my_df[my_df.polarity == 4][len(df_neg) - n_validate:len(df_neg)]
        frames_2 = [df_neg_2, df_neu_2, df_pos_2]
        result_2 = pd.concat(frames_2, ignore_index = True)
        x_validation = result_2.comment
        y_validation = result_2.polarity 
        # for testing purposes, to see the validation data 
        result_2.to_excel(target + "/valid1.xlsx")

    #d = {'text':x_validation,'polarity':y_validation}
    #result_2 = pd.DataFrame(d)

    #x_validation_clean = clean_text_for_training(df_content = result_2, column_name = 'text', stopword = 'french')
    #x_validation_clean['text'] = x_validation_clean['comment']
    #x_validation = x_validation_clean['text'] 



    print ("Train set has total {0} entries with {1:.2f}% negative, {2:.2f}% Good, {3:.2f}% positive".format(len(x_train),
        (len(x_train[y_train == 0]) / (len(x_train)*1.))*100,
        (len(x_train[y_train == 2]) / (len(x_train)*1.))*100,
        (len(x_train[y_train == 4]) / (len(x_train)*1.))*100) )


    print ("Validation set has total {0} entries with {1:.2f}% negative, {2:.2f}% Good, {3:.2f}% positive".format(len(x_validation),
        (len(x_validation[y_validation == 0]) / (len(x_validation)*1.))*100,
        (len(x_validation[y_validation == 2]) / (len(x_validation)*1.))*100,
        (len(x_validation[y_validation == 4]) / (len(x_validation)*1.))*100) )

    print ("Test set has total {0} entries with {1:.2f}% negative, {2:.2f}% Good, {3:.2f}% positive".format(len(x_test),
        (len(x_test[y_test == 0]) / (len(x_test)*1.))*100,
        (len(x_test[y_test == 2]) / (len(x_test)*1.))*100,
        (len(x_test[y_test == 4]) / (len(x_test)*1.))*100) )


    from textblob import TextBlob
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import classification_report, confusion_matrix
    # Feature extraction
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from time import time




    def accuracy_summary(pipeline, x_train, y_train, x_test, y_test):
        if len(x_test[y_test == 0]) / (len(x_test)*1.) > 0.33:
            null_accuracy = len(x_test[y_test == 0]) / (len(x_test)*1.)
        else:
            null_accuracy = 1. - (len(x_test[y_test == 0]) / (len(x_test)*1.))
        t0 = time()
        sentiment_fit = pipeline.fit(x_train, y_train)
        y_pred = sentiment_fit.predict(x_test)
        train_test_time = time() - t0
        accuracy = accuracy_score(y_test, y_pred)
        print ("null accuracy: {0:.2f}%".format(null_accuracy*100) )
        print ("accuracy score: {0:.2f}%".format(accuracy*100) )
        if accuracy > null_accuracy:
            print ("model is {0:.2f}% more accurate than null accuracy".format((accuracy-null_accuracy)*100) )
        elif accuracy == null_accuracy:
            print ("model has the same accuracy with the null accuracy")
        else:
            print ("model is {0:.2f}% less accurate than null accuracy".format((null_accuracy-accuracy)*100) )
        print ("train and test time: {0:.2f}s".format(train_test_time) )
        print ("-"*80 )
        return accuracy, train_test_time


    cvec = CountVectorizer()
    # get the number of feature names , but first we fit the text
    cvec.fit(x)
    len(cvec.get_feature_names())
    num_features = len(cvec.get_feature_names())
    cv_steps = round(num_features / 8)

    lr = LogisticRegression(solver='newton-cg', multi_class='multinomial')
    n_features = np.arange(cv_steps, num_features, cv_steps)


    # get the stemmer function, this is optional 
    # with stemming words
    import nltk
    from nltk.stem.isri import ISRIStemmer
    stemmer = ISRIStemmer()
    analyzer = CountVectorizer().build_analyzer()

    def stemmed_words(doc):
        return (stemmer.stem(w) for w in analyzer(doc))
    # remove the above code as it is used in another function(if any errors)


    def nfeature_accuracy_checker(vectorizer=cvec, n_features=n_features, stop_words=None, ngram_range=(1, 1), classifier=lr):
        result = []
        print (classifier)
        print ("\n")
        for n in n_features:
            vectorizer.set_params(stop_words=stop_words, max_features=n, ngram_range=ngram_range)
            checker_pipeline = Pipeline([
                ('vectorizer', vectorizer),
                ('classifier', classifier)
            ])
            print ("Validation result for {} features".format(n) )
            nfeature_accuracy,tt_time = accuracy_summary(checker_pipeline, x_train, y_train, x_validation, y_validation)
            result.append((n,nfeature_accuracy,tt_time))
        return result


    my_stop_words = ['selmane', 'aissa', 'brahim']

    print ("RESULT FOR UNIGRAM WITHOUT STOP WORDS\n")
    feature_result_wosw = nfeature_accuracy_checker(stop_words = 'english') 

    print ("RESULT FOR UNIGRAM WITH STOP WORDS\n")
    feature_result_ug = nfeature_accuracy_checker() 

    print ("RESULT FOR UNIGRAM WITHOUT CUSTOM STOP WORDS (Top 10 frequent words)\n" )
    feature_result_wocsw = nfeature_accuracy_checker(stop_words=my_stop_words) 


    nfeatures_plot_ug = pd.DataFrame(feature_result_ug,columns=['nfeatures','validation_accuracy','train_test_time'])
    nfeatures_plot_ug_wocsw = pd.DataFrame(feature_result_wocsw,columns=['nfeatures','validation_accuracy','train_test_time'])
    nfeatures_plot_ug_wosw = pd.DataFrame(feature_result_wosw,columns=['nfeatures','validation_accuracy','train_test_time'])

    plt.figure(figsize=(8,6))
    plt.plot(nfeatures_plot_ug.nfeatures, nfeatures_plot_ug.validation_accuracy, label='with stop words')
    plt.plot(nfeatures_plot_ug_wosw.nfeatures, nfeatures_plot_ug_wosw.validation_accuracy,label='without stop words')
    plt.plot(nfeatures_plot_ug_wocsw.nfeatures, nfeatures_plot_ug_wocsw.validation_accuracy,label='without custom stop words')
    plt.title("Without stop words VS With stop words (Unigram): Accuracy")
    plt.xlabel("Number of features")
    plt.ylabel("Validation set accuracy")
    #plt.legend()

    ug_labels = nfeatures_plot_ug.nfeatures
    ug_values = nfeatures_plot_ug.validation_accuracy

    ug_wocsw_labels = nfeatures_plot_ug_wocsw.nfeatures
    ug_wocsw_values = nfeatures_plot_ug_wocsw.validation_accuracy

    ug_wosw_labels = nfeatures_plot_ug_wosw.nfeatures
    ug_wosw_values = nfeatures_plot_ug_wosw.validation_accuracy


    # Showing the tests for bigrams and trigrams 
    print ("RESULT FOR BIGRAM WITH STOP WORDS\n")
    feature_result_bg = nfeature_accuracy_checker(ngram_range=(1, 2))

    print ("RESULT FOR TRIGRAM WITH STOP WORDS\n")
    feature_result_tg = nfeature_accuracy_checker(ngram_range=(1, 3))


    nfeatures_plot_tg = pd.DataFrame(feature_result_tg,columns=['nfeatures','validation_accuracy','train_test_time'])
    nfeatures_plot_bg = pd.DataFrame(feature_result_bg,columns=['nfeatures','validation_accuracy','train_test_time'])
    nfeatures_plot_ug = pd.DataFrame(feature_result_ug,columns=['nfeatures','validation_accuracy','train_test_time'])

    plt.figure(figsize=(8,6))
    plt.plot(nfeatures_plot_tg.nfeatures, nfeatures_plot_tg.validation_accuracy,label='trigram')
    plt.plot(nfeatures_plot_bg.nfeatures, nfeatures_plot_bg.validation_accuracy,label='bigram')
    plt.plot(nfeatures_plot_ug.nfeatures, nfeatures_plot_ug.validation_accuracy, label='unigram')
    plt.title("N-gram(1~3) test result : Accuracy")
    plt.xlabel("Number of features")
    plt.ylabel("Validation set accuracy")
    plt.legend()
    #plt.show()

    bg_values = nfeatures_plot_bg.validation_accuracy
    tg_values = nfeatures_plot_tg.validation_accuracy

    def train_test_and_evaluate(pipeline, x_train, y_train, x_test, y_test):
        if len(x_test[y_test == 0]) / (len(x_test)*1.) > 0.33:
            null_accuracy = len(x_test[y_test == 0]) / (len(x_test)*1.)
        else:
            null_accuracy = 1. - (len(x_test[y_test == 0]) / (len(x_test)*1.))
        sentiment_fit = pipeline.fit(x_train, y_train)
        y_pred = sentiment_fit.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        conmat = np.array(confusion_matrix(y_test, y_pred, labels=[0,2,4]))
        confusion = pd.DataFrame(conmat, index=['negative', 'neutral', 'positive'],
                             columns=['predicted_negative','predicted_neutral','predicted_positive'])
        class_report = classification_report(y_test, y_pred, output_dict=True)
        class_report_df = pd.DataFrame(class_report).transpose()
        print ("null accuracy: {0:.2f}%".format(null_accuracy*100) )
        print ("accuracy score: {0:.2f}%".format(accuracy*100) )
        if accuracy > null_accuracy:
            print ("model is {0:.2f}% more accurate than null accuracy".format((accuracy-null_accuracy)*100) )
        elif accuracy == null_accuracy:
            print ("model has the same accuracy with the null accuracy" )
        else:
            print ("model is {0:.2f}% less accurate than null accuracy".format((null_accuracy-accuracy)*100) )
        print ("-"*80 )
        print ("Confusion Matrix\n" )
        print (confusion)
        print ("-"*80 )
        print ("Classification Report\n" )
        print (classification_report(y_test, y_pred, target_names=['negative', 'neutral', 'positive']) )
        return confusion , class_report_df


    tg_cvec = CountVectorizer(max_features=num_features,ngram_range=(1, 3))
    tg_pipeline = Pipeline([
            ('vectorizer', tg_cvec),
            ('classifier', lr)
        ])
    r_t_confusion, r_t_class_report = train_test_and_evaluate(tg_pipeline, x_train, y_train, x_test, y_test)

    r_v_confusion, r_v_class_report = train_test_and_evaluate(tg_pipeline, x_train, y_train,x_validation, y_validation )
    # getting the number of true positives and negatives to show on the website 

    true_negative = r_t_confusion["predicted_negative"][0]
    false_negative1 = r_t_confusion["predicted_negative"][1]
    false_negative2 = r_t_confusion["predicted_negative"][2]

    false_neutral1 = r_t_confusion["predicted_neutral"][0]
    true_neutral = r_t_confusion["predicted_neutral"][1]
    false_neutral2 = r_t_confusion["predicted_neutral"][2]

    false_positive1 = r_t_confusion["predicted_positive"][0]
    false_positive2 = r_t_confusion["predicted_positive"][1]
    true_positive = r_t_confusion["predicted_positive"][2]

    count_pos = true_positive + false_positive1 + false_positive2
    count_neu = true_neutral + false_neutral1 + false_neutral2
    count_neg = true_negative + false_negative1 + false_negative2

    count_all = count_pos + count_neu + count_neg
    accuracy_avg = r_t_class_report.precision[3]
    accuracy_avg = float("{0:.3f}".format(accuracy_avg))
    # there are some functions that have been used before, and 
    # i am testing on tf-idf , unigrams, bigrams, trigrams 
    # and printing the results on a graph
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from time import time


    def accuracy_summary(pipeline, x_train, y_train, x_test, y_test):
        if len(x_test[y_test == 0]) / (len(x_test)*1.) > 0.33:
            null_accuracy = len(x_test[y_test == 0]) / (len(x_test)*1.)
        else:
            null_accuracy = 1. - (len(x_test[y_test == 0]) / (len(x_test)*1.))
        t0 = time()
        sentiment_fit = pipeline.fit(x_train, y_train)
        y_pred = sentiment_fit.predict(x_test)
        train_test_time = time() - t0
        accuracy = accuracy_score(y_test, y_pred)
        print ("null accuracy: {0:.2f}%".format(null_accuracy*100) )
        print ("accuracy score: {0:.2f}%".format(accuracy*100) )
        if accuracy > null_accuracy:
            print ("model is {0:.2f}% more accurate than null accuracy".format((accuracy-null_accuracy)*100) )
        elif accuracy == null_accuracy:
            print ("model has the same accuracy with the null accuracy")
        else:
            print ("model is {0:.2f}% less accurate than null accuracy".format((null_accuracy-accuracy)*100) )
        print ("train and test time: {0:.2f}s".format(train_test_time) )
        print ("-"*80 )
        return (accuracy, train_test_time )

    cvec = CountVectorizer()
    lr = LogisticRegression(solver='newton-cg', multi_class='multinomial')
    n_features = np.arange(5000,num_features,5000)

    def nfeature_accuracy_checker(vectorizer=cvec, n_features=n_features, stop_words=None, ngram_range=(1, 1), classifier=lr):
        result = []
        print (classifier)
        print ("\n")
        for n in n_features:
            vectorizer.set_params(stop_words=stop_words, max_features=n, ngram_range=ngram_range)
            checker_pipeline = Pipeline([
                ('vectorizer', vectorizer),
                ('classifier', classifier)
            ])
            print ("Validation result for {} features".format(n) )
            nfeature_accuracy,tt_time = accuracy_summary(checker_pipeline, x_train, y_train, x_validation, y_validation)
            result.append((n,nfeature_accuracy,tt_time))
        return result

    from sklearn.feature_extraction.text import TfidfVectorizer
    tvec = TfidfVectorizer()

    feature_result_ugt = nfeature_accuracy_checker(vectorizer=tvec)
    feature_result_bgt = nfeature_accuracy_checker(vectorizer=tvec,ngram_range=(1, 2))
    feature_result_tgt = nfeature_accuracy_checker(vectorizer=tvec,ngram_range=(1, 3))


    nfeatures_plot_tgt = pd.DataFrame(feature_result_tgt,columns=['nfeatures','validation_accuracy','train_test_time'])
    nfeatures_plot_bgt = pd.DataFrame(feature_result_bgt,columns=['nfeatures','validation_accuracy','train_test_time'])
    nfeatures_plot_ugt = pd.DataFrame(feature_result_ugt,columns=['nfeatures','validation_accuracy','train_test_time'])
    plt.figure(figsize=(8,6))
    plt.plot(nfeatures_plot_tgt.nfeatures, nfeatures_plot_tgt.validation_accuracy,label='trigram tfidf vectorizer',color='royalblue')
    plt.plot(nfeatures_plot_tg.nfeatures, nfeatures_plot_tg.validation_accuracy,label='trigram count vectorizer',linestyle=':', color='royalblue')
    plt.plot(nfeatures_plot_bgt.nfeatures, nfeatures_plot_bgt.validation_accuracy,label='bigram tfidf vectorizer',color='orangered')
    plt.plot(nfeatures_plot_bg.nfeatures, nfeatures_plot_bg.validation_accuracy,label='bigram count vectorizer',linestyle=':',color='orangered')
    plt.plot(nfeatures_plot_ugt.nfeatures, nfeatures_plot_ugt.validation_accuracy, label='unigram tfidf vectorizer',color='gold')
    plt.plot(nfeatures_plot_ug.nfeatures, nfeatures_plot_ug.validation_accuracy, label='unigram count vectorizer',linestyle=':',color='gold')
    plt.title("N-gram(1~3) test result : Accuracy")
    plt.xlabel("Number of features")
    plt.ylabel("Validation set accuracy")
    plt.legend()
    #plt.show()

    ugt_values = nfeatures_plot_bgt.validation_accuracy
    bgt_values = nfeatures_plot_bgt.validation_accuracy
    tgt_values = nfeatures_plot_tgt.validation_accuracy



    # Comparing with other classifiers
    # Now , we compare using different algorithms:
    from sklearn.svm import LinearSVC
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.naive_bayes import MultinomialNB, BernoulliNB
    from sklearn.linear_model import RidgeClassifier
    from sklearn.linear_model import PassiveAggressiveClassifier
    from sklearn.linear_model import Perceptron
    from sklearn.neighbors import NearestCentroid
    from sklearn.feature_selection import SelectFromModel
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer

    names = ["Logistic Regression", "Linear SVC", "LinearSVC with L1-based feature selection","Multinomial NB", 
             "Bernoulli NB", "Ridge Classifier", "AdaBoost", "Perceptron","Passive-Aggresive", "Nearest Centroid"]
    classifiers = [
        LogisticRegression(solver='newton-cg', multi_class='multinomial'),
        LinearSVC(),
        Pipeline([
      ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False))),
      ('classification', LinearSVC(penalty="l2"))]),
        MultinomialNB(),
        BernoulliNB(),
        RidgeClassifier(),
        AdaBoostClassifier(),
        Perceptron(),
        PassiveAggressiveClassifier(),
        NearestCentroid()
        ]
    zipped_clf = zip(names,classifiers)

    tvec = TfidfVectorizer()

    def classifier_comparator(vectorizer=tvec, n_features= num_features, stop_words=None, ngram_range=(1, 1), classifier=zipped_clf):
        result = []
        vectorizer.set_params(stop_words=stop_words, max_features=n_features, ngram_range=ngram_range)
        for n,c in classifier:
            checker_pipeline = Pipeline([
                ('vectorizer', vectorizer),
                ('classifier', c)
            ])
            print ("Validation result for {}".format(n) )
            print (c)
            clf_accuracy,tt_time = accuracy_summary(checker_pipeline, x_train, y_train, x_validation, y_validation)
            result.append((n,clf_accuracy,tt_time))
        return result

    comparing_algorithms = classifier_comparator(n_features=num_features,ngram_range=(1,3))



    comparing_algorithms = pd.DataFrame(comparing_algorithms, columns=['Model','Validation set accuracy', 'Train and test time'])









    # Getting The positives and Negatives 
    my_df = result


    # Creating the count vectorizer for later use 
    from sklearn.feature_extraction.text import CountVectorizer
    cvec = CountVectorizer()
    my_df['comment'] = my_df.comment.astype(str)
    cvec.fit(my_df.comment)

    # This will give you the number of feature names
    len(cvec.get_feature_names())
    # this above variable is important 


    # This will generate the terpm frequency for pos and neg and then combined 
    neg_doc_matrix = cvec.transform(my_df[my_df.polarity == 0].comment)
    neu_doc_matrix = cvec.transform(my_df[my_df.polarity == 2].comment)
    pos_doc_matrix = cvec.transform(my_df[my_df.polarity == 4].comment)
    neg_tf = np.sum(neg_doc_matrix,axis=0)
    neu_tf = np.sum(neu_doc_matrix,axis=0)
    pos_tf = np.sum(pos_doc_matrix,axis=0)
    neg = np.squeeze(np.asarray(neg_tf))
    neu = np.squeeze(np.asarray(neu_tf))
    pos = np.squeeze(np.asarray(pos_tf))
    words = cvec.get_feature_names()
    term_freq_df = pd.DataFrame([neg,neu,pos], columns = cvec.get_feature_names()).transpose()



    document_matrix = cvec.transform(my_df.comment)
    my_df[my_df.polarity == 0].tail()

    neg_batches = np.linspace(0,4500,100).astype(int)
    i=0
    neg_tf = []
    while i < len(neg_batches)-1:
        batch_result = np.sum(document_matrix[neg_batches[i]:neg_batches[i+1]].toarray(),axis=0)
        neg_tf.append(batch_result)
        if (i % 10 == 0) | (i == len(neg_batches)-2):
            print(neg_batches[i+1],"entries' term freuquency calculated")
        i += 1


    neu_batches = np.linspace(4501,9000,100).astype(int)
    i=0
    neu_tf = []
    while i < len(neu_batches)-1:
        batch_result = np.sum(document_matrix[neu_batches[i]:neu_batches[i+1]].toarray(),axis=0)
        neu_tf.append(batch_result)
        if (i % 10 == 0) | (i == len(neu_batches)-2):
            print(neu_batches[i+1],"entries' term freuquency calculated")
        i += 1


    pos_batches = np.linspace(9001,13500,100).astype(int)
    i=0
    pos_tf = []
    while i < len(pos_batches)-1:
        batch_result = np.sum(document_matrix[pos_batches[i]:pos_batches[i+1]].toarray(),axis=0)
        pos_tf.append(batch_result)
        if (i % 10 == 0) | (i == len(pos_batches)-2):
            print(pos_batches[i+1],"entries' term freuquency calculated")
        i += 1


    # Now we have positives and negatives 
    term_freq_df.columns = ['negative', 'neutral', 'positive']
    term_freq_df['total'] = term_freq_df['negative']+ term_freq_df['neutral'] + term_freq_df['positive']
    term_freq_df.sort_values(by='total', ascending=False).iloc[:10]



    #term_freq_df2 = term_freq_df
    term_freq_df2 =  term_freq_df

    term_freq_df2['pos_rate'] = term_freq_df2['positive'] * 1./term_freq_df2['total']
    term_freq_df2.sort_values(by='pos_rate', ascending=False).iloc[:10]


    term_freq_df2['pos_freq_pct'] = term_freq_df2['positive'] * 1./term_freq_df2['positive'].sum()
    term_freq_df2.sort_values(by='pos_freq_pct', ascending=False).iloc[:10]

    from scipy.stats import hmean

    term_freq_df2['pos_hmean'] = term_freq_df2.apply(lambda x: (hmean([x['pos_rate'], x['pos_freq_pct']])                                                               if x['pos_rate'] > 0 and x['pos_freq_pct'] > 0 else 0), axis=1)
                                                           
    term_freq_df2.sort_values(by='pos_hmean', ascending=False).iloc[:10]



    from scipy.stats import norm
    from scipy.stats import gmean

    #from scipy.stats import *



    def normcdf(x):
        return norm.cdf(x, x.mean(), x.std())


    term_freq_df2['pos_rate_normcdf'] = normcdf(term_freq_df2['pos_rate'])

    term_freq_df2['pos_freq_pct_normcdf'] = normcdf(term_freq_df2['pos_freq_pct'])

    term_freq_df2['pos_normcdf_hmean'] = gmean([term_freq_df2['pos_rate_normcdf'], term_freq_df2['pos_freq_pct_normcdf']])


    # Now the pos_rate_normcdf is the column to be shown for Super words



    term_freq_df2['neg_rate'] = term_freq_df2['negative'] * 1./term_freq_df2['total']

    term_freq_df2['neg_freq_pct'] = term_freq_df2['negative'] * 1./term_freq_df2['negative'].sum()

    term_freq_df2['neg_hmean'] = term_freq_df2.apply(lambda x: (hmean([x['neg_rate'], x['neg_freq_pct']]) if x['neg_rate'] > 0 and x['neg_freq_pct'] > 0 else 0), axis=1)
                                                           
    term_freq_df2['neg_rate_normcdf'] = normcdf(term_freq_df2['neg_rate'])

    term_freq_df2['neg_freq_pct_normcdf'] = normcdf(term_freq_df2['neg_freq_pct'])

    term_freq_df2['neg_normcdf_hmean'] = gmean([term_freq_df2['neg_rate_normcdf'], term_freq_df2['neg_freq_pct_normcdf']])



    # Good 
    term_freq_df2['neu_rate'] = term_freq_df2['neutral'] * 1./term_freq_df2['total']

    term_freq_df2['neu_freq_pct'] = term_freq_df2['neutral'] * 1./term_freq_df2['neutral'].sum()

    term_freq_df2['neu_hmean'] = term_freq_df2.apply(lambda x: (hmean([x['neu_rate'], x['neu_freq_pct']]) if x['neu_rate'] > 0 and x['neu_freq_pct'] > 0 else 0), axis=1)

    term_freq_df2['neu_rate_normcdf'] = normcdf(term_freq_df2['neu_rate'])

    term_freq_df2['neu_freq_pct_normcdf'] = normcdf(term_freq_df2['neu_freq_pct'])

    term_freq_df2['neu_normcdf_hmean'] = gmean([term_freq_df2['neu_rate_normcdf'], term_freq_df2['neu_freq_pct_normcdf']])





    term_freq_df2['word'] = words

    top_50_pos = term_freq_df2.sort_values(by='pos_normcdf_hmean',ascending=False).iloc[:50]
    top_50_neg = term_freq_df2.sort_values(by='neg_normcdf_hmean', ascending=False).iloc[:50]
    top_50_neu = term_freq_df2.sort_values(by='neu_normcdf_hmean', ascending=False).iloc[:50]

    pos_labels = top_50_pos['word']
    pos_values = top_50_pos['positive']

    neu_labels = top_50_neu['word']
    neu_values = top_50_neu['neutral']

    neg_labels = top_50_neg['word']
    neg_values = top_50_neg['negative']


    top_300_pos = term_freq_df2.sort_values(by='pos_normcdf_hmean',ascending=False).iloc[:50]
    top_300_neg = term_freq_df2.sort_values(by='neg_normcdf_hmean', ascending=False).iloc[:50]
    top_300_neu = term_freq_df2.sort_values(by='neu_normcdf_hmean', ascending=False).iloc[:50]

    top_300_pos.to_excel(target + "/top_300_pos.xlsx")
    top_300_neg.to_excel(target + "/top_300_neg.xlsx")
    top_300_neu.to_excel(target + "/top_300_neu.xlsx")


    sentiment_fit2 = tg_pipeline.fit(x_train, y_train)
    y_pred_test = sentiment_fit2.predict(x_test)

    # end of zipfs law tutorial
    # Showing the data as red when they are wrong predictions
    indexes = []
    for i in range(0, len(result_1.polarity)):
        indexes.append(i)

    result_1.index = indexes

    test_data_len = len(result_1)

    wrong_pred = []
    for i in result_1.index:
        if y_pred_test[i] != result_1.polarity.iloc[i]:
            wrong_pred.append(i)

    wrong_class = 'WrongPred alert alert-danger'




    #target = os.path.join(APP_ROOT, 'models/' + current_user.username + '/')
    #print(target)


    #if not os.path.isdir(target):
    #    os.mkdir(target)
    #with open(target + 'sentiment_fit.pkl', 'wb') as f:
    #    pickle.dump(sentiment_fit2, f)
    #with open(target + 'tg_cvec.pkl', 'wb') as f:
    #    pickle.dump(tg_cvec, f)


    return render_template('train_results_db.html', 
        df = df, 
        accuracy_avg = accuracy_avg, 
        count_all = count_all, 
        count_pos = count_pos,
        count_neu = count_neu,  
        count_neg = count_neg,
        feature_result_wosw = feature_result_wosw, 
        ug_labels = ug_labels, ug_values = ug_values, 
        ug_wocsw_labels = ug_wocsw_labels, 
        ug_wocsw_values = ug_wocsw_values, 
        ug_wosw_labels = ug_wosw_labels, 
        ug_wosw_values = ug_wosw_values, 
        bg_values = bg_values, tg_values = tg_values, 
        r_t_confusion = r_t_confusion, 
        r_t_class_report = r_t_class_report, 
        r_v_confusion = r_v_confusion, 
        r_v_class_report = r_v_class_report, 
        ugt_values = ugt_values, 
        bgt_values = bgt_values, 
        tgt_values = tgt_values, 
        comparing_algorithms = comparing_algorithms, 
        top_50_neg = top_50_neg, 
        top_50_neu = top_50_neu, 
        top_50_pos = top_50_pos, 
        result_1 = result_1, 
        result_2 = result_2    , 
        pos_labels = pos_labels,
        pos_values = pos_values,
        neg_labels = neg_labels,
        neg_values = neg_values, 
        neu_labels = neu_labels,
        neu_values = neu_values, 
        wrong_pred = wrong_pred, 
        wrong_class = wrong_class, 
        test_data_len = test_data_len, 
        y_pred_test = y_pred_test, 
        sentiment_fit2 = sentiment_fit2, 
        tg_cvec = tg_cvec, 
        form = form, 
        models = models
        )
    



@analyse.route('/train_results', methods = ['POST', 'GET'])
def train_results():
    if not current_user.is_authenticated:
        flash('You should be logged in to acced this page', 'info')
        next_page =  request.endpoint
        return redirect(url_for('users.login', next = next_page))
    if request.method != "POST":
        flash("You don't have permission accessing this page", 'info')
        return redirect(url_for('main.dashboard'))
    global tg_cvec, sentiment_fit2, df_save
    form = SaveModel()
    #df = pd.read_csv('YoutubeSpamMergedData.csv')
    target = os.path.join(APP_ROOT, 'trainingcsv/')
    print(target)

    if not os.path.isdir(target):
        os.mkdir(target)


    if request.method == "POST":
        startdate = request.form['startdate']
        enddate = request.form['enddate']
        #start = datetime.datetime(start)
        #end = datetime.datetime(end)
        startdate = startdate + ' 00:00:00'
        enddate = enddate + ' 00:00:00'
        date_str = type(startdate)
        # search by author or book
        cursor.execute("SELECT comment, polarity from training")
        conn.commit()
        data = cursor.fetchall()
        type_data = type(data)
        fetched_data = pd.DataFrame(list(data), columns = ['comment', 'polarity'])
        df = pd.DataFrame()
        df['text'] = fetched_data['comment'].astype(str)
        df['polarity'] = fetched_data['polarity']
        df_data = df[["text", "polarity"]]
        df_x = df_data['text']
        df_y = df_data.polarity
        df_save = df

    df_neg = df[df.polarity == -1]
    df_pos = df[df.polarity == 1]
    df_neu = df[df.polarity == 0]
    df_neg = df_neg.assign(polarity= 0)
    df_neu = df_neu.assign(polarity= 2)
    df_pos = df_pos.assign(polarity= 4)
    df3 = df_neg.append(df_neu, ignore_index=True)
    df3 = df3.append(df_pos, ignore_index=True)
    # NOW , it has only POSITIVES AND NEGATIVES 
    df = df3

    my_df = df

    # get the lengths of the three categories
    len_neg = len(df_neg)
    len_neu = len(df_neu)
    len_pos = len(df_pos)

    min_n_cats = min(len_neg, len_neu, len_pos)

    n_train =  int(min_n_cats * 80 // 100)
    n_test = int(min_n_cats * 10 // 100)
    n_validate = int(min_n_cats * 10 // 100)

    total = n_train + n_test + n_validate

    my_df['text'] = my_df.text.astype(str)

    df_neg = my_df[my_df.polarity == 0][0:n_train]
    df_neu = my_df[my_df.polarity == 2][0:n_train]
    df_pos = my_df[my_df.polarity == 4][0:n_train]
    frames = [df_neg, df_neu, df_pos]
    result = pd.concat(frames, ignore_index=True)
    # THIS VARIABLE HAS ALL THE TRAINING DATA SEPARATED 
    result

    # This will give you the number of feature names

    df.head()

    x = result.text
    y = result.polarity
    
    SEED = 2000

    # assigning x and y directly as the train set
    x_train = x
    y_train = y 


    # setting the x_test and y_test 
    # Testing Sample
    df_neg_1 = my_df[my_df.polarity == 0][n_train + 1:n_train + n_test]
    df_neu_1 = my_df[my_df.polarity == 2][n_train + 1:n_train + n_test]
    df_pos_1 = my_df[my_df.polarity == 4][n_train + 1:n_train + n_test]
    frames_1 = [df_neg_1, df_neu_1, df_pos_1]
    result_1 = pd.concat(frames_1)
    x_test = result_1.text
    y_test = result_1.polarity

    #Validation sample
    df_neg_2 = my_df[my_df.polarity == 0][n_train + n_test + 1:n_train + n_test + n_validate]
    df_neu_2 = my_df[my_df.polarity == 2][n_train + n_test + 1:n_train + n_test + n_validate]
    df_pos_2 = my_df[my_df.polarity == 4][n_train + n_test + 1:n_train + n_test + n_validate]
    frames_2 = [df_neg_2, df_neu_2, df_pos_2]
    result_2 = pd.concat(frames_2)
    x_validation = result_2.text
    y_validation = result_2.polarity  



    print ("Train set has total {0} entries with {1:.2f}% negative, {2:.2f}% Good, {3:.2f}% positive".format(len(x_train),
        (len(x_train[y_train == 0]) / (len(x_train)*1.))*100,
        (len(x_train[y_train == 2]) / (len(x_train)*1.))*100,
        (len(x_train[y_train == 4]) / (len(x_train)*1.))*100) )


    print ("Validation set has total {0} entries with {1:.2f}% negative, {2:.2f}% Good, {3:.2f}% positive".format(len(x_validation),
        (len(x_validation[y_validation == 0]) / (len(x_validation)*1.))*100,
        (len(x_validation[y_validation == 2]) / (len(x_validation)*1.))*100,
        (len(x_validation[y_validation == 4]) / (len(x_validation)*1.))*100) )

    print ("Test set has total {0} entries with {1:.2f}% negative, {2:.2f}% Good, {3:.2f}% positive".format(len(x_test),
        (len(x_test[y_test == 0]) / (len(x_test)*1.))*100,
        (len(x_test[y_test == 2]) / (len(x_test)*1.))*100,
        (len(x_test[y_test == 4]) / (len(x_test)*1.))*100) )


    from textblob import TextBlob
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import classification_report, confusion_matrix
    # Feature extraction
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from time import time




    def accuracy_summary(pipeline, x_train, y_train, x_test, y_test):
        if len(x_test[y_test == 0]) / (len(x_test)*1.) > 0.33:
            null_accuracy = len(x_test[y_test == 0]) / (len(x_test)*1.)
        else:
            null_accuracy = 1. - (len(x_test[y_test == 0]) / (len(x_test)*1.))
        t0 = time()
        sentiment_fit = pipeline.fit(x_train, y_train)
        y_pred = sentiment_fit.predict(x_test)
        train_test_time = time() - t0
        accuracy = accuracy_score(y_test, y_pred)
        print ("null accuracy: {0:.2f}%".format(null_accuracy*100) )
        print ("accuracy score: {0:.2f}%".format(accuracy*100) )
        if accuracy > null_accuracy:
            print ("model is {0:.2f}% more accurate than null accuracy".format((accuracy-null_accuracy)*100) )
        elif accuracy == null_accuracy:
            print ("model has the same accuracy with the null accuracy")
        else:
            print ("model is {0:.2f}% less accurate than null accuracy".format((null_accuracy-accuracy)*100) )
        print ("train and test time: {0:.2f}s".format(train_test_time) )
        print ("-"*80 )
        return accuracy, train_test_time


    cvec = CountVectorizer()
    # get the number of feature names , but first we fit the text
    cvec.fit(x)
    len(cvec.get_feature_names())
    num_features = len(cvec.get_feature_names())

    lr = LogisticRegression(solver='newton-cg', multi_class='multinomial')
    n_features = np.arange(1000,num_features,1000)


    # get the stemmer function, this is optional 
    # with stemming words
    import nltk
    from nltk.stem.isri import ISRIStemmer
    stemmer = ISRIStemmer()
    analyzer = CountVectorizer().build_analyzer()

    def stemmed_words(doc):
        return (stemmer.stem(w) for w in analyzer(doc))
    # remove the above code as it is used in another function(if any errors)


    def nfeature_accuracy_checker(vectorizer=cvec, n_features=n_features, stop_words=None, ngram_range=(1, 1), classifier=lr):
        result = []
        print (classifier)
        print ("\n")
        for n in n_features:
            vectorizer.set_params(stop_words=stop_words, max_features=n, ngram_range=ngram_range)
            checker_pipeline = Pipeline([
                ('vectorizer', vectorizer),
                ('classifier', classifier)
            ])
            print ("Validation result for {} features".format(n) )
            nfeature_accuracy,tt_time = accuracy_summary(checker_pipeline, x_train, y_train, x_validation, y_validation)
            result.append((n,nfeature_accuracy,tt_time))
        return result


    my_stop_words = ['selmane', 'aissa', 'brahim']

    print ("RESULT FOR UNIGRAM WITHOUT STOP WORDS\n")
    feature_result_wosw = nfeature_accuracy_checker(stop_words = 'english') 

    print ("RESULT FOR UNIGRAM WITH STOP WORDS\n")
    feature_result_ug = nfeature_accuracy_checker() 

    print ("RESULT FOR UNIGRAM WITHOUT CUSTOM STOP WORDS (Top 10 frequent words)\n" )
    feature_result_wocsw = nfeature_accuracy_checker(stop_words=my_stop_words) 


    nfeatures_plot_ug = pd.DataFrame(feature_result_ug,columns=['nfeatures','validation_accuracy','train_test_time'])
    nfeatures_plot_ug_wocsw = pd.DataFrame(feature_result_wocsw,columns=['nfeatures','validation_accuracy','train_test_time'])
    nfeatures_plot_ug_wosw = pd.DataFrame(feature_result_wosw,columns=['nfeatures','validation_accuracy','train_test_time'])

    plt.figure(figsize=(8,6))
    plt.plot(nfeatures_plot_ug.nfeatures, nfeatures_plot_ug.validation_accuracy, label='with stop words')
    plt.plot(nfeatures_plot_ug_wosw.nfeatures, nfeatures_plot_ug_wosw.validation_accuracy,label='without stop words')
    plt.plot(nfeatures_plot_ug_wocsw.nfeatures, nfeatures_plot_ug_wocsw.validation_accuracy,label='without custom stop words')
    plt.title("Without stop words VS With stop words (Unigram): Accuracy")
    plt.xlabel("Number of features")
    plt.ylabel("Validation set accuracy")
    plt.legend()

    ug_labels = nfeatures_plot_ug.nfeatures
    ug_values = nfeatures_plot_ug.validation_accuracy

    ug_wocsw_labels = nfeatures_plot_ug_wocsw.nfeatures
    ug_wocsw_values = nfeatures_plot_ug_wocsw.validation_accuracy

    ug_wosw_labels = nfeatures_plot_ug_wosw.nfeatures
    ug_wosw_values = nfeatures_plot_ug_wosw.validation_accuracy


    # Showing the tests for bigrams and trigrams 
    print ("RESULT FOR BIGRAM WITH STOP WORDS\n")
    feature_result_bg = nfeature_accuracy_checker(ngram_range=(1, 2))

    print ("RESULT FOR TRIGRAM WITH STOP WORDS\n")
    feature_result_tg = nfeature_accuracy_checker(ngram_range=(1, 3))


    nfeatures_plot_tg = pd.DataFrame(feature_result_tg,columns=['nfeatures','validation_accuracy','train_test_time'])
    nfeatures_plot_bg = pd.DataFrame(feature_result_bg,columns=['nfeatures','validation_accuracy','train_test_time'])
    nfeatures_plot_ug = pd.DataFrame(feature_result_ug,columns=['nfeatures','validation_accuracy','train_test_time'])

    plt.figure(figsize=(8,6))
    plt.plot(nfeatures_plot_tg.nfeatures, nfeatures_plot_tg.validation_accuracy,label='trigram')
    plt.plot(nfeatures_plot_bg.nfeatures, nfeatures_plot_bg.validation_accuracy,label='bigram')
    plt.plot(nfeatures_plot_ug.nfeatures, nfeatures_plot_ug.validation_accuracy, label='unigram')
    plt.title("N-gram(1~3) test result : Accuracy")
    plt.xlabel("Number of features")
    plt.ylabel("Validation set accuracy")
    plt.legend()
    # plt.show()

    bg_values = nfeatures_plot_bg.validation_accuracy
    tg_values = nfeatures_plot_tg.validation_accuracy

    def train_test_and_evaluate(pipeline, x_train, y_train, x_test, y_test):
        if len(x_test[y_test == 0]) / (len(x_test)*1.) > 0.33:
            null_accuracy = len(x_test[y_test == 0]) / (len(x_test)*1.)
        else:
            null_accuracy = 1. - (len(x_test[y_test == 0]) / (len(x_test)*1.))
        sentiment_fit = pipeline.fit(x_train, y_train)
        y_pred = sentiment_fit.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        conmat = np.array(confusion_matrix(y_test, y_pred, labels=[0,2,4]))
        confusion = pd.DataFrame(conmat, index=['negative', 'neutral', 'positive'],
                             columns=['predicted_negative','predicted_neutral','predicted_positive'])
        class_report = classification_report(y_test, y_pred, target_names=['negative','neutral', 'positive'], output_dict=True)
        class_report_df = pd.DataFrame(class_report).transpose()
        print ("null accuracy: {0:.2f}%".format(null_accuracy*100) )
        print ("accuracy score: {0:.2f}%".format(accuracy*100) )
        if accuracy > null_accuracy:
            print ("model is {0:.2f}% more accurate than null accuracy".format((accuracy-null_accuracy)*100) )
        elif accuracy == null_accuracy:
            print ("model has the same accuracy with the null accuracy" )
        else:
            print ("model is {0:.2f}% less accurate than null accuracy".format((null_accuracy-accuracy)*100) )
        print ("-"*80 )
        print ("Confusion Matrix\n" )
        print (confusion)
        print ("-"*80 )
        print ("Classification Report\n" )
        print (classification_report(y_test, y_pred, target_names=['negative', 'neutral', 'positive']) )
        return confusion , class_report_df


    tg_cvec = CountVectorizer(max_features=num_features,ngram_range=(1, 3))
    tg_pipeline = Pipeline([
            ('vectorizer', tg_cvec),
            ('classifier', lr)
        ])
    r_t_confusion, r_t_class_report = train_test_and_evaluate(tg_pipeline, x_train, y_train, x_test, y_test)

    r_v_confusion, r_v_class_report = train_test_and_evaluate(tg_pipeline, x_train, y_train,x_validation, y_validation )

    # getting the number of true positives and negatives to show on the website 

    true_negative = r_t_confusion["predicted_negative"][0]
    false_negative1 = r_t_confusion["predicted_negative"][1]
    false_negative2 = r_t_confusion["predicted_negative"][2]

    false_neutral1 = r_t_confusion["predicted_neutral"][0]
    true_neutral = r_t_confusion["predicted_neutral"][1]
    false_neutral2 = r_t_confusion["predicted_neutral"][2]

    false_positive1 = r_t_confusion["predicted_positive"][0]
    false_positive2 = r_t_confusion["predicted_positive"][1]
    true_positive = r_t_confusion["predicted_positive"][2]

    count_pos = true_positive + false_positive1 + false_positive2
    count_neu = true_neutral + false_neutral1 + false_neutral2
    count_neg = true_negative + false_negative1 + false_negative2

    count_all = count_pos + count_neu + count_neg
    accuracy_avg = r_t_class_report.precision[3]
    accuracy_avg = float("{0:.3f}".format(accuracy_avg))
    # there are some functions that have been used before, and 
    # i am testing on tf-idf , unigrams, bigrams, trigrams 
    # and printing the results on a graph
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from time import time


    def accuracy_summary(pipeline, x_train, y_train, x_test, y_test):
        if len(x_test[y_test == 0]) / (len(x_test)*1.) > 0.33:
            null_accuracy = len(x_test[y_test == 0]) / (len(x_test)*1.)
        else:
            null_accuracy = 1. - (len(x_test[y_test == 0]) / (len(x_test)*1.))
        t0 = time()
        sentiment_fit = pipeline.fit(x_train, y_train)
        y_pred = sentiment_fit.predict(x_test)
        train_test_time = time() - t0
        accuracy = accuracy_score(y_test, y_pred)
        print ("null accuracy: {0:.2f}%".format(null_accuracy*100) )
        print ("accuracy score: {0:.2f}%".format(accuracy*100) )
        if accuracy > null_accuracy:
            print ("model is {0:.2f}% more accurate than null accuracy".format((accuracy-null_accuracy)*100) )
        elif accuracy == null_accuracy:
            print ("model has the same accuracy with the null accuracy")
        else:
            print ("model is {0:.2f}% less accurate than null accuracy".format((null_accuracy-accuracy)*100) )
        print ("train and test time: {0:.2f}s".format(train_test_time) )
        print ("-"*80 )
        return (accuracy, train_test_time )

    cvec = CountVectorizer()
    lr = LogisticRegression(solver='newton-cg', multi_class='multinomial')
    n_features = np.arange(1000,num_features,1000)

    def nfeature_accuracy_checker(vectorizer=cvec, n_features=n_features, stop_words=None, ngram_range=(1, 1), classifier=lr):
        result = []
        print (classifier)
        print ("\n")
        for n in n_features:
            vectorizer.set_params(stop_words=stop_words, max_features=n, ngram_range=ngram_range)
            checker_pipeline = Pipeline([
                ('vectorizer', vectorizer),
                ('classifier', classifier)
            ])
            print ("Validation result for {} features".format(n) )
            nfeature_accuracy,tt_time = accuracy_summary(checker_pipeline, x_train, y_train, x_validation, y_validation)
            result.append((n,nfeature_accuracy,tt_time))
        return result

    from sklearn.feature_extraction.text import TfidfVectorizer
    tvec = TfidfVectorizer()

    feature_result_ugt = nfeature_accuracy_checker(vectorizer=tvec)
    feature_result_bgt = nfeature_accuracy_checker(vectorizer=tvec,ngram_range=(1, 2))
    feature_result_tgt = nfeature_accuracy_checker(vectorizer=tvec,ngram_range=(1, 3))


    nfeatures_plot_tgt = pd.DataFrame(feature_result_tgt,columns=['nfeatures','validation_accuracy','train_test_time'])
    nfeatures_plot_bgt = pd.DataFrame(feature_result_bgt,columns=['nfeatures','validation_accuracy','train_test_time'])
    nfeatures_plot_ugt = pd.DataFrame(feature_result_ugt,columns=['nfeatures','validation_accuracy','train_test_time'])
    plt.figure(figsize=(8,6))
    plt.plot(nfeatures_plot_tgt.nfeatures, nfeatures_plot_tgt.validation_accuracy,label='trigram tfidf vectorizer',color='royalblue')
    plt.plot(nfeatures_plot_tg.nfeatures, nfeatures_plot_tg.validation_accuracy,label='trigram count vectorizer',linestyle=':', color='royalblue')
    plt.plot(nfeatures_plot_bgt.nfeatures, nfeatures_plot_bgt.validation_accuracy,label='bigram tfidf vectorizer',color='orangered')
    plt.plot(nfeatures_plot_bg.nfeatures, nfeatures_plot_bg.validation_accuracy,label='bigram count vectorizer',linestyle=':',color='orangered')
    plt.plot(nfeatures_plot_ugt.nfeatures, nfeatures_plot_ugt.validation_accuracy, label='unigram tfidf vectorizer',color='gold')
    plt.plot(nfeatures_plot_ug.nfeatures, nfeatures_plot_ug.validation_accuracy, label='unigram count vectorizer',linestyle=':',color='gold')
    plt.title("N-gram(1~3) test result : Accuracy")
    plt.xlabel("Number of features")
    plt.ylabel("Validation set accuracy")
    plt.legend()
    # plt.show()

    ugt_values = nfeatures_plot_bgt.validation_accuracy
    bgt_values = nfeatures_plot_bgt.validation_accuracy
    tgt_values = nfeatures_plot_tgt.validation_accuracy



    # Comparing with other classifiers
    # Now , we compare using different algorithms:
    from sklearn.svm import LinearSVC
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.naive_bayes import MultinomialNB, BernoulliNB
    from sklearn.linear_model import RidgeClassifier
    from sklearn.linear_model import PassiveAggressiveClassifier
    from sklearn.linear_model import Perceptron
    from sklearn.neighbors import NearestCentroid
    from sklearn.feature_selection import SelectFromModel
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer

    names = ["Logistic Regression", "Linear SVC", "LinearSVC with L1-based feature selection","Multinomial NB", 
             "Bernoulli NB", "Ridge Classifier", "AdaBoost", "Perceptron","Passive-Aggresive", "Nearest Centroid"]
    classifiers = [
        LogisticRegression(solver='newton-cg', multi_class='multinomial'),
        LinearSVC(),
        Pipeline([
      ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False))),
      ('classification', LinearSVC(penalty="l2"))]),
        MultinomialNB(),
        BernoulliNB(),
        RidgeClassifier(),
        AdaBoostClassifier(),
        Perceptron(),
        PassiveAggressiveClassifier(),
        NearestCentroid()
        ]
    zipped_clf = zip(names,classifiers)

    tvec = TfidfVectorizer()

    def classifier_comparator(vectorizer=tvec, n_features= num_features, stop_words=None, ngram_range=(1, 1), classifier=zipped_clf):
        result = []
        vectorizer.set_params(stop_words=stop_words, max_features=n_features, ngram_range=ngram_range)
        for n,c in classifier:
            checker_pipeline = Pipeline([
                ('vectorizer', vectorizer),
                ('classifier', c)
            ])
            print ("Validation result for {}".format(n) )
            print (c)
            clf_accuracy,tt_time = accuracy_summary(checker_pipeline, x_train, y_train, x_validation, y_validation)
            result.append((n,clf_accuracy,tt_time))
        return result

    comparing_algorithms = classifier_comparator(n_features=num_features,ngram_range=(1,3))



    comparing_algorithms = pd.DataFrame(comparing_algorithms, columns=['Model','Validation set accuracy', 'Train and test time'])









    # Getting The positives and Negatives 
    my_df = result


    # Creating the count vectorizer for later use 
    from sklearn.feature_extraction.text import CountVectorizer
    cvec = CountVectorizer()
    my_df['text'] = my_df.text.astype(str)
    cvec.fit(my_df.text)

    # This will give you the number of feature names
    len(cvec.get_feature_names())
    # this above variable is important 


    # This will generate the terpm frequency for pos and neg and then combined 
    neg_doc_matrix = cvec.transform(my_df[my_df.polarity == 0].text)
    neu_doc_matrix = cvec.transform(my_df[my_df.polarity == 2].text)
    pos_doc_matrix = cvec.transform(my_df[my_df.polarity == 4].text)
    neg_tf = np.sum(neg_doc_matrix,axis=0)
    neu_tf = np.sum(neu_doc_matrix,axis=0)
    pos_tf = np.sum(pos_doc_matrix,axis=0)
    neg = np.squeeze(np.asarray(neg_tf))
    neu = np.squeeze(np.asarray(neu_tf))
    pos = np.squeeze(np.asarray(pos_tf))
    words = cvec.get_feature_names()
    term_freq_df = pd.DataFrame([neg,neu,pos], columns = cvec.get_feature_names()).transpose()



    document_matrix = cvec.transform(my_df.text)
    my_df[my_df.polarity == 0].tail()

    neg_batches = np.linspace(0,4500,100).astype(int)
    i=0
    neg_tf = []
    while i < len(neg_batches)-1:
        batch_result = np.sum(document_matrix[neg_batches[i]:neg_batches[i+1]].toarray(),axis=0)
        neg_tf.append(batch_result)
        if (i % 10 == 0) | (i == len(neg_batches)-2):
            print(neg_batches[i+1],"entries' term freuquency calculated")
        i += 1


    neu_batches = np.linspace(4501,9000,100).astype(int)
    i=0
    neu_tf = []
    while i < len(neu_batches)-1:
        batch_result = np.sum(document_matrix[neu_batches[i]:neu_batches[i+1]].toarray(),axis=0)
        neu_tf.append(batch_result)
        if (i % 10 == 0) | (i == len(neu_batches)-2):
            print(neu_batches[i+1],"entries' term freuquency calculated")
        i += 1


    pos_batches = np.linspace(9001,13500,100).astype(int)
    i=0
    pos_tf = []
    while i < len(pos_batches)-1:
        batch_result = np.sum(document_matrix[pos_batches[i]:pos_batches[i+1]].toarray(),axis=0)
        pos_tf.append(batch_result)
        if (i % 10 == 0) | (i == len(pos_batches)-2):
            print(pos_batches[i+1],"entries' term freuquency calculated")
        i += 1


    # Now we have positives and negatives 
    term_freq_df.columns = ['negative', 'neutral', 'positive']
    term_freq_df['total'] = term_freq_df['negative']+ term_freq_df['neutral'] + term_freq_df['positive']
    term_freq_df.sort_values(by='total', ascending=False).iloc[:10]



    #term_freq_df2 = term_freq_df
    term_freq_df2 =  term_freq_df

    term_freq_df2['pos_rate'] = term_freq_df2['positive'] * 1./term_freq_df2['total']
    term_freq_df2.sort_values(by='pos_rate', ascending=False).iloc[:10]


    term_freq_df2['pos_freq_pct'] = term_freq_df2['positive'] * 1./term_freq_df2['positive'].sum()
    term_freq_df2.sort_values(by='pos_freq_pct', ascending=False).iloc[:10]

    from scipy.stats import hmean

    term_freq_df2['pos_hmean'] = term_freq_df2.apply(lambda x: (hmean([x['pos_rate'], x['pos_freq_pct']])                                                               if x['pos_rate'] > 0 and x['pos_freq_pct'] > 0 else 0), axis=1)
                                                           
    term_freq_df2.sort_values(by='pos_hmean', ascending=False).iloc[:10]



    from scipy.stats import norm
    from scipy.stats import gmean

    #from scipy.stats import *



    def normcdf(x):
        return norm.cdf(x, x.mean(), x.std())


    term_freq_df2['pos_rate_normcdf'] = normcdf(term_freq_df2['pos_rate'])

    term_freq_df2['pos_freq_pct_normcdf'] = normcdf(term_freq_df2['pos_freq_pct'])

    term_freq_df2['pos_normcdf_hmean'] = gmean([term_freq_df2['pos_rate_normcdf'], term_freq_df2['pos_freq_pct_normcdf']])


    # Now the pos_rate_normcdf is the column to be shown for Super words



    term_freq_df2['neg_rate'] = term_freq_df2['negative'] * 1./term_freq_df2['total']

    term_freq_df2['neg_freq_pct'] = term_freq_df2['negative'] * 1./term_freq_df2['negative'].sum()

    term_freq_df2['neg_hmean'] = term_freq_df2.apply(lambda x: (hmean([x['neg_rate'], x['neg_freq_pct']]) if x['neg_rate'] > 0 and x['neg_freq_pct'] > 0 else 0), axis=1)
                                                           
    term_freq_df2['neg_rate_normcdf'] = normcdf(term_freq_df2['neg_rate'])

    term_freq_df2['neg_freq_pct_normcdf'] = normcdf(term_freq_df2['neg_freq_pct'])

    term_freq_df2['neg_normcdf_hmean'] = gmean([term_freq_df2['neg_rate_normcdf'], term_freq_df2['neg_freq_pct_normcdf']])



    # Good 
    term_freq_df2['neu_rate'] = term_freq_df2['neutral'] * 1./term_freq_df2['total']

    term_freq_df2['neu_freq_pct'] = term_freq_df2['neutral'] * 1./term_freq_df2['neutral'].sum()

    term_freq_df2['neu_hmean'] = term_freq_df2.apply(lambda x: (hmean([x['neu_rate'], x['neu_freq_pct']]) if x['neu_rate'] > 0 and x['neu_freq_pct'] > 0 else 0), axis=1)

    term_freq_df2['neu_rate_normcdf'] = normcdf(term_freq_df2['neu_rate'])

    term_freq_df2['neu_freq_pct_normcdf'] = normcdf(term_freq_df2['neu_freq_pct'])

    term_freq_df2['neu_normcdf_hmean'] = gmean([term_freq_df2['neu_rate_normcdf'], term_freq_df2['neu_freq_pct_normcdf']])





    term_freq_df2['word'] = words

    top_50_pos = term_freq_df2.sort_values(by='pos_normcdf_hmean',ascending=False).iloc[:50]
    top_50_neg = term_freq_df2.sort_values(by='neg_normcdf_hmean', ascending=False).iloc[:50]
    top_50_neu = term_freq_df2.sort_values(by='neu_normcdf_hmean', ascending=False).iloc[:50]

    pos_labels = top_50_pos['word']
    pos_values = top_50_pos['positive']

    neu_labels = top_50_neu['word']
    neu_values = top_50_neu['neutral']

    neg_labels = top_50_neg['word']
    neg_values = top_50_neg['negative']



    sentiment_fit2 = tg_pipeline.fit(x_train, y_train)
    y_pred_test = sentiment_fit2.predict(x_test)

    # end of zipfs law tutorial
    # Showing the data as red when they are wrong predictions
    indexes = []
    for i in range(0, len(result_1.polarity)):
        indexes.append(i)

    result_1.index = indexes

    test_data_len = len(result_1)

    wrong_pred = []
    for i in result_1.index:
        if y_pred_test[i] != result_1.polarity.iloc[i]:
            wrong_pred.append(i)

    wrong_class = 'WrongPred alert alert-danger'




    #target = os.path.join(APP_ROOT, 'models/' + current_user.username + '/')
    #print(target)


    #if not os.path.isdir(target):
    #    os.mkdir(target)
    #with open(target + 'sentiment_fit.pkl', 'wb') as f:
    #    pickle.dump(sentiment_fit2, f)
    #with open(target + 'tg_cvec.pkl', 'wb') as f:
    #    pickle.dump(tg_cvec, f)



    return render_template('train_results.html', 
        df = df, 
        accuracy_avg = accuracy_avg, 
        count_all = count_all, 
        count_pos = count_pos,
        count_neu = count_neu,  
        count_neg = count_neg,
        feature_result_wosw = feature_result_wosw, 
        ug_labels = ug_labels, ug_values = ug_values, 
        ug_wocsw_labels = ug_wocsw_labels, 
        ug_wocsw_values = ug_wocsw_values, 
        ug_wosw_labels = ug_wosw_labels, 
        ug_wosw_values = ug_wosw_values, 
        bg_values = bg_values, tg_values = tg_values, 
        r_t_confusion = r_t_confusion, 
        r_t_class_report = r_t_class_report, 
        r_v_confusion = r_v_confusion, 
        r_v_class_report = r_v_class_report, 
        ugt_values = ugt_values, 
        bgt_values = bgt_values, 
        tgt_values = tgt_values, 
        comparing_algorithms = comparing_algorithms, 
        top_50_neg = top_50_neg, 
        top_50_neu = top_50_neu, 
        top_50_pos = top_50_pos, 
        result_1 = result_1, 
        result_2 = result_2    , 
        pos_labels = pos_labels,
        pos_values = pos_values,
        neg_labels = neg_labels,
        neg_values = neg_values, 
        neu_labels = neu_labels,
        neu_values = neu_values, 
        wrong_pred = wrong_pred, 
        wrong_class = wrong_class, 
        test_data_len = test_data_len, 
        y_pred_test = y_pred_test, 
        sentiment_fit2 = sentiment_fit2, 
        tg_cvec = tg_cvec, 
        form = form
        )































@analyse.route('/clean_dareja')
def clean_dareja():
    if not current_user.is_authenticated:
        flash('You should be logged in to acced this page', 'info')
        next_page =  request.endpoint
        return redirect(url_for('users.login', next = next_page))
    title = "Cleaning Darja"
    return render_template('clean_dareja.html', title = title)



@analyse.route('/results_cleaning', methods = ['POST', 'GET'])
def results_cleaning():
    global results_df
    if not current_user.is_authenticated:
        flash('You should be logged in to acced this page', 'info')
        next_page =  request.endpoint
        return redirect(url_for('users.login', next = next_page))

    if request.method != "POST":
        flash("You don't have permission accessing this page", 'info')
        return redirect(url_for('main.dashboard'))
            
    #df = pd.read_csv('YoutubeSpamMergedData.csv')
    target = os.path.join(APP_ROOT, 'cleaningexcel/')
    print(target)

    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist("file"):
        print(file)
        filename = file.filename
        destination = "/".join([target, filename])
        print(destination)
        file.save(destination)
        df = pd.read_excel(os.path.join(APP_ROOT, 'cleaningexcel/'+filename))
        df['text'] = df['text'].astype(str)
        df_data = df[["text", "polarity"]]
        #df_data = clean_text(df_content = df_data, column_name = 'text', stopword = 'french')
        #df_data['text'] = df_data['comment']
        df_x = df_data['text']
        df_y = df_data['polarity']
        df_save = df


    df_save['text'] = df_save.text.astype(str)


    x_train_clean = clean_text_for_training(df_content = df_save, column_name = 'text', stopword = 'french')
    #x_train_clean['text'] = x_train_clean['comment']
    #x = x_train_clean['text'] 
    
    SEED = 2000

    new_df = pd.concat([x_train_clean['comment'], df_data['polarity']], axis=1, keys=['Comment', 'Polarity'])
    results_df = new_df


    return render_template('results_cleaning.html', 
        new_df = new_df, 
        x_train_clean = x_train_clean, 
        results_df = results_df
        )







import torch.nn as nn
from string import punctuation



import torch
train_on_gpu = torch.cuda.is_available()

class SentimentRNN(nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        """
        Initialize the model by setting up the layers.
        """
        super(SentimentRNN, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, 
                            dropout=drop_prob, batch_first=True)
        # dropout layer
        self.dropout = nn.Dropout(0.3)
        # linear and sigmoid layers
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()
        
    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        batch_size = x.size(0)
        # embeddings and lstm_out
        x = x.long()
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        # stack up lstm outputs
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        # dropout and fully-connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)
        # sigmoid function
        sig_out = self.sig(out)
        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1] # get last batch of labels
        # return last sigmoid output and hidden state
        return sig_out, hidden
    
    
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        return hidden










@analyse.route('/sentiment_rnn', methods=['POST', 'GET'])
def sentiment_rnn():
    cursor.execute("""SELECT name, type FROM `models`""")
    my_models = cursor.fetchall()
    models = []
    for i in my_models:
        models.append(i[0])

    return render_template('sentiment_rnn.html', models = models
            )


@analyse.route('/sentiment_rnn_result', methods=['POST', 'GET'])
def sentiment_rnn_result():
    if not current_user.is_authenticated:
        flash('You should be logged in to acced this page', 'info')
        next_page =  request.endpoint
        return redirect(url_for('users.login', next = next_page))
    if request.method != "POST":
        flash("You don't have permission accessing this page", 'info')
        return redirect(url_for('main.dashboard'))
    # save excel if post request was made
    if request.method == "POST":
        secteur = request.form['secteur']
        model = request.form['model']
        stopwords = request.form['stopwords']
        all_comments, df_x, df, column_name = save_excel()

    cursor.execute("""SELECT name, type, seq_length, n_layers 
        FROM `models` WHERE name = %s""", ( model, ))
    my_data = cursor.fetchall()
    model_info = pd.DataFrame(list(my_data), columns=['name', 'type', 'seq_length', 'n_layers'])

    target = os.path.join(APP_ROOT, 'models/' + model + '/')
    # get model infos
    n_layers = model_info['n_layers'][0]
    seq_length=model_info['seq_length'][0]
    model_type = model_info['type'][0]
    
    def tokenize_review(test_review, vocab_to_int = []):
        test_review = test_review.lower() # lowercase
        # get rid of punctuation
        test_text = ''.join([c for c in test_review if c not in punctuation])
        # splitting by spaces
        test_words = test_text.split()
        # tokens
        test_ints = []
        if vocab_to_int:
            test_ints.append([vocab_to_int[word] if word in vocab_to_int.keys() else 0 for word in test_text.split()])
            test_ints = [[i for i in test_ints[0] if i != 0]]
        else:
            test_ints = [0]
        return test_ints



    def pad_features(reviews_ints, seq_length):
        ''' Return features of review_ints, where each review is padded with 0'sor truncated to the input seq_length.'''
        # getting the correct rows x cols shape
        features = np.zeros((len(reviews_ints), seq_length), dtype=int)
        if np.count_nonzero(features) != 0:
            for i, row in enumerate(reviews_ints):
                features[i, -len(row):] = np.array(row)[:seq_length]
        # return features
        return features




    def predict(net, test_review, sequence_length=200, vocab_to_int = []):
        net.eval()
        # tokenize review
        test_ints = tokenize_review(test_review, vocab_to_int = vocab_to_int)
        # pad tokenized sequence
        seq_length = sequence_length
        features = pad_features(test_ints, seq_length)
        # convert to tensor to pass into your model
        feature_tensor = torch.from_numpy(features)
        batch_size = feature_tensor.size(0)
        # initialize hidden state
        h = net.init_hidden(batch_size)
        if(train_on_gpu):
            feature_tensor = feature_tensor.cuda()
        # get the output from the model
        output, h = net(feature_tensor, h)
        # convert output probabilities to predicted class (0 or 1)
        pred = torch.round(output.squeeze()) 
        # printing output value, before rounding
        print('Prediction value, pre-rounding: {:.6f}'.format(output.item()))
        # print custom response
        if (output.item() > 0.40):
            polarity_pred = 'positive'
        elif (output.item() >= 0.40 and output.item() <= 0.60 ):
            polarity_pred = 'neutre'
        else:
            polarity_pred = 'negative'
        return polarity_pred


    def get_sentiment_analysis2(table = '', model = 'threeClass', type="rnn", vocab_to_int = []):
        ### Load and Test
        # Use a default model 
        target = os.path.join(APP_ROOT, 'models/' + model + '/')
        # Specify a path
        from sklearn.externals import joblib 
        cursor.execute("""SELECT name, type, seq_length, n_layers, user_id
            FROM `models` WHERE name = %s""", ( model, ))
        my_data = cursor.fetchall()
        model_info = pd.DataFrame(list(my_data), columns=['name', 'type', 'seq_length', 'n_layers', 'user_id'])
        if type == 'rnn':
            from sklearn.externals import joblib 
            vocab_to_int = open( target + "vocab_to_int.pkl", "rb")
            vocab_to_int = joblib.load(vocab_to_int)
            vocab_size = len(vocab_to_int)+1
            output_size = 1
            embedding_dim = 400
            hidden_dim = 256
            n_layers = model_info['n_layers'][0]
            seq_length = model_info['seq_length'][0]
            net = SentimentRNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
            #if(train_on_gpu):
            #    print('Training on GPU.')
            #else:
            #    print('No GPU available, training on CPU.')
            PATH = target + "state_dict_model.pt"
            net.load_state_dict(torch.load(PATH))
            net.eval()
            my_prediction = []
            results_df = pd.DataFrame()
            results_df['Comment'] = table.Comment
            for i in table['Comment']:
                my_str = i
                print(i)
                sent = predict(net, i, seq_length)
                my_prediction.append(sent)
            results_df['Polarity']  = my_prediction
        elif type == "regression":
            lr = LogisticRegression()
            if model == 'default':
                ytb_model = open(target +"/tg_cvec.pkl", "rb")
                tg_cvec = joblib.load(ytb_model)
                ytb_model = open(test_path  +"/sentiment_fit.pkl", "rb")
            else:
                ytb_model = open( target +"/tg_cvec.pkl", "rb")
                tg_cvec = joblib.load(ytb_model)
                ytb_model = open(target +"/sentiment_fit.pkl", "rb")
            sentiment_fit = joblib.load(ytb_model)
            tg_pipeline = Pipeline([
                ('vectorizer', tg_cvec),
                ('classifier', lr)
            ])
            positive_words = get_determining_words("positive")
            negative_words = get_determining_words("negative")
            #df = pd.read_excel("./test-dareja.xlsx", encoding='utf-8')
            results_df = pd.DataFrame()
            results_df['Comment'] = table.Comment
            s2 = pd.Series(results_df['Comment'])
            my_prediction = sentiment_fit.predict(s2)
            my_prediction_length = len(my_prediction)
            #i = 0
            #for list_element in results_df.Comment:
            #    print(list_element)
            #    negative_test = [ele for ele in negative_words if(ele in list_element)]
            #    positive_test = [ele for ele in positive_words if(ele in list_element)]
            #    if negative_test:
            #        my_prediction[i] = 0
            #    elif positive_test:
            #        my_prediction[i] = 4
            #    i += 1
            test_preds = []
            i = 0
            for i in my_prediction:
                if i == 4: 
                    my_prediction_cat = 'Super'
                elif i == 2:
                    my_prediction_cat = 'Good'
                else:
                    my_prediction_cat = 'Bad'
                test_preds.append(my_prediction_cat)
                i = i + 1
            results_df['Polarity']  = test_preds
        results_df_pos = results_df[results_df['Polarity'] == 'Super']
        results_df_neu = results_df[results_df['Polarity'] == 'Good']
        results_df_neg = results_df[results_df['Polarity'] == 'Bad']
        pos_percentage = round(len(results_df[results_df['Polarity'] == "Super"]) * 100/len(results_df['Polarity']), 2)
        neu_percentage = round(len(results_df[results_df['Polarity'] == "Good"]) * 100/len(results_df['Polarity']), 2)
        neg_percentage = round(len(results_df[results_df['Polarity'] == "Bad"]) * 100/len(results_df['Polarity']), 2)
        results_percents =  {
              "pos_percentage": pos_percentage,
              "neu_percentage": neu_percentage,
              "neg_percentage": neg_percentage
            }
        return results_df, results_percents, results_df_pos, results_df_neu, results_df_neg, my_prediction




    results_df, results_percents, results_df_pos, results_df_neu, results_df_neg, my_prediction = get_sentiment_analysis2(table = df, model = model, type=model_type)


    #my_preds = []
    #for i in df['Comment']:
    #    my_str = i
    #    print(i)
    #    sent = predict(net, i, seq_length)
    #    my_preds.append(sent)


    return render_template('sentiment_rnn_result.html', 
        df_x = column_name, 
        my_preds = results_df
        )





































@analyse.route('/train_results_rnn', methods = ['POST', 'GET'])
def train_results_rnn():
    if not current_user.is_authenticated:
        flash('You should be logged in to acced this page', 'info')
        next_page =  request.endpoint
        return redirect(url_for('users.login', next = next_page))
    if request.method != "POST":
        flash("You don't have permission accessing this page", 'info')
        return redirect(url_for('main.dashboard'))
    global tmp_folder, model_info
    form = SaveModel()
    title = "Resultats d'apprentissage"
    #df = pd.read_csv('YoutubeSpamMergedData.csv')
    #target = os.path.join(APP_ROOT, 'trainingcsv/')
    import csv
    import tensorflow as tf
    import numpy as np
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words('english'))
    if request.method == "POST":
        startdate = request.form['startdate']
        enddate = request.form['enddate']
        #start = datetime.datetime(start)
        #end = datetime.datetime(end)
        startdate = startdate + ' 00:00:00'
        enddate = enddate + ' 00:00:00'
        date_str = type(startdate)
        # search by author or book
        #cursor.execute("SELECT comment, polarity from training")
        target = os.path.join(APP_ROOT, 'data/')
        # reading the data
        print("FETCHING THE DATA")
        df = pd.read_excel( target + "/nft_sales.xlsx")
        my_data = df
        #conn.commit()
        type_data = type(my_data)
        df['asset_description'] = df.asset_description.astype('str')
        df['asset_name'] = df.asset_name.astype('str')
        df['collection_name'] = df.collection_name.astype('str')
        df['comment'] = df['collection_name'] + df['asset_name'] + df['asset_description'] 
        polarity_list = []
        # creatiing the types of nfts
        for i in range(0, len(df)):
            if df['event_total_price'][i] >= 0.5:
                polarity_list.append('1')
            elif df['event_total_price'][i] <= 0.1:
                polarity_list.append('0')
            else:
                polarity_list.append('-1')
        len(polarity_list)
        len(df['event_total_price'])
        df['polarity'] = polarity_list
        df['polarity'] = polarity_list
        df_data = df[["comment", "polarity"]]
        df_x = df_data['comment']
        df_y = df_data.polarity
        df_save = df
    

    #df = df2

    print(tf.__version__)

    model_info = {}
    vocab_size = 1000
    embedding_dim = model_info['embedding_dim'] = int(request.form['embedding_dim'])
    max_length = model_info['max_length'] = int(request.form['max_length'])
    if_shuffle = request.form['shuffle_set']
    trunc_type = model_info['trunc_type'] = 'post'
    padding_type = model_info['padding_type'] = 'post'
    oov_tok = model_info['oov_tok'] = '<OOV>'
    training_portion = model_info['training_portion'] = int(request.form['ratio_train'])
    num_epochs = model_info['n_epochs'] = int(request.form['epoch_n'])

    #articles = []
    #labels = []

    df_neg = df[df['polarity'] == "-1"]
    df_neu = df[df['polarity'] == "0"]
    df_pos = df[df['polarity'] == "1"]

    df_neg = df_neg.assign(polarity= "negative")
    df_neu = df_neu.assign(polarity= "neutre")
    df_pos = df_pos.assign(polarity= "positive")
    len_neg = len(df_neg)
    len_neu = len(df_neu)
    len_pos = len(df_pos)

    #min_n_cats = min(len_neg, len_neu, len_pos)
    min_n_cats = 500

    ratio_train = training_portion
    ratio_test = int(100 - int(training_portion))

    n_train =  int(min_n_cats * ratio_train // 100)
    n_validate = int(min_n_cats * ratio_test // 100)

    df_neg2 = df_neg[0:min_n_cats]
    df_neu2 = df_neu[0:min_n_cats]
    df_pos2 = df_pos[0:min_n_cats]
    frames = [df_neg2, df_neu2, df_pos2]
    result = pd.concat(frames, ignore_index=True)
    df = result
    if if_shuffle == "yes":
        df.sample(frac=1)
        df = df.sample(frac=1).reset_index(drop=True)
    my_df = df
    articles = df.comment
    labels = df.polarity
    print(len(labels))
    print(len(articles))


    df_neg = my_df[my_df.polarity == "negative"][:n_train]
    df_neu = my_df[my_df.polarity == "neutre"][:n_train]
    df_pos = my_df[my_df.polarity == "positive"][:n_train]
    frames = [df_neg, df_neu, df_pos]
    df_train = pd.concat(frames, ignore_index=True)
    # THIS VARIABLE HAS ALL THE TRAINING DATA SEPARATED 
    #result_clean = clean_text_for_training(df_content = result, column_name = 'text', stopword = 'french')
    #result_clean['text'] = result_clean['comment']
    #result = result_clean
    # This will give you the number of feature names

    df.head()

    x = df_train.comment
    y = df_train.polarity

    df_neg_1 = my_df[my_df.polarity == "negative"][n_train:n_train + n_validate]
    df_neu_1 = my_df[my_df.polarity == "neutre"][n_train:n_train + n_validate]
    df_pos_1 = my_df[my_df.polarity == "positive"][n_train:n_train + n_validate]
    frames_1 = [df_neg_1, df_neu_1, df_pos_1]
    result_1 = pd.concat(frames_1, ignore_index = True)
    x_test = result_1.comment
    y_test = result_1.polarity

    #add the save dataset
    df_save = df
    # end of add the save dataset
    

    train_articles = x
    train_labels = y

    validation_articles = x_test
    validation_labels = y_test


    print(train_articles.head())
    #print(train_size)
    print(len(train_articles))
    print(len(train_labels))
    print(len(validation_articles))
    print(len(validation_labels))
    model_info['train_len'] = len(train_articles)
    model_info['validation_len'] = len(validation_articles)

    tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(train_articles)
    word_index = tokenizer.word_index
    word_index_len = len(word_index)
    vocab_size = model_info['vocab_size'] = word_index_len

    train_sequences = tokenizer.texts_to_sequences(train_articles)

    train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    validation_sequences = tokenizer.texts_to_sequences(validation_articles)
    validation_padded = pad_sequences(validation_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    print(len(validation_sequences))
    print(validation_padded.shape)

    label_tokenizer = Tokenizer()
    label_tokenizer.fit_on_texts(labels)

    training_label_seq = np.array(label_tokenizer.texts_to_sequences(train_labels))
    validation_label_seq = np.array(label_tokenizer.texts_to_sequences(validation_labels))

    print(training_label_seq.shape)

    print(validation_label_seq.shape)

    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

    def decode_article(text):
        return ' '.join([reverse_word_index.get(i, '?') for i in text])

    model = tf.keras.Sequential([
        # Add an Embedding layer expecting input vocab of size 5000, and output embedding dimension of size 64 we set at the top
        tf.keras.layers.Embedding(vocab_size, embedding_dim),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
        tf.keras.layers.Dropout(0.01),
    #    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        # use ReLU in place of tanh function since they are very good alternatives of each other.
        tf.keras.layers.Dense(embedding_dim, activation='relu'),
        # Add a Dense layer with 6 units and softmax activation.
        # When we have multiple outputs, softmax convert outputs layers into a probability distribution.
        tf.keras.layers.Dense(4, activation='softmax')
    ])
    model.summary()

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    

    history = model.fit(train_padded, training_label_seq, epochs=num_epochs, validation_data=(validation_padded, validation_label_seq), verbose=2)

    train_labels = np.array(np.unique(training_label_seq, return_counts=True)).T

    validation_labels = np.array(np.unique(validation_label_seq, return_counts=True)).T

    y_pred = model.predict(validation_padded)

    from sklearn.metrics import classification_report, confusion_matrix
    matrix = confusion_matrix(validation_label_seq, y_pred.argmax(axis=1))

    confusion = pd.DataFrame(matrix, index=['bad', 'good', 'super'],
                             columns=['predicted_bad', 'predicted_good','predicted_super'])

    confusion

    class_report = classification_report(validation_label_seq, y_pred.argmax(axis=1), output_dict=True)
    class_report_df = pd.DataFrame(class_report).transpose()

    last_accuracy = model_info['acc'] = round(history.history["acc"][-1] * 100, 2)
    last_val_accuracy = model_info['val_acc'] = round(history.history["val_acc"][-1] * 100, 2)

    last_loss = round(history.history["loss"][-1] * 100, 2)
    last_val_loss = round(history.history["val_loss"][-1] * 100, 2)


    last_acc_per = int(round(last_accuracy * 0.1) * 10)
    last_val_acc_per = int(round(last_val_accuracy * 0.1) * 10)

    last_loss_per = int(round(last_loss * 0.1) * 10)
    last_val_loss_per = int(round(last_val_loss * 0.1) * 10)

    all_accuracy = history.history["acc"]
    all_val_accuracy = history.history["val_acc"]

    all_loss = history.history["loss"]
    all_val_loss = history.history["val_loss"]
    epoch_step = []
    for i in range(1, num_epochs + 1):
        epoch_step.append(i)

    model_summary = model.summary()
    from collections import defaultdict
    myMsgs = defaultdict(list)

    # creating custom messages of the model 
    # calculating the differenec between accucracy and value accuracy
    acc_diff = last_accuracy - last_val_accuracy
    neg_acc = class_report_df['f1-score'][0]
    neu_acc = class_report_df['f1-score'][1]
    pos_acc = class_report_df['f1-score'][2]

        # creating custom messages of the model 
    from collections import defaultdict
    myMsgs = defaultdict(list)

    if last_accuracy > 90:
        myMsgs['success'].append("The Model Is perfect, you can start using it")
    elif last_accuracy > 80:
        myMsgs['success'].append("The Model Is Good, you can start using it")
    elif last_accuracy < 80:
        myMsgs['danger'].append(" The model is poorly trained, you should modify some of the parameters and train again")

    # calculating the differenec between accucracy and value accuracy
    if acc_diff > 0.5:
        myMsgs['warning'].append("The model is overfitting, the model has failed in generalizing training Data, you must give it more input and train it again")
    elif acc_diff < 0.5:
        myMsgs['info'].append("Model is not overfitting, it is working fine on newly presented data")

    if neg_acc > 80:
        myMsgs['success'].append("Bad Prediction is Good ")
    elif neg_acc > 70:
        myMsgs['info'].append("Bad Prediction is almost Good, you can get better results, if you feed the model more data")
    elif neg_acc < 70:
        myMsgs['danger'].append("Bad Prediction is bad for this model")

    if neu_acc > 80:
        myMsgs['success'].append("Good Prediction is Good ")
    elif neu_acc > 70:
        myMsgs['info'].append("Good Prediction is almost Good, you can get better results, if you feed the model more data")
    elif neu_acc < 70:
        myMsgs['danger'].append("Good Prediction is bad for this model")

    if pos_acc > 80:
        myMsgs['success'].append("Super Prediction is Good ")
    elif pos_acc > 70:
        myMsgs['info'].append("Super Prediction is almost Good, you can get better results, if you feed the model more data")
    elif pos_acc < 70:
        myMsgs['danger'].append("Super Prediction is bad for this model")


    import string
    tmp_folder = string.ascii_lowercase
    target = os.path.join(APP_ROOT, 'tmp/' + tmp_folder + "/")
    if not os.path.isdir(target):
        os.mkdir(target)

    model_name = "my_keras_model.h5"
    model.save(target + model_name)

    import pickle
    with open(target + 'tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)

    model_info['previous_link'] = "saving"
    ma_type = type(last_accuracy)
    #save_path = "keras_rnn_2"
    #model_type = "keras"
    #seq_length = 200
    #n_layers = 3
    #user_id = 1
#
    #result = cursor.execute( """INSERT INTO models (name, type, seq_length, 
    #    n_layers, user_id) VALUES ( %s , %s , %s, %s , %s)""" , 
    #    (save_path, model_type, seq_length, n_layers ,user_id)  )
    #conn.commit()
    #print("model details are saved now ")

    return render_template('train_results_rnn.html', title = title,
        result = history, model = model, tokenizer = tokenizer, 
        articles = articles, labels = labels, 
        training_portion = training_portion,
        num_epochs = num_epochs, 
        train_labels = train_labels, 
        validation_labels = validation_labels, 
        confusion = confusion, 
        model_summary = model_summary, 
        last_accuracy = last_accuracy, 
        last_val_accuracy = last_val_accuracy,
        last_loss = last_loss,
        last_val_loss = last_val_loss,
        all_accuracy = all_accuracy, 
        all_val_accuracy = all_val_accuracy,
        last_acc_per = last_acc_per,
        last_val_acc_per = last_val_acc_per,
        last_loss_per = last_loss_per,
        last_val_loss_per = last_val_loss_per,
        all_loss = all_loss,
        all_val_loss = all_val_loss,
        epoch_step = epoch_step, 
        class_report_df = class_report_df, 
        myMsgs = myMsgs, 
        model_info = model_info, 
        tmp_folder = tmp_folder , 
        ma_type = ma_type
        )































































#endpoint for search
@analyse.route('/train_rnn', methods=['GET', 'POST'])
def train_rnn():
    if not current_user.is_authenticated:
        flash('You should be logged in to acced this page', 'info')
        next_page =  request.endpoint
        return redirect(url_for('users.login', next = next_page))
    folders = get_models()
    return render_template('train_rnn.html', title='Train From Database', 
        folders = folders)








# sentiment analysis rnn with keras
@analyse.route('/sentiment_rnn2', methods=['POST', 'GET'])
def sentiment_rnn2():
    cursor.execute("""SELECT name, type FROM `models`""")
    my_models = cursor.fetchall()
    models = []
    for i in my_models:
        models.append(i[0])

    return render_template('sentiment_rnn2.html', models = models
            )


@analyse.route('/sentiment_rnn_result2', methods=['POST', 'GET'])
def sentiment_rnn_result2():
    if not current_user.is_authenticated:
        flash('You should be logged in to acced this page', 'info')
        next_page =  request.endpoint
        return redirect(url_for('users.login', next = next_page))
    if request.method != "POST":
        flash("You don't have permission accessing this page", 'info')
        return redirect(url_for('main.dashboard'))
    # save excel if post request was made
    if request.method == "POST":
        secteur = request.form['secteur']
        model = request.form['model']
        stopwords = request.form['stopwords']
        all_comments, df_x, df, column_name = save_excel()

    cursor.execute("""SELECT name, type, seq_length, n_layers 
        FROM `models` WHERE name = %s""", ( model, ))
    my_data = cursor.fetchall()
    model_info = pd.DataFrame(list(my_data), columns=['name', 'type', 'seq_length', 'n_layers'])

    target = os.path.join(APP_ROOT, 'models/' + model + '/')
    # get model infos
    n_layers = model_info['n_layers'][0]
    seq_length=model_info['seq_length'][0]
    model_type = model_info['type'][0]
    
    


    



    results_df, results_percents, results_df_pos, results_df_neu, results_df_neg, my_prediction = get_sentiment_analysis2(table = df, model = model, type=model_type)


    #my_preds = []
    #for i in df['Comment']:
    #    my_str = i
    #    print(i)
    #    sent = predict(net, i, seq_length)
    #    my_preds.append(sent)


    return render_template('sentiment_rnn_result2.html', 
        df_x = column_name, 
        results_df = results_df
        )





@analyse.route('/save_model_rnn', methods = ['POST', 'GET'])
def save_model_rnn():
    if not current_user.is_authenticated:
        flash('You should be logged in to acced this page', 'info')
        next_page =  request.endpoint
        return redirect(url_for('users.login', next = next_page))
    
    previous_link = request.referrer 
    title = "Sauvegarde du modele"
    if previous_link == None:
        return redirect(url_for('main.dashboard'))
    if request.method != "POST":
        return redirect(url_for('main.dashboard'))
    form = SaveModel()
    import csv
    import tensorflow as tf
    import numpy as np
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words('english'))

    print(tf.__version__)

    save_path = request.form['modelName']

    source = os.path.join(APP_ROOT, 'tmp/' + tmp_folder + '/')

    target = os.path.join(APP_ROOT, 'models/' + save_path + '/')
    print(target)

    if not os.path.isdir(target):
        os.mkdir(target)

    import shutil
    import pickle
    from decimal import Decimal
    shutil.copy(source + "my_keras_model.h5", target + "my_keras_model.h5", follow_symlinks=True)
    shutil.copy(source + "tokenizer.pkl", target + "tokenizer.pkl", follow_symlinks=True)

    seq_length = model_info['max_length']
    acc = round(Decimal(model_info['acc']), 2)
    val_acc = round(Decimal(model_info['val_acc']), 2)
    n_epochs = model_info['n_epochs']
    model_type = 'keras'
    n_layers = 3
    user_id = current_user.id
    model_name = save_path
    result = cursor.execute( """INSERT INTO models (name, type, seq_length, 
        n_layers, user_id, acc, val_acc, n_epochs) 
        VALUES ( %s , %s , %s, %s , %s, %s, %s, %s)""" , 
        (save_path, model_type, seq_length, n_layers ,user_id, acc, val_acc, n_epochs)  )
    conn.commit()
    print("model details are saved now ")
    return render_template('model_saved.html', title = title, save_path = save_path, 
        model_name = model_name, model_info = model_info )







































































































































































