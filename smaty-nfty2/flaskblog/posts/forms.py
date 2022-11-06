from flask_wtf import FlaskForm 
from flask_wtf.file import FileField, FileAllowed
from flask_login import current_user
from wtforms import StringField, PasswordField , SubmitField, BooleanField, TextAreaField
from wtforms.validators import DataRequired, Length, Email , EqualTo, ValidationError
from flaskblog.models import User 


class PostForm(FlaskForm):
	title = StringField('Title', validators=[DataRequired()])
	content = TextAreaField('Content', validators=[DataRequired()])
	submit = SubmitField('Post')




class WordsForm(FlaskForm):
	word = StringField('Word', validators=[DataRequired()])
	word_latin = StringField('Word Latin', validators=[DataRequired()])
	variations = TextAreaField('Variations', validators=[DataRequired()])
	submit = SubmitField('Post')




class CommentsForm(FlaskForm):
	comment_id = StringField('Word', validators=[DataRequired()])
	word_latin = StringField('Word Latin', validators=[DataRequired()])
	variations = TextAreaField('Variations', validators=[DataRequired()])
	submit = SubmitField('Update')





