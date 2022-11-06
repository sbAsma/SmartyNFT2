from flask_wtf import FlaskForm 
from flask_wtf.file import FileField, FileAllowed
from flask_login import current_user
from wtforms import StringField, PasswordField , SubmitField, BooleanField, TextAreaField
from wtforms.validators import DataRequired, Length, Email , EqualTo, ValidationError
from flaskblog.models import User 


class SaveModel(FlaskForm):
	modelName = StringField('modelName', 
						validators = [DataRequired(), Length(min=2, max=20)] )
	submit = SubmitField('Save Model')


