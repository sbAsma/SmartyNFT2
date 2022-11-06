from datetime import datetime
from itsdangerous import TimedJSONWebSignatureSerializer as Serializer 
from flaskblog import db , login_manager, app
from flask_login import UserMixin 

from flask_user import current_user, login_required, roles_required, UserManager, UserMixin


@login_manager.user_loader
def load_user(user_id):
	return User.query.get(int(user_id))

class User(db.Model, UserMixin):
	id = db.Column(db.Integer, primary_key = True)
	username = db.Column(db.String(20), unique=True, nullable=False )
	keywords = db.Column(db.String(250), unique=False, nullable=False )
	competitors = db.Column(db.String(250), unique=False, nullable=False )
	email = db.Column(db.String(120), unique=True, nullable= False )
	image_file = db.Column(db.String(20), nullable = False, default = 'default.jpg' )
	password = db.Column(db.String(60), nullable=False)
	active = db.Column(db.Boolean()),
	posts = db.relationship('Post', backref = 'author', lazy = True)

	# Relationships
	roles = db.relationship('Role', secondary='user_roles')

	def get_reset_token(self, expires_sec = 1800):
		s = Serializer(app.config['SECRET_KEY'], expires_sec)
		return s.dumps({"user_id": self.id}).decode('utf-8')

	@staticmethod
	def verify_reset_token(token):
		s = Serializer(app.config['SECRET_KEY'])
		try: 
			user_id = s.loads(token)['user_id']
		except: 
			return None
		return User.query.get(user_id)

	def __repr__(self):
		return f"User ( '{self.username}', '{self.email} ', {self.image_file} )"


# Define the Role data-model
class Role(db.Model):
    __tablename__ = 'roles'
    id = db.Column(db.Integer(), primary_key=True)
    name = db.Column(db.String(50), unique=True)

# Define the UserRoles association table
class UserRoles(db.Model):
    __tablename__ = 'user_roles'
    id = db.Column(db.Integer(), primary_key=True)
    user_id = db.Column(db.Integer(), db.ForeignKey('user.id', ondelete='CASCADE'))
    role_id = db.Column(db.Integer(), db.ForeignKey('roles.id', ondelete='CASCADE'))


user_manager = UserManager(app, db, User)
# Setup Flask-User
#user_manager = UserManager(app, db, User)

class Post(db.Model):
	id = db.Column(db.Integer, primary_key = True)
	title = db.Column(db.String(100), nullable = False)
	date_posted = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
	content = db.Column(db.Text, nullable = False)
	user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable = False)

	def __repr__(self):
		return f"Post ('{self.title}', 'self.date_posted') "




class Words(db.Model):
	ID = db.Column(db.Integer, primary_key = True)
	word = db.Column(db.String(100), unique = True, nullable = False)
	word_latin = db.Column(db.String(100) )
	variations = db.Column(db.Text, nullable = False)

	def __repr__(self):
		return f"Words ('{self.word}', '{self.word_latin}', '{self.variations}') "

