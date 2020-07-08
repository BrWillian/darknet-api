from flask_api import FlaskAPI


app = FlaskAPI(__name__)
app.config.from_object('config')

from app.controllers import default