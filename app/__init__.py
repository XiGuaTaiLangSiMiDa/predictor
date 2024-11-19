from flask import Flask
from flask_cors import CORS
import logging
from logging.handlers import RotatingFileHandler
import os
from app.config import Config

app = Flask(__name__, static_folder='static')
CORS(app)
app.config.from_object(Config)

# Ensure the logs directory exists
if not os.path.exists('logs'):
    os.mkdir('logs')

# Set up logging
file_handler = RotatingFileHandler('logs/app.log', maxBytes=10240, backupCount=10)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
file_handler.setLevel(logging.INFO)
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)
app.logger.info('Crypto Predictor startup')

# Ensure static directory exists
static_dir = os.path.join(app.root_path, 'static')
plots_dir = os.path.join(static_dir, 'plots')
os.makedirs(plots_dir, exist_ok=True)

from app import routes
