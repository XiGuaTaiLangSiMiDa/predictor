import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-12345'
    UPLOAD_FOLDER = 'static/plots'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # Ensure the plots directory exists
    @staticmethod
    def init_app(app):
        os.makedirs(os.path.join(app.root_path, Config.UPLOAD_FOLDER), exist_ok=True)
