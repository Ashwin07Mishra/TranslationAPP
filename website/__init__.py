from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import os
from .configuration import DevelopmentConfig, ProductionConfig, TestingConfig

# Config map for different environments
config_map = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig
}

# SQLAlchemy instance
db = SQLAlchemy()
DB_NAME = "translations.db"

def create_app(config_name='development'):
    # Create the Flask app
    app = Flask(__name__, instance_relative_config=True)
    
    # Load config
    app.config.from_object(config_map.get(config_name, DevelopmentConfig))
    
    # Ensure instance folder exists (used for DB and other persistent files)
    os.makedirs(app.instance_path, exist_ok=True)
    
    # Absolute path for DB in instance folder
    db_path = os.path.join(app.instance_path, DB_NAME)
    app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{db_path}"
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    # Initialize SQLAlchemy
    db.init_app(app)
    
    # Absolute path for uploads folder (project_root/uploads)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    upload_folder = os.path.join(project_root, 'uploads')
    app.config['UPLOAD_FOLDER'] = upload_folder
    os.makedirs(upload_folder, exist_ok=True)
    
    # Register blueprint
    from .views import main
    app.register_blueprint(main, url_prefix='/')
    
    # Context processor for templates
    @app.context_processor
    def inject_request():
        from flask import request
        def is_active_page(path):
            return request.path == path
        return dict(request=request, is_active_page=is_active_page)
    
    # Create DB if it doesn't exist
    from .models import TranslationJob, Annotation
    create_database(db_path, app)
    
    return app


def create_database(db_path, app):
    if not os.path.exists(db_path):
        with app.app_context():
            db.create_all()
        print(f"✅ Created database at: {db_path}")
