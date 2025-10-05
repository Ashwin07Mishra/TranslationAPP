from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from os import path
import os
from .configuration import DevelopmentConfig, ProductionConfig, TestingConfig

config_map = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig
}

db = SQLAlchemy()
DB_NAME = "translations.db"

def create_app():
    # Setting up the app
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'your-secret-key-here'
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
    
    # Setting up the Database
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_NAME}'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    db.init_app(app)
    
    # Create upload directory
    app.config['UPLOAD_FOLDER'] = 'uploads'
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Register blueprints (keeping it simple - just one main blueprint)
    from .views import main
    app.register_blueprint(main, url_prefix='/')
    
    # Add context processor to make request available in templates
    @app.context_processor
    def inject_request():
        from flask import request
        
        def is_active_page(path):
            """Helper function to determine if a path is the current page"""
            return request.path == path
        
        return dict(request=request, is_active_page=is_active_page)
    
    # Database creation
    from .models import TranslationJob, Annotation
    create_database(app)
    
    return app

def create_database(app):
    if not path.exists('app/' + DB_NAME):
        with app.app_context():
            db.create_all()
        print('Created Translation Database!')
