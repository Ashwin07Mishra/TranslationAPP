class Config:
    SECRET_KEY = 'fgefubfkfweofjodfvndklje347598ujfnflwefhfof'
    UPLOAD_FOLDER = 'uploads'

class DevelopmentConfig(Config):
    BASE_URL = 'http://localhost:5025'
    DEBUG = True

class ProductionConfig(Config):
    BASE_URL = 'https://cdis.iitk.ac.in/translation_tool'
    DEBUG = False

class TestingConfig(Config):
    BASE_URL = 'http://localhost:5025'
    TESTING = True