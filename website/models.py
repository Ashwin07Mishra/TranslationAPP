from . import db
from datetime import datetime

class TranslationJob(db.Model):
    __tablename__ = 'translation_jobs'
    
    # Primary identifiers
    id = db.Column(db.String(36), primary_key=True)
    original_filename = db.Column(db.String(255), nullable=False)  # Store actual filename
    
    # Language information
    source_language = db.Column(db.String(50), nullable=False, default='unknown')
    target_language = db.Column(db.String(50), nullable=False, default='unknown')
    
    # Text content
    original_text = db.Column(db.Text)
    translated_text = db.Column(db.Text)
    
    # Analytics fields (keeping them since you want analytics)
    original_word_count = db.Column(db.Integer)
    translated_word_count = db.Column(db.Integer)
    processing_time_seconds = db.Column(db.Float)
    
    # Status and timestamps
    status = db.Column(db.String(20), default='processing')
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    completed_at = db.Column(db.DateTime)
    
    # Relationships
    annotations = db.relationship('Annotation', backref='translation_job', lazy=True, cascade='all, delete-orphan')

class Annotation(db.Model):
    __tablename__ = 'annotations'
    
    id = db.Column(db.Integer, primary_key=True)
    job_id = db.Column(db.String(36), db.ForeignKey('translation_jobs.id'), nullable=False)
    
    # Position and text data
    start_position = db.Column(db.Integer, nullable=False)
    end_position = db.Column(db.Integer, nullable=False)
    selected_text = db.Column(db.Text, nullable=False)
    
    # Error categorization
    error_type = db.Column(db.String(50), nullable=False, default='general')
    comment = db.Column(db.Text)
    suggested_correction = db.Column(db.Text)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    
    # Analytics properties (calculated, not stored)
    @property
    def selected_text_length(self):
        return len(self.selected_text) if self.selected_text else 0
    
    @property
    def comment_length(self):
        return len(self.comment) if self.comment else 0
    
    @property
    def correction_length(self):
        return len(self.suggested_correction) if self.suggested_correction else 0
