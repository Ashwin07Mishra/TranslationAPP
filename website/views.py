from flask import Blueprint, render_template, request, jsonify, current_app, send_file
from werkzeug.utils import secure_filename
import uuid
import os
import time
import requests
import threading
from datetime import datetime
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import io
from . import db
from .models import TranslationJob, Annotation
from .translation_service import translate_pdf_document


main = Blueprint('main', __name__)

@main.route('/')
def index():
    return render_template('index.html')

users = {
    'admin': 'admin',  # In production, use environment variables or a secure vault,
    'm': 'm'
}


@main.route('/login', methods=['POST'])
def check_password():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({
            'success': False,
            'message': 'Username and password are required'
        }), 400
        
    if username in users and users[username] == password:
        return jsonify({
            'success': True,
            'message': 'Login successful',
            'redirect': '/translation_tool.html'  # Frontend can use this to redirect
        })
    else:
        return jsonify({
            'success': False,
            'message': 'Invalid username or password'
        }), 401

@main.route('/translation_tool')
def translation_tool():
    """Render the translation tool page"""
    return render_template('translation_tool.html')


@main.route('/upload', methods=['POST'])
def upload_pdf():
    file_path = None
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if not file.filename or not file.filename.lower().endswith('.pdf'):
            return jsonify({'error': 'Only PDF files allowed'}), 400
        
        # Store original filename and create secure version for temporary storage
        original_filename = file.filename
        job_id = str(uuid.uuid4())
        secure_temp_filename = secure_filename(file.filename)
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], f"{job_id}_{secure_temp_filename}")
        file.save(file_path)
        
        # Create job record with ORIGINAL filename
        job = TranslationJob(
            id=job_id, 
            original_filename=original_filename,
            status='processing',
            source_language='unknown',
            target_language='unknown'
        )
        db.session.add(job)
        db.session.commit()
        
        # START BACKGROUND PROCESSING - Pass the app instance!
        app = current_app._get_current_object()  # Get the actual app instance
        thread = threading.Thread(target=process_translation_background, args=(file_path, job_id, app))
        thread.daemon = True
        thread.start()
        
        # Return immediately for progress tracking
        return jsonify({
            'job_id': job_id,
            'status': 'processing',
            'message': 'Translation started'
        })
        
    except Exception as e:
        db.session.rollback()
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass
        
        print(f"Upload error: {str(e)}")
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

def process_translation_background(file_path, job_id, app):
    """Background function that actually does the translation"""
    # Use the passed app instance instead of current_app
    with app.app_context():
        try:
            # This runs in background thread with progress tracking
            start_time = time.time()
            result = translate_pdf_document(file_path, job_id=job_id)  # Pass job_id for progress
            processing_time = time.time() - start_time
            
            # Update database with results
            job = TranslationJob.query.get(job_id)
            if job:
                job.source_language = str(result['source_language'])
                job.target_language = str(result['target_language'])
                job.original_text = str(result['original_text'])
                job.translated_text = str(result['translated_text'])
                job.original_word_count = len(result['original_text'].split())
                job.translated_word_count = len(result['translated_text'].split())
                job.processing_time_seconds = float(processing_time)
                job.status = 'completed'
                job.completed_at = datetime.utcnow()
                db.session.commit()
            
            # Clean up temporary file
            if os.path.exists(file_path):
                os.remove(file_path)
                
        except Exception as e:
            # Mark job as failed
            job = TranslationJob.query.get(job_id)
            if job:
                job.status = 'error'
                db.session.commit()
            
            if os.path.exists(file_path):
                os.remove(file_path)
            
            print(f"Background processing error: {str(e)}")

@main.route('/progress/<job_id>')
def check_progress(job_id):
    """Get translation progress"""
    from .translation_service import translation_progress
    
    # Get live progress from translation service
    progress_data = translation_progress.get(job_id, {'progress': 0, 'step': 'Starting...'})
    
    # Check if job completed in database
    job = TranslationJob.query.get(job_id)
    if job and job.status == 'completed':
        return jsonify({
            'progress': 100,
            'step': 'Translation completed',
            'completed': True,
            'result': {
                'translated_text': job.translated_text,
                'source_language': job.source_language,
                'target_language': job.target_language,
                'original_filename': job.original_filename
            }
        })
    elif job and job.status == 'error':
        return jsonify({
            'progress': 0,
            'step': 'Translation failed',
            'completed': True,
            'error': True
        })
    
    return jsonify(progress_data)

@main.route('/annotate', methods=['POST'])
def save_annotation():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate required fields
        required_fields = ['job_id', 'start_position', 'end_position', 'selected_text']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Check if job exists
        job_id = str(data['job_id'])
        existing_job = TranslationJob.query.get(job_id)
        if not existing_job:
            return jsonify({'error': 'Translation job not found'}), 404
        
        # Validate positions
        try:
            start_pos = int(data['start_position'])
            end_pos = int(data['end_position'])
            if start_pos < 0 or end_pos < 0 or start_pos >= end_pos:
                return jsonify({'error': 'Invalid text positions'}), 400
        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid position data types'}), 400
        
        # Create annotation
        annotation = Annotation(
            job_id=job_id,
            start_position=start_pos,
            end_position=end_pos,
            selected_text=str(data['selected_text']),
            error_type=str(data.get('error_type', 'general')),
            comment=str(data.get('comment', '')),
            suggested_correction=str(data.get('suggested_correction', ''))
        )
        
        db.session.add(annotation)
        db.session.commit()
        
        return jsonify({
            'message': 'Annotation saved successfully',
            'annotation_id': annotation.id,
            'filename': existing_job.original_filename
        })
        
    except Exception as e:
        db.session.rollback()
        print(f"Annotation error: {str(e)}")
        return jsonify({'error': f'Database error: {str(e)}'}), 500

@main.route('/finalize/<job_id>', methods=['POST'])
def finalize_review(job_id):
    try:
        job = TranslationJob.query.get(job_id)
        if not job:
            return jsonify({'error': 'Translation job not found'}), 404
        
        annotations_count = Annotation.query.filter_by(job_id=job_id).count()
        
        return jsonify({
            'message': 'Review completed successfully',
            'total_annotations': annotations_count,
            'job_id': job_id,
            'filename': job.original_filename
        })
        
    except Exception as e:
        print(f"Finalize error: {str(e)}")
        return jsonify({'error': f'Failed to finalize review: {str(e)}'}), 500

@main.route('/documents')
def documents_list():
    """Show all translation jobs and their annotations"""
    try:
        # Query all translation jobs, newest first
        jobs = TranslationJob.query.order_by(TranslationJob.created_at.desc()).all()
        
        # Prepare document data for display
        documents = []
        for job in jobs:
            documents.append({
                'id': job.id,
                'translated_text': job.translated_text,
                'filename': job.original_filename,
                'source_language': job.source_language,
                'target_language': job.target_language,
                'status': job.status,
                'created_at': job.created_at.strftime('%Y-%m-%d %H:%M:%S') if job.created_at else '',
                'completed_at': job.completed_at.strftime('%Y-%m-%d %H:%M:%S') if job.completed_at else '',
                'annotations_count': len(job.annotations),
                'original_word_count': job.original_word_count or 0,
                'translated_word_count': job.translated_word_count or 0,
                'processing_time': f"{job.processing_time_seconds:.1f}s" if job.processing_time_seconds else "N/A"
            })
        
        return render_template('documents.html', documents=documents)
        
    except Exception as e:
        print(f"Documents list error: {str(e)}")
        return render_template('documents.html', documents=[])

@main.route('/document/<job_id>')
def document_detail(job_id):
    """Show detailed view of a specific document and its annotations"""
    try:
        job = TranslationJob.query.get_or_404(job_id)
        annotations = Annotation.query.filter_by(job_id=job_id).order_by(Annotation.created_at.desc()).all()
        
        # Pass the model objects directly - no conversion needed
        return render_template('document_detail.html', job=job, annotations=annotations)
        
    except Exception as e:
        print(f"Document detail error: {str(e)}")
        return "Document not found", 404

@main.route('/annotation/<int:annotation_id>', methods=['PUT'])
def update_annotation(annotation_id):
    try:
        data = request.get_json()
        annotation = Annotation.query.get_or_404(annotation_id)

        annotation.error_type = data.get('error_type', annotation.error_type)
        annotation.comment = data.get('comment', annotation.comment)
        annotation.suggested_correction = data.get('suggested_correction', annotation.suggested_correction)
        db.session.commit()

        return jsonify({'message': 'Annotation updated successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Update failed: {str(e)}'}), 500

@main.route('/annotation/<int:annotation_id>', methods=['DELETE'])
def delete_annotation(annotation_id):
    try:
        annotation = Annotation.query.get_or_404(annotation_id)
        db.session.delete(annotation)
        db.session.commit()
        return jsonify({'message': 'Annotation deleted successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Deletion failed: {str(e)}'}), 500
    
@main.route('/delete/<job_id>', methods=['POST'])
def delete_translation(job_id):
    try:
        # Get the translation job
        job = TranslationJob.query.get_or_404(job_id)
        
        # Delete all associated annotations first (due to foreign key constraints)
        Annotation.query.filter_by(job_id=job_id).delete()
        
        # Delete the translation job
        db.session.delete(job)
        db.session.commit()
        
        return jsonify({
            'message': 'Translation deleted successfully',
            'filename': job.original_filename
        })
        
    except Exception as e:
        db.session.rollback()
        print(f"Delete error: {str(e)}")
        return jsonify({'error': f'Failed to delete translation: {str(e)}'}), 500

@main.route('/rename/<job_id>', methods=['POST'])
def rename_file(job_id):
    try:
        # Get the translation job
        job = TranslationJob.query.get_or_404(job_id)
        
        # Get new filename from request
        data = request.get_json()
        if not data or 'new_filename' not in data:
            return jsonify({
                'success': False,
                'message': 'New filename is required'
            }), 400
            
        new_filename = data['new_filename'].strip()
        
        # Validate new filename
        if not new_filename:
            return jsonify({
                'success': False,
                'message': 'Filename cannot be empty'
            }), 400
            
        # Update the filename in the database
        job.original_filename = new_filename
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'File renamed successfully',
            'new_filename': new_filename
        })
        
    except Exception as e:
        db.session.rollback()
        print(f"Rename error: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Failed to rename file: {str(e)}'
        }), 500

@main.route('/download/<job_id>')
def download_translation(job_id):
    # Get the translation job
    job = TranslationJob.query.get(job_id)
    
    if not job or not job.translated_text:
        return jsonify({'error': 'Translation not available'}), 404

    # Create PDF in memory
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Register a Unicode Hindi font (make sure the TTF file is available)
    font_path = "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf"
    pdfmetrics.registerFont(TTFont("NotoSansDevanagari", font_path))

    # Set font
    font_size = 12
    p.setFont("NotoSansDevanagari", font_size)
    
    # Define margins with more bottom padding
    left_margin = 50
    right_margin = width - 50
    top_margin = height - 50
    bottom_margin = 70  # Increased bottom margin for safety
    
    # Available width for text
    available_width = right_margin - left_margin
    
    # Start position
    y_position = top_margin
    
    # Process text paragraph by paragraph
    paragraphs = job.translated_text.split('\n')
    for paragraph in paragraphs:
        if not paragraph.strip():
            # Add space for empty lines
            y_position -= font_size
            # Check if we need a new page after this empty line
            if y_position < bottom_margin:
                p.showPage()
                p.setFont("NotoSansDevanagari", font_size)
                y_position = top_margin
            continue
            
        # Split the paragraph into words for better handling of diacritics
        words = paragraph.split()
        
        # Initialize a line
        current_line = ""
        current_width = 0
        
        for word in words:
            # Get width of this word with a space
            word_width = p.stringWidth(word + " ", "NotoSansDevanagari", font_size)
            
            # If this word would exceed the line width
            if current_width + word_width > available_width:
                # Check if we're at the bottom of the page
                if y_position < bottom_margin:
                    p.showPage()
                    p.setFont("NotoSansDevanagari", font_size)
                    y_position = top_margin
                
                # Draw the current line
                p.drawString(left_margin, y_position, current_line)
                y_position -= font_size * 1.2  # Move down for next line
                
                # Start a new line with this word
                current_line = word + " "
                current_width = word_width
            else:
                # Add word to current line
                current_line += word + " "
                current_width += word_width
        
        # Draw any remaining text in the current line
        if current_line:
            # Check if we're at the bottom of the page
            if y_position < bottom_margin:
                p.showPage()
                p.setFont("NotoSansDevanagari", font_size)
                y_position = top_margin
            
            p.drawString(left_margin, y_position, current_line)
            y_position -= font_size * 1.5  # Extra space after paragraph
        
    # Finalize the document
    p.showPage()
    p.save()

    # Return PDF
    buffer.seek(0)
    return send_file(
        buffer,
        as_attachment=True,
        download_name=f"{job.original_filename.replace('.pdf', '')}_translation.pdf",
        mimetype='application/pdf'
    )