from flask import Blueprint, render_template, request, jsonify, current_app, send_file
from werkzeug.utils import secure_filename
import uuid
import os
import time
import threading
from datetime import datetime
import traceback

from . import db
from .models import TranslationJob, Annotation
from .translation_service import translate_pdf_document, translation_progress

main = Blueprint('main', __name__)

@main.route('/')
def index():
    return render_template('index.html')

users = {
    'admin': 'admin',
    'm': 'm'
}

@main.route('/login', methods=['POST'])
def check_password():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({'success': False, 'message': 'Username and password are required'}), 400
    
    if username in users and users[username] == password:
        return jsonify({'success': True, 'message': 'Login successful', 'redirect': '/translation_tool'})
    else:
        return jsonify({'success': False, 'message': 'Invalid username or password'}), 401

@main.route('/translation_tool')
def translation_tool():
    return render_template('translation_tool.html')

@main.route('/upload', methods=['POST'])
def upload_pdf():
    file_path = None
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({'error': 'Only PDF files allowed'}), 400
        
        job_id = str(uuid.uuid4())
        secure_temp_filename = secure_filename(file.filename)
        upload_folder = current_app.config.get('UPLOAD_FOLDER')
        os.makedirs(upload_folder, exist_ok=True)
        file_path = os.path.join(upload_folder, f"{job_id}_{secure_temp_filename}")
        file.save(file_path)

        job = TranslationJob(
            id=job_id,
            original_filename=file.filename,
            status='processing',
            source_language='unknown',
            target_language='unknown'
        )
        db.session.add(job)
        db.session.commit()

        thread = threading.Thread(target=process_translation_background, args=(file_path, job_id, current_app._get_current_object(), upload_folder))
        thread.daemon = True
        thread.start()

        return jsonify({'job_id': job_id, 'status': 'processing', 'message': 'Translation started'})
    
    except Exception as e:
        db.session.rollback()
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
        current_app.logger.error(f"Upload error: {e}")
        traceback.print_exc()
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

def process_translation_background(file_path, job_id, app, upload_folder):
    with app.app_context():
        try:
            output_pdf_path = os.path.join(upload_folder, f"{job_id}_translated_output.pdf")
            result = translate_pdf_document(file_path, job_id=job_id, output_path=output_pdf_path)
            job = TranslationJob.query.get(job_id)
            if job:
                job.source_language = str(result.get('source_language', 'unknown'))
                job.target_language = str(result.get('target_language', 'unknown'))
                job.original_text = str(result.get('original_text', ''))[:50000]
                job.translated_text = str(result.get('translated_text', ''))[:50000]
                job.original_word_count = len(result.get('original_text', '').split())
                job.translated_word_count = len(result.get('translated_text', '').split())
                job.processing_time_seconds = float(result.get('processing_time_seconds', 0))
                job.status = 'completed' if os.path.exists(output_pdf_path) else 'error'
                if job.status == 'completed':
                    job.completed_at = datetime.utcnow()
                db.session.commit()
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            job = TranslationJob.query.get(job_id)
            if job:
                job.status = 'error'
                job.translated_text = f"Error: {str(e)}"
                db.session.commit()
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
            current_app.logger.error(f"Background processing error: {str(e)}")
            traceback.print_exc()

@main.route('/progress/<job_id>')
def check_progress(job_id):
    job = TranslationJob.query.get(job_id)
    progress_data = translation_progress.get(job_id, {'progress': 0, 'step': 'Starting...'})

    if job and job.status == 'completed':
        return jsonify({
            'progress': 100,
            'step': 'Translation completed',
            'completed': True,
            'result': {
                'translated_text': job.translated_text or '',
                'source_language': job.source_language or '',
                'target_language': job.target_language or '',
                'original_filename': job.original_filename or ''
            }
        })
    elif job and job.status == 'error':
        return jsonify({
            'progress': 0,
            'step': 'Translation failed',
            'completed': True,
            'error': True,
            'message': job.translated_text if job.translated_text else 'Translation failed'
        })
    return jsonify(progress_data)

@main.route('/download/<job_id>')
def download_translation(job_id):
    try:
        job = TranslationJob.query.get(job_id)
        if not job or job.status != 'completed':
            return jsonify({'error': 'Translation not ready or job not found.'}), 404
        upload_folder = current_app.config.get('UPLOAD_FOLDER')
        pdf_path = os.path.join(upload_folder, f"{job_id}_translated_output.pdf")
        if not os.path.exists(pdf_path):
            return jsonify({'error': 'PDF file not found.'}), 404
        download_name = f"{os.path.splitext(job.original_filename)[0]}_translated.pdf"
        return send_file(pdf_path, as_attachment=True, download_name=download_name, mimetype='application/pdf')
    except Exception as e:
        current_app.logger.error(f"Download error: {e}")
        traceback.print_exc()
        return jsonify({'error': f'Download failed: {str(e)}'}), 500

@main.route('/annotate', methods=['POST'])
def save_annotation():
    try:
        data = request.get_json()
        required_fields = ['job_id', 'start_position', 'end_position', 'selected_text']
        if not data or not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
        job_id = str(data['job_id'])
        job = TranslationJob.query.get(job_id)
        if not job:
            return jsonify({'error': 'Translation job not found'}), 404
        start_pos = int(data['start_position'])
        end_pos = int(data['end_position'])
        if start_pos < 0 or end_pos < 0 or start_pos >= end_pos:
            return jsonify({'error': 'Invalid text positions'}), 400
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
        return jsonify({'message': 'Annotation saved successfully', 'annotation_id': annotation.id, 'filename': job.original_filename})
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Annotation error: {e}")
        traceback.print_exc()
        return jsonify({'error': f'Database error: {str(e)}'}), 500

@main.route('/finalize/<job_id>', methods=['POST'])
def finalize_review(job_id):
    try:
        job = TranslationJob.query.get(job_id)
        if not job:
            return jsonify({'error': 'Translation job not found'}), 404
        annotations_count = Annotation.query.filter_by(job_id=job_id).count()
        return jsonify({'message': 'Review completed successfully', 'total_annotations': annotations_count, 'job_id': job_id, 'filename': job.original_filename})
    except Exception as e:
        current_app.logger.error(f"Finalize error: {e}")
        traceback.print_exc()
        return jsonify({'error': f'Failed to finalize review: {str(e)}'}), 500

@main.route('/documents')
def documents_list():
    try:
        jobs = TranslationJob.query.order_by(TranslationJob.created_at.desc()).all()
        documents = []
        for job in jobs:
            translated_text = job.translated_text or ''
            documents.append({
                'id': job.id,
                'translated_text': (translated_text[:500] + '...') if len(translated_text) > 500 else translated_text,
                'filename': job.original_filename or '',
                'source_language': job.source_language or '',
                'target_language': job.target_language or '',
                'status': job.status or '',
                'created_at': job.created_at.strftime('%Y-%m-%d %H:%M:%S') if job.created_at else '',
                'completed_at': job.completed_at.strftime('%Y-%m-%d %H:%M:%S') if job.completed_at else '',
                'annotations_count': len(job.annotations) if job.annotations else 0,
                'original_word_count': job.original_word_count or 0,
                'translated_word_count': job.translated_word_count or 0,
                'processing_time': f"{job.processing_time_seconds:.1f}s" if job.processing_time_seconds else "N/A"
            })
        return render_template('documents.html', documents=documents)
    except Exception as e:
        current_app.logger.error(f"Documents list error: {e}")
        traceback.print_exc()
        return render_template('documents.html', documents=[])

@main.route('/document/<job_id>')
def document_detail(job_id):
    try:
        job = TranslationJob.query.get_or_404(job_id)
        annotations = Annotation.query.filter_by(job_id=job_id).order_by(Annotation.created_at.desc()).all() or []
        return render_template('document_detail.html', job=job, annotations=annotations)
    except Exception as e:
        current_app.logger.error(f"Document detail error: {e}")
        traceback.print_exc()
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
        current_app.logger.error(f"Update annotation error: {e}")
        traceback.print_exc()
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
        current_app.logger.error(f"Delete annotation error: {e}")
        traceback.print_exc()
        return jsonify({'error': f'Deletion failed: {str(e)}'}), 500

@main.route('/delete/<job_id>', methods=['POST'])
def delete_translation(job_id):
    try:
        job = TranslationJob.query.get_or_404(job_id)
        Annotation.query.filter_by(job_id=job_id).delete()
        upload_folder = current_app.config.get('UPLOAD_FOLDER')
        pdf_path = os.path.join(upload_folder, f"{job_id}_translated_output.pdf")
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        db.session.delete(job)
        db.session.commit()
        return jsonify({'message': 'Translation job deleted successfully'})
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Delete translation error: {e}")
        traceback.print_exc()
        return jsonify({'error': f'Deletion failed: {str(e)}'}), 500
