import sys
import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor
from pypdf import PdfReader
import spacy
from tqdm import tqdm
import pdfplumber
import re
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import logging
import time
import gc
import fitz  # PyMuPDF for better structure detection
from collections import defaultdict
import pdfkit

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set seed for consistent results
DetectorFactory.seed = 0

def clear_gpu_cache():
    """Clear GPU cache to free memory"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        logger.info("GPU cache cleared")
    except Exception as e:
        logger.warning(f"Failed to clear GPU cache: {e}")

def clear_cpu_cache():
    """Clear CPU cache and run garbage collection"""
    try:
        gc.collect()
        logger.info("CPU cache cleared")
    except Exception as e:
        logger.warning(f"Failed to clear CPU cache: {e}")

def clear_all_cache():
    """Clear both GPU and CPU cache"""
    clear_gpu_cache()
    clear_cpu_cache()

def detect_language(text):
    """Find if text is Hindi or English using multiple methods"""
    if not text or not text.strip():
        logger.warning("Empty text provided for language detection")
        return "eng_Latn"
    
    try:
        detected_lang = detect(text)
        # Convert standard language codes to our format
        if detected_lang == 'hi':
            return "hin_Deva"
        elif detected_lang == 'en':
            return "eng_Latn"
        else:
            # Check script when language detection is uncertain
            devanagari_chars = len(re.findall(r'[\u0900-\u097F]', text))
            total_chars = len(re.sub(r'\s', '', text))
            if total_chars > 0 and (devanagari_chars / total_chars) > 0.3:
                return "hin_Deva"
            else:
                return "eng_Latn"
    except (LangDetectException, Exception) as e:
        logger.warning(f"Language detection failed: {e}. Using character analysis.")
        # Backup method: count Devanagari characters
        try:
            devanagari_chars = len(re.findall(r'[\u0900-\u097F]', text))
            total_chars = len(re.sub(r'\s', '', text))
            if total_chars == 0:
                return "eng_Latn"
            hindi_ratio = devanagari_chars / total_chars
            return "hin_Deva" if hindi_ratio > 0.3 else "eng_Latn"
        except Exception as e:
            logger.error(f"Character analysis also failed: {e}")
            return "eng_Latn"  # Default to English

# Force CPU usage for compatibility
try:
    DEVICE = "cpu"  # Force CPU usage
    logger.info(f"Using device: {DEVICE}")
except Exception as e:
    logger.error(f"Device detection failed: {e}")
    DEVICE = "cpu"

def initialize_models():
    """Load both translation models with error handling and retry logic"""
    models = {}
    tokenizers = {}
    max_retries = 3
    
    try:
        # Load English to Hindi model
        logger.info("Loading English to Hindi model...")
        en_hi_model_name = "ai4bharat/indictrans2-en-indic-dist-200M"  # 200M model for CPU
        
        for attempt in range(max_retries):
            try:
                clear_all_cache()  # Clear cache before loading
                tokenizers['en_hi'] = AutoTokenizer.from_pretrained(en_hi_model_name, trust_remote_code=True)
                models['en_hi'] = AutoModelForSeq2SeqLM.from_pretrained(
                    en_hi_model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float32,  # CPU-friendly
                    # Removed flash_attention_2 for CPU compatibility
                ).to(DEVICE)
                logger.info("English to Hindi model loaded successfully")
                break
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed to load English to Hindi model: {e}")
                if attempt == max_retries - 1:
                    logger.error("All attempts failed for English to Hindi model")
                    raise
                time.sleep(2)  # Wait before retry
        
        # Load Hindi to English model
        logger.info("Loading Hindi to English model...")
        hi_en_model_name = "ai4bharat/indictrans2-indic-en-dist-200M"  # 200M model for CPU
        
        for attempt in range(max_retries):
            try:
                clear_all_cache()  # Clear cache before loading
                tokenizers['hi_en'] = AutoTokenizer.from_pretrained(hi_en_model_name, trust_remote_code=True)
                models['hi_en'] = AutoModelForSeq2SeqLM.from_pretrained(
                    hi_en_model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float32,  # CPU-friendly
                    # Removed flash_attention_2 for CPU compatibility
                ).to(DEVICE)
                logger.info("Hindi to English model loaded successfully")
                break
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed to load Hindi to English model: {e}")
                if attempt == max_retries - 1:
                    logger.error("All attempts failed for Hindi to English model")
                    raise
                time.sleep(2)  # Wait before retry
        
        return models, tokenizers
    except Exception as e:
        logger.error(f"Model initialization failed: {e}")
        clear_all_cache()  # Clear cache on failure
        raise

# Initialize processor and spacy with error handling
try:
    ip = IndicProcessor(inference=True)
    logger.info("IndicProcessor initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize IndicProcessor: {e}")
    sys.exit(1)

try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("SpaCy English model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load SpaCy model: {e}. Please install with: python -m spacy download en_core_web_sm")
    sys.exit(1)

def analyze_font_hierarchy(pdf_path):
    """Analyze font sizes and styles to determine document hierarchy"""
    try:
        doc = fitz.open(pdf_path)
        font_info = defaultdict(int)
        
        for page in doc:
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            font_size = span["size"]
                            font_flags = span["flags"]
                            font_info[(font_size, font_flags)] += len(span["text"].strip())
        
        doc.close()
        
        # Sort by frequency to identify common text vs headers
        sorted_fonts = sorted(font_info.items(), key=lambda x: x[1], reverse=True)
        
        # Main text font (most common)
        main_font = sorted_fonts[0][0] if sorted_fonts else (12, 0)
        
        # Identify heading fonts (larger or bold)
        heading_fonts = []
        for (size, flags), count in sorted_fonts:
            if size > main_font[0] or flags & 2**4:  # Larger or bold
                heading_fonts.append((size, flags))
        
        return main_font, heading_fonts
    
    except Exception as e:
        logger.warning(f"Font analysis failed: {e}")
        return (12, 0), [(14, 16), (16, 16)]  # Default values

def create_translated_pdf(html_content: str, output_path: str = "transop.pdf"):
    """
    Converts HTML-formatted translated content into a PDF file.
    Saves as 'transop.pdf' by default.
    """
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        # You can customize options (margins, page size, etc.) as needed
        options = {
            "enable-local-file-access": None,
            "quiet": "",
            "page-size": "A4",
            "margin-top": "20mm",
            "margin-bottom": "20mm",
            "margin-left": "15mm",
            "margin-right": "15mm"
        }
        pdfkit.from_string(html_content, output_path, options=options)
        logger.info(f"Translated PDF saved to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to create PDF: {e}")
        raise

def extract_structured_text(pdf_path):
    """Extract text with structure information (headings, paragraphs, lists)"""
    try:
        doc = fitz.open(pdf_path)
        main_font, heading_fonts = analyze_font_hierarchy(pdf_path)
        
        structured_elements = []
        
        for page_num, page in enumerate(doc):
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if "lines" in block:  # Text block
                    block_text = ""
                    block_font_info = []
                    
                    for line in block["lines"]:
                        line_text = ""
                        line_fonts = []
                        
                        for span in line["spans"]:
                            text = span["text"]
                            font_size = span["size"]
                            font_flags = span["flags"]
                            
                            line_text += text
                            line_fonts.append((font_size, font_flags))
                        
                        if line_text.strip():
                            block_text += line_text + "\n"
                            block_font_info.extend(line_fonts)
                    
                    if block_text.strip():
                        # Determine element type based on font
                        avg_font_size = sum(f[0] for f in block_font_info) / len(block_font_info) if block_font_info else 12
                        is_bold = any(f[1] & 2**4 for f in block_font_info)
                        
                        element_type = "paragraph"
                        
                        # Check if it's a heading
                        for size, flags in heading_fonts:
                            if abs(avg_font_size - size) < 1 and (not is_bold or flags & 2**4):
                                if avg_font_size >= main_font[0] + 2:
                                    element_type = "heading"
                                break
                        
                        # Check for numbered lists or legal sections
                        text_clean = block_text.strip()
                        if re.match(r'^\d+\.?\s', text_clean):
                            element_type = "numbered_item"
                        elif re.match(r'^[•\-\*]\s', text_clean):
                            element_type = "bullet_item"
                        elif re.match(r'^[A-Z]\.\s', text_clean):
                            element_type = "lettered_item"
                        elif len(text_clean) < 100 and (is_bold or avg_font_size > main_font[0]):
                            element_type = "subheading"
                        
                        structured_elements.append({
                            'type': element_type,
                            'text': text_clean,
                            'page': page_num + 1,
                            'font_size': avg_font_size,
                            'is_bold': is_bold
                        })
        
        doc.close()
        return structured_elements
    
    except Exception as e:
        logger.error(f"Structured text extraction failed: {e}")
        # Fallback to basic text extraction
        return extract_text_fallback(pdf_path)

def extract_text_fallback(pdf_path):
    """Fallback text extraction method"""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            elements = []
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    paragraphs = text.split('\n\n')
                    for para in paragraphs:
                        if para.strip():
                            elements.append({
                                'type': 'paragraph',
                                'text': para.strip(),
                                'page': page_num + 1,
                                'font_size': 12,
                                'is_bold': False
                            })
            return elements
    except Exception as e:
        logger.error(f"Fallback extraction failed: {e}")
        return []

def reconstruct_document_structure(elements):
    """Reconstruct document with proper hierarchy and formatting"""
    structured_doc = []
    current_section = None
    
    for element in elements:
        if element['type'] == 'heading':
            if current_section:
                structured_doc.append(current_section)
            
            current_section = {
                'type': 'section',
                'heading': element['text'],
                'content': [],
                'page': element['page']
            }
        
        elif element['type'] in ['numbered_item', 'bullet_item', 'lettered_item']:
            if not current_section:
                current_section = {
                    'type': 'section',
                    'heading': '',
                    'content': [],
                    'page': element['page']
                }
            
            current_section['content'].append({
                'type': 'list_item',
                'text': element['text'],
                'list_type': element['type']
            })
        
        else:  # paragraph or subheading
            if not current_section:
                current_section = {
                    'type': 'section',
                    'heading': '',
                    'content': [],
                    'page': element['page']
                }
            
            current_section['content'].append({
                'type': element['type'],
                'text': element['text']
            })
    
    if current_section:
        structured_doc.append(current_section)
    
    return structured_doc

def translate_structured_document(structured_doc, src_language, tgt_language, tokenizer, model):
    """Translate structured document while preserving formatting"""
    translated_doc = []
    
    for section in tqdm(structured_doc, desc="Translating sections"):
        translated_section = {
            'type': section['type'],
            'heading': '',
            'content': [],
            'page': section['page']
        }
        
        # Translate heading if present
        if section['heading']:
            translated_heading = translate_chunk(section['heading'], src_language, tgt_language, tokenizer, model)
            translated_section['heading'] = translated_heading
        
        # Translate content items
        for item in section['content']:
            if item['text'].strip():
                translated_text = translate_chunk(item['text'], src_language, tgt_language, tokenizer, model)
                
                translated_item = {
                    'type': item['type'],
                    'text': translated_text
                }
                
                if 'list_type' in item:
                    translated_item['list_type'] = item['list_type']
                
                translated_section['content'].append(translated_item)
        
        translated_doc.append(translated_section)
        
        # Clear cache periodically
        if len(translated_doc) % 5 == 0:
            clear_cpu_cache()
    
    return translated_doc

def format_translated_document(translated_doc, target_language):
    """Format the translated document as HTML for PDF generation"""
    html_content = [
        '<html>',
        '<head>',
        '<meta charset="UTF-8">',
        '<style>',
        'body { font-family: Arial, "Noto Serif Devanagari", sans-serif; font-size: 12pt; margin: 20px; }',
        'h2 { font-size: 18pt; font-weight: bold; margin-top: 20px; }',
        'h3 { font-size: 16pt; font-weight: bold; margin-top: 15px; }',
        'h4 { font-size: 14pt; font-weight: bold; margin-top: 10px; }',
        'p { margin: 10px 0; line-height: 1.5; }',
        'li { margin: 5px 0; }',
        'ul, ol { margin-left: 20px; }',
        '</style>',
        '</head>',
        '<body>'
    ]
    
    for section in translated_doc:
        # Add section heading
        if section['heading']:
            html_content.append(f'<h2>{section["heading"]}</h2>')
        
        # Add content items
        for item in section['content']:
            if item['type'] == 'heading':
                html_content.append(f'<h3>{item["text"]}</h3>')
            
            elif item['type'] == 'subheading':
                html_content.append(f'<h4>{item["text"]}</h4>')
            
            elif item['type'] == 'list_item':
                if item.get('list_type') == 'numbered_item':
                    number_match = re.match(r'^(\d+\.?)\s*(.*)', item['text'])
                    if number_match:
                        number, text = number_match.groups()
                        html_content.append(f'<ol><li value="{number.rstrip(".")}">{text}</li></ol>')
                    else:
                        html_content.append(f'<ol><li>{item["text"]}</li></ol>')
                
                elif item.get('list_type') == 'bullet_item':
                    html_content.append(f'<ul><li>{item["text"]}</li></ul>')
                
                elif item.get('list_type') == 'lettered_item':
                    letter_match = re.match(r'^([A-Z]\.?)\s*(.*)', item['text'])
                    if letter_match:
                        letter, text = letter_match.groups()
                        html_content.append(f'<ol style="list-style-type: upper-alpha;"><li value="{letter.rstrip(".")}">{text}</li></ol>')
                    else:
                        html_content.append(f'<ol style="list-style-type: upper-alpha;"><li>{item["text"]}</li></ol>')
            
            else:  # regular paragraph
                html_content.append(f'<p>{item["text"]}</p>')
        
        html_content.append('<br>')
    
    html_content.append('</body></html>')
    return "".join(html_content)

def clean_hindi_text(text):
    """Remove extra spaces and fix Hindi punctuation"""
    if not text:
        return ""
    
    try:
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Fix double punctuation marks
        text = text.replace('।।', '।')  # Remove double danda
        text = text.replace('॥', '।')  # Normalize double danda to single
        # Clean excessive punctuation
        text = re.sub(r'[।]{2,}', '।', text)
        text = re.sub(r'[.]{2,}', '.', text)
        return text.strip()
    except Exception as e:
        logger.warning(f"Hindi text cleaning failed: {e}")
        return text

def clean_english_output(text):
    """Remove translation artifacts from English text"""
    if not text:
        return ""
    
    try:
        # Remove any leftover Devanagari characters
        text = re.sub(r'[\u0900-\u097F]+', '', text)
        # Fix spacing issues
        text = re.sub(r'\s+', ' ', text)
        # Remove word repetitions like "the the"
        text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text, flags=re.IGNORECASE)
        # Fix punctuation repetition
        text = re.sub(r'([.!?])\1+', r'\1', text)
        # Clean spacing around punctuation
        text = re.sub(r'\s+([.!?,:;])', r'\1', text)
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
        # Remove standalone Hindi punctuation
        text = re.sub(r'\s+[।]+\s+', ' ', text)
        return text.strip()
    except Exception as e:
        logger.warning(f"English text cleaning failed: {e}")
        return text

def translate_chunk(chunk, src_lang, tgt_lang, tokenizer, model):
    """Translate a text chunk using the AI model with improved error recovery"""
    if not chunk or not chunk.strip():
        return ""
    
    max_retries = 2
    for attempt in range(max_retries):
        try:
            # Clear cache before translation if this is a retry
            if attempt > 0:
                clear_cpu_cache()
            
            # Clean input if Hindi
            if src_lang == "hin_Deva":
                chunk = clean_hindi_text(chunk)
            
            # Preprocess for translation
            try:
                batch = ip.preprocess_batch([chunk], src_lang=src_lang, tgt_lang=tgt_lang)
            except Exception as e:
                logger.error(f"Preprocessing failed (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    return ""
                continue
            
            try:
                inputs = tokenizer(batch, truncation=True, padding=True, return_tensors="pt").to(DEVICE)
            except Exception as e:
                logger.error(f"Tokenization failed (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    return ""
                continue
            
            try:
                with torch.no_grad():
                    generated_tokens = model.generate(
                        inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_length=200,
                        min_length=5,
                        num_beams=4,
                        early_stopping=True,
                        no_repeat_ngram_size=3,
                        repetition_penalty=1.2,
                        length_penalty=1.0,
                        do_sample=False,
                        temperature=1.0,
                    )
            except Exception as e:
                logger.error(f"Model generation failed (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    return ""
                continue
            
            try:
                translated = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
                translated = ip.postprocess_batch([translated], lang=tgt_lang)[0]
                
                if tgt_lang == "eng_Latn":
                    translated = clean_english_output(translated)
                
                return translated
            except Exception as e:
                logger.error(f"Post-processing failed (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    return ""
                continue
                
        except Exception as e:
            logger.error(f"Translation attempt {attempt + 1} failed for chunk: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)  # Wait before retry
            else:
                return ""

def get_token_count(paragraph, tokenizer, ip, src_lang, tgt_lang=None):
    """Count tokens in text for chunk size limits"""
    if not paragraph:
        return 0
    
    try:
        if not isinstance(paragraph, str):
            paragraph = str(paragraph)
        
        # Auto-determine target language if not provided
        if tgt_lang is None:
            tgt_lang = "hin_Deva" if src_lang == "eng_Latn" else "eng_Latn"
        
        # Preprocess for translation
        batch = ip.preprocess_batch([paragraph], src_lang=src_lang, tgt_lang=tgt_lang)
        inputs = tokenizer(batch, return_tensors="pt", truncation=False)
        return len(inputs["input_ids"][0])
    
    except Exception as e:
        logger.warning(f"Token counting failed: {e}")
        # Fallback: rough estimate based on words
        return int(len(paragraph.split()) * 1.3)

translation_progress = {}

def translate_pdf_document(pdf_path, job_id=None):
    """
    Translate PDF document with structure preservation
    Saves translated PDF as 'transop.pdf' in the workspace directory
    Returns: dict with original_text, translated_text, source_language, target_language
    """
    models = None
    tokenizers = None
    
    try:
        # Clear cache at the start
        clear_all_cache()
        
        # Initialize models
        logger.info("Loading translation models...")
        models, tokenizers = initialize_models()
        
        # Check if file exists
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        logger.info("Extracting structured text from PDF...")
        structured_elements = extract_structured_text(pdf_path)
        
        if not structured_elements:
            raise ValueError("No text could be extracted from the PDF")
        
        # Combine all text for language detection
        all_text = " ".join([elem['text'] for elem in structured_elements[:10]])  # Sample first 10 elements
        
        # Detect document language
        logger.info("Detecting document language...")
        src_language = detect_language(all_text)
        tgt_language = "hin_Deva" if src_language == "eng_Latn" else "eng_Latn"
        
        logger.info(f"Detected source language: {src_language}")
        logger.info(f"Target language: {tgt_language}")
        
        # Choose appropriate model
        if src_language == "eng_Latn":
            model_key = 'en_hi'
        else:
            model_key = 'hi_en'
        
        model = models[model_key]
        tokenizer = tokenizers[model_key]
        
        # Reconstruct document structure
        logger.info("Reconstructing document structure...")
        structured_doc = reconstruct_document_structure(structured_elements)
        logger.info(f"Found {len(structured_doc)} sections")
        
        if job_id:
            translation_progress[job_id] = {'progress': 10, 'step': 'Starting structured translation...'}
        
        # Translate structured document
        logger.info("Translating structured document...")
        translated_doc = translate_structured_document(structured_doc, src_language, tgt_language, tokenizer, model)
        
        if job_id:
            translation_progress[job_id] = {'progress': 90, 'step': 'Formatting final document...'}
        
        # Format final document
        logger.info("Formatting translated document...")
        final_translation = format_translated_document(translated_doc, tgt_language)
        original_text = format_translated_document(structured_doc, src_language)
        
        # Generate PDF
        output_pdf = "transop.pdf"
        logger.info(f"Generating PDF: {output_pdf}")
        create_translated_pdf(final_translation, output_pdf)
        
        # Final cache clear
        clear_all_cache()
        
        # Return the required format for Flask app
        return {
            'original_text': original_text,
            'translated_text': final_translation,
            'source_language': src_language,
            'target_language': tgt_language
        }
    
    except Exception as e:
        logger.error(f"Translation process failed: {e}")
        # Clear cache on error
        clear_all_cache()
        
        # Return error result instead of raising exception
        return {
            'original_text': "",
            'translated_text': f"Translation failed: {str(e)}",
            'source_language': "unknown",
            'target_language': "unknown"
        }
    
    finally:
        # Ensure models are cleaned up
        try:
            if models:
                for model in models.values():
                    if hasattr(model, 'cpu'):
                        model.cpu()
                    del model
            if tokenizers:
                del tokenizers
            clear_all_cache()
        except Exception as e:
            logger.warning(f"Model cleanup failed: {e}")