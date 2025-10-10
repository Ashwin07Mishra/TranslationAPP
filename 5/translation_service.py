import sys
import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor
import spacy
from tqdm import tqdm
import re
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import logging
import time
import gc
import fitz  # PyMuPDF
from collections import defaultdict
import pdfkit

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DetectorFactory.seed = 0

# Device configuration
DEVICE = "cpu"
logger.info(f"Using device: {DEVICE}")

# Global progress tracker
translation_progress = {}

# ============= CACHE MANAGEMENT =============
def clear_gpu_cache():
    """Clear GPU cache"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception as e:
        logger.warning(f"GPU cache clear failed: {e}")

def clear_cpu_cache():
    """Clear CPU cache"""
    try:
        gc.collect()
    except Exception as e:
        logger.warning(f"CPU cache clear failed: {e}")

def clear_all_cache():
    """Clear both GPU and CPU cache"""
    clear_gpu_cache()
    clear_cpu_cache()

# ============= LANGUAGE DETECTION =============
def detect_language(text):
    """Detect if text is Hindi or English"""
    if not text or not text.strip():
        return "eng_Latn"
    
    try:
        detected_lang = detect(text)
        if detected_lang == 'hi':
            return "hin_Deva"
        elif detected_lang == 'en':
            return "eng_Latn"
        else:
            # Check Devanagari script
            devanagari_chars = len(re.findall(r'[\u0900-\u097F]', text))
            total_chars = len(re.sub(r'\s', '', text))
            if total_chars > 0 and (devanagari_chars / total_chars) > 0.3:
                return "hin_Deva"
            return "eng_Latn"
    except Exception:
        # Fallback to character analysis
        devanagari_chars = len(re.findall(r'[\u0900-\u097F]', text))
        total_chars = len(re.sub(r'\s', '', text))
        if total_chars == 0:
            return "eng_Latn"
        return "hin_Deva" if (devanagari_chars / total_chars) > 0.3 else "eng_Latn"

# ============= TEXT CLEANING =============
def clean_hindi_text(text):
    """Clean Hindi text"""
    if not text:
        return ""
    try:
        text = re.sub(r'\s+', ' ', text)
        text = text.replace('।।', '।')
        text = re.sub(r'[।]{2,}', '।', text)
        return text.strip()
    except Exception:
        return text

def clean_english_output(text):
    """Clean English translation output"""
    if not text:
        return ""
    try:
        text = re.sub(r'[\u0900-\u097F]+', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text, flags=re.IGNORECASE)
        text = re.sub(r'([.!?])\1+', r'\1', text)
        text = re.sub(r'\s+([.!?,:;])', r'\1', text)
        return text.strip()
    except Exception:
        return text

# ============= INITIALIZE PROCESSORS =============
try:
    ip = IndicProcessor(inference=True)
    logger.info("IndicProcessor initialized")
except Exception as e:
    logger.error(f"IndicProcessor failed: {e}")
    sys.exit(1)

try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("SpaCy loaded")
except Exception as e:
    logger.error(f"SpaCy failed: {e}")
    sys.exit(1)

# ============= MODEL INITIALIZATION =============
def initialize_models():
    """Load translation models"""
    models = {}
    tokenizers = {}
    max_retries = 3
    
    try:
        # English to Hindi
        logger.info("Loading EN→HI model...")
        en_hi_model = "ai4bharat/indictrans2-en-indic-dist-200M"
        
        for attempt in range(max_retries):
            try:
                clear_all_cache()
                tokenizers['en_hi'] = AutoTokenizer.from_pretrained(en_hi_model, trust_remote_code=True)
                models['en_hi'] = AutoModelForSeq2SeqLM.from_pretrained(
                    en_hi_model,
                    trust_remote_code=True,
                    torch_dtype=torch.float32,
                ).to(DEVICE)
                logger.info("EN→HI model loaded")
                break
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(2)
        
        # Hindi to English
        logger.info("Loading HI→EN model...")
        hi_en_model = "ai4bharat/indictrans2-indic-en-dist-200M"
        
        for attempt in range(max_retries):
            try:
                clear_all_cache()
                tokenizers['hi_en'] = AutoTokenizer.from_pretrained(hi_en_model, trust_remote_code=True)
                models['hi_en'] = AutoModelForSeq2SeqLM.from_pretrained(
                    hi_en_model,
                    trust_remote_code=True,
                    torch_dtype=torch.float32,
                ).to(DEVICE)
                logger.info("HI→EN model loaded")
                break
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(2)
        
        return models, tokenizers
    
    except Exception as e:
        logger.error(f"Model initialization failed: {e}")
        clear_all_cache()
        raise

# ============= TRANSLATION FUNCTION =============
def translate_chunk(chunk, src_lang, tgt_lang, tokenizer, model):
    """Translate a text chunk"""
    if not chunk or not chunk.strip():
        return ""
    
    max_retries = 2
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                clear_cpu_cache()
            
            if src_lang == "hin_Deva":
                chunk = clean_hindi_text(chunk)
            
            batch = ip.preprocess_batch([chunk], src_lang=src_lang, tgt_lang=tgt_lang)
            inputs = tokenizer(batch, truncation=True, padding=True, return_tensors="pt").to(DEVICE)
            
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
                )
            
            translated = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
            translated = ip.postprocess_batch([translated], lang=tgt_lang)[0]
            
            if tgt_lang == "eng_Latn":
                translated = clean_english_output(translated)
            
            return translated
            
        except Exception as e:
            logger.error(f"Translation attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                return ""

# ============= DOCUMENT LAYOUT ANALYSIS =============
def extract_structured_document_with_layout(pdf_path):
    """Extract document preserving exact layout and positioning"""
    try:
        doc = fitz.open(pdf_path)
        structured_document = []
        
        for page_num, page in enumerate(doc):
            blocks = page.get_text("dict")["blocks"]
            page_width = page.rect.width
            page_height = page.rect.height
            
            # Sort blocks by their y-coordinate (top to bottom)
            text_blocks = []
            for block in blocks:
                if "lines" in block:
                    bbox = block['bbox']
                    x0, y0, x1, y1 = bbox
                    
                    # Extract all text from this block
                    block_text = ""
                    font_sizes = []
                    font_flags = []
                    
                    for line in block["lines"]:
                        line_text = ""
                        for span in line["spans"]:
                            line_text += span["text"]
                            font_sizes.append(span["size"])
                            font_flags.append(span["flags"])
                        block_text += line_text + "\n"
                    
                    if block_text.strip():
                        avg_font = sum(font_sizes) / len(font_sizes) if font_sizes else 12
                        is_bold = any(f & 2**4 for f in font_flags)
                        is_underline = any(f & 2**2 for f in font_flags)
                        
                        # Determine alignment based on position
                        center_x = (x0 + x1) / 2
                        page_center = page_width / 2
                        
                        if abs(center_x - page_center) < 50:
                            alignment = 'center'
                        elif x0 < page_width * 0.2:
                            alignment = 'left'
                        elif x0 > page_width * 0.3:
                            alignment = 'indented'
                        else:
                            alignment = 'left'
                        
                        text_blocks.append({
                            'text': block_text.strip(),
                            'bbox': bbox,
                            'font_size': avg_font,
                            'is_bold': is_bold,
                            'is_underline': is_underline,
                            'alignment': alignment,
                            'page': page_num + 1,
                            'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1
                        })
            
            # Sort by y-coordinate to maintain reading order
            text_blocks.sort(key=lambda x: x['y0'])
            
            # Calculate vertical spacing between consecutive blocks
            for i, block in enumerate(text_blocks):
                if i > 0:
                    prev_block = text_blocks[i-1]
                    spacing = block['y0'] - prev_block['y1']
                    block['spacing_before'] = max(0, spacing)
                else:
                    block['spacing_before'] = 0
                
                structured_document.append(block)
        
        doc.close()
        return structured_document
        
    except Exception as e:
        logger.error(f"Layout extraction failed: {e}")
        return []

# ============= TRANSLATION WITH STRUCTURE PRESERVATION =============
def translate_structured_elements(structured_doc, src_lang, tgt_lang, tokenizer, model):
    """Translate while preserving ALL structure metadata"""
    translated_elements = []
    
    for element in tqdm(structured_doc, desc="Translating document"):
        if not element['text'].strip():
            translated_elements.append(element)
            continue
        
        # Translate the text content only
        translated_text = translate_chunk(
            element['text'], src_lang, tgt_lang, tokenizer, model
        )
        
        # Create new element with translated text but IDENTICAL structure
        translated_element = {
            **element,  # Copy ALL original metadata
            'text': translated_text,
            'original_text': element['text']  # Keep original for reference
        }
        translated_elements.append(translated_element)
        
        # Periodic cache clearing
        if len(translated_elements) % 10 == 0:
            clear_cpu_cache()
    
    return translated_elements

# ============= LAYOUT-PRESERVING OUTPUT FORMATTING =============
def format_with_exact_layout_preservation(structured_elements, target_language):
    """Format document preserving EXACT original layout and structure"""
    output_lines = []
    
    for i, element in enumerate(structured_elements):
        text = element['text']
        spacing = element.get('spacing_before', 0)
        alignment = element.get('alignment', 'left')
        is_bold = element.get('is_bold', False)
        is_underline = element.get('is_underline', False)
        
        # Add vertical spacing exactly as in original
        if spacing > 30:
            # Large spacing - multiple line breaks
            spacing_lines = int(spacing / 15)  # Approximate line height
            output_lines.extend([''] * min(spacing_lines, 5))
        elif spacing > 15:
            # Medium spacing
            output_lines.extend([''] * 2)
        elif spacing > 5:
            # Small spacing
            output_lines.append('')
        
        # Format text based on original styling
        formatted_text = text
        
        # Apply text formatting if present in original
        if is_bold and is_underline:
            # In plain text, we can't do both, so just use bold indicator
            formatted_text = text  # Keep as is - no markup
        elif is_bold:
            formatted_text = text  # Keep as is - no markup
        elif is_underline:
            formatted_text = text  # Keep as is - no markup
        
        # Handle alignment by adding appropriate spacing
        if alignment == 'center':
            # For centered text, add appropriate padding (estimate)
            lines = formatted_text.split('\n')
            centered_lines = []
            for line in lines:
                if line.strip():
                    # Simple center approximation
                    padding = max(0, (60 - len(line)) // 2)
                    centered_lines.append(' ' * padding + line)
                else:
                    centered_lines.append(line)
            formatted_text = '\n'.join(centered_lines)
        elif alignment == 'indented':
            # Add indentation
            lines = formatted_text.split('\n')
            indented_lines = []
            for line in lines:
                if line.strip():
                    indented_lines.append('    ' + line)  # 4 spaces indent
                else:
                    indented_lines.append(line)
            formatted_text = '\n'.join(indented_lines)
        
        # Add the formatted text
        if formatted_text.strip():
            output_lines.append(formatted_text)
    
    return '\n'.join(output_lines)

# ============= PDF RECREATION WITH LAYOUT =============
def create_layout_preserved_pdf(structured_elements, output_path):
    """Create PDF with exact layout preservation"""
    try:
        # Create a new PDF document
        doc = fitz.open()  # new empty document
        page = doc.new_page(width=595, height=842)  # A4 size
        
        for element in structured_elements:
            text = element['text']
            if not text.strip():
                continue
            
            # Get original positioning
            bbox = element.get('bbox', [50, 50, 545, 100])  # default if missing
            font_size = element.get('font_size', 12)
            is_bold = element.get('is_bold', False)
            
            # Insert text at original position
            font_flags = 0
            if is_bold:
                font_flags |= 2**4  # Bold flag
            
            try:
                # Insert text block at original position
                page.insert_textbox(
                    fitz.Rect(bbox),
                    text,
                    fontsize=font_size,
                    fontname="helv" if not is_bold else "hebo",
                    color=(0, 0, 0),
                    align=fitz.TEXT_ALIGN_LEFT
                )
            except Exception as e:
                logger.warning(f"Failed to insert text block: {e}")
                # Fallback: insert as simple text
                try:
                    point = fitz.Point(bbox[0], bbox[1] + font_size)
                    page.insert_text(point, text, fontsize=font_size)
                except Exception as e2:
                    logger.error(f"Fallback text insertion failed: {e2}")
        
        # Save the PDF
        doc.save(output_path)
        doc.close()
        logger.info(f"✅ Layout-preserved PDF created: {output_path}")
        
    except Exception as e:
        logger.error(f"PDF creation failed: {e}")
        # Fallback to simple text file
        try:
            with open(output_path.replace('.pdf', '.txt'), 'w', encoding='utf-8') as f:
                for element in structured_elements:
                    if element['text'].strip():
                        f.write(element['text'] + '\n\n')
            logger.info(f"✅ Fallback text file created: {output_path.replace('.pdf', '.txt')}")
        except Exception as e2:
            logger.error(f"Fallback text creation failed: {e2}")

# ============= MAIN TRANSLATION FUNCTION =============
def translate_pdf_document(pdf_path, job_id=None):
    """
    Main translation function with EXACT structure preservation
    """
    models = None
    tokenizers = None
    
    try:
        clear_all_cache()
        
        # Load models
        logger.info("Loading translation models...")
        models, tokenizers = initialize_models()
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Extract structured document with exact layout
        logger.info("Extracting document with layout preservation...")
        structured_doc = extract_structured_document_with_layout(pdf_path)
        
        if not structured_doc:
            raise ValueError("Failed to extract document structure")
        
        # Detect language
        sample_text = " ".join([e['text'] for e in structured_doc[:5]])
        src_language = detect_language(sample_text)
        tgt_language = "hin_Deva" if src_language == "eng_Latn" else "eng_Latn"
        
        logger.info(f"Source: {src_language} → Target: {tgt_language}")
        
        # Select model
        model_key = 'en_hi' if src_language == "eng_Latn" else 'hi_en'
        model = models[model_key]
        tokenizer = tokenizers[model_key]
        
        if job_id:
            translation_progress[job_id] = {'progress': 20, 'step': 'Translating...'}
        
        # Translate with EXACT structure preservation
        logger.info("Translating document...")
        translated_elements = translate_structured_elements(
            structured_doc, src_language, tgt_language, tokenizer, model
        )
        
        if job_id:
            translation_progress[job_id] = {'progress': 80, 'step': 'Formatting output...'}
        
        # Format output with EXACT layout preservation
        logger.info("Formatting with layout preservation...")
        original_formatted = format_with_exact_layout_preservation(structured_doc, src_language)
        translated_formatted = format_with_exact_layout_preservation(translated_elements, tgt_language)
        
        # Create layout-preserved PDF
        output_pdf = "translated_output_layout_preserved.pdf"
        create_layout_preserved_pdf(translated_elements, output_pdf)
        
        clear_all_cache()
        
        return {
            'original_text': original_formatted,
            'translated_text': translated_formatted,
            'source_language': src_language,
            'target_language': tgt_language
        }
        
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        clear_all_cache()
        return {
            'original_text': "",
            'translated_text': f"Translation failed: {str(e)}",
            'source_language': "unknown",
            'target_language': "unknown"
        }
    
    finally:
        if models:
            for model in models.values():
                if hasattr(model, 'cpu'):
                    model.cpu()
                del model
        if tokenizers:
            del tokenizers
        clear_all_cache()