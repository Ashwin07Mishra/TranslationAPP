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
def analyze_document_layout(pdf_path):
    """Analyze document layout including spacing, alignment, and structure"""
    try:
        doc = fitz.open(pdf_path)
        layout_info = {
            'font_hierarchy': {},
            'alignment_patterns': defaultdict(int),
            'spacing_patterns': [],
            'page_structure': []
        }
        
        for page_num, page in enumerate(doc):
            blocks = page.get_text("dict")["blocks"]
            page_elements = []
            prev_y1 = 0
            
            for block in blocks:
                if "lines" in block:
                    bbox = block['bbox']
                    x0, y0, x1, y1 = bbox
                    
                    # Calculate spacing from previous element
                    vertical_spacing = y0 - prev_y1 if prev_y1 > 0 else 0
                    
                    # Detect alignment
                    page_width = page.rect.width
                    if x0 < page_width * 0.15:
                        alignment = 'left'
                    elif x0 > page_width * 0.3:
                        alignment = 'indented'
                    else:
                        alignment = 'left_margin'
                    
                    if abs(x0 - page_width/2 + (x1-x0)/2) < 50:
                        alignment = 'center'
                    
                    layout_info['alignment_patterns'][alignment] += 1
                    
                    # Extract text and font info
                    block_text = ""
                    font_sizes = []
                    font_flags = []
                    
                    for line in block["lines"]:
                        for span in line["spans"]:
                            block_text += span["text"]
                            font_sizes.append(span["size"])
                            font_flags.append(span["flags"])
                    
                    if block_text.strip():
                        avg_font = sum(font_sizes) / len(font_sizes) if font_sizes else 12
                        is_bold = any(f & 2**4 for f in font_flags)
                        
                        page_elements.append({
                            'text': block_text.strip(),
                            'bbox': bbox,
                            'font_size': avg_font,
                            'is_bold': is_bold,
                            'alignment': alignment,
                            'spacing_before': vertical_spacing,
                            'page': page_num + 1
                        })
                        
                        layout_info['font_hierarchy'][(avg_font, is_bold)] = \
                            layout_info['font_hierarchy'].get((avg_font, is_bold), 0) + 1
                        
                        prev_y1 = y1
            
            layout_info['page_structure'].append(page_elements)
        
        doc.close()
        return layout_info
    
    except Exception as e:
        logger.error(f"Layout analysis failed: {e}")
        return None


# ============= ELEMENT CLASSIFICATION =============
def classify_element_type(element, font_hierarchy):
    """Classify document element based on font, position, and content"""
    text = element['text']
    font_size = element['font_size']
    is_bold = element['is_bold']
    alignment = element['alignment']
    spacing = element.get('spacing_before', 0)
    
    # Sort fonts by frequency to find main text size
    sorted_fonts = sorted(font_hierarchy.items(), key=lambda x: x[1], reverse=True)
    main_font_size = sorted_fonts[0][0][0] if sorted_fonts else 12
    
    # TITLE/HEADER - centered, bold, larger font
    if alignment == 'center' and (is_bold or font_size > main_font_size + 1):
        if spacing > 20 or len(text) < 100:
            return 'title'
    
    # MAIN HEADING - bold, larger, significant spacing
    if is_bold and font_size > main_font_size and spacing > 15:
        return 'heading'
    
    # SUBHEADING - bold or slightly larger
    if (is_bold or font_size > main_font_size * 0.9) and len(text) < 150:
        return 'subheading'
    
    # CASE NUMBER / DOCUMENT ID - specific patterns
    if re.match(r'^\d{4}.*(?:Appeal|SLP|Civil|Criminal)', text, re.IGNORECASE):
        return 'case_number'
    
    # NUMBERED SECTIONS - legal document numbering
    if re.match(r'^\d+[\.)]\s+[A-Z]', text):
        return 'numbered_section'
    
    # LIST ITEMS
    if re.match(r'^[•\-\*●○]\s', text):
        return 'bullet_item'
    if re.match(r'^[a-z][\.)]\s', text):
        return 'lettered_item'
    if re.match(r'^\([a-z]{1,3}\)\s', text):
        return 'parenthetical_item'
    
    # INDENTED PARAGRAPH
    if alignment == 'indented':
        return 'indented_paragraph'
    
    # PARTY NAMES (vs / versus pattern)
    if re.search(r'\s+(?:vs?\.?|versus)\s+', text, re.IGNORECASE):
        return 'party_names'
    
    # Regular paragraph
    return 'paragraph'


# ============= DOCUMENT EXTRACTION =============
def extract_structured_document(pdf_path):
    """Extract document with complete structure preservation"""
    try:
        layout_info = analyze_document_layout(pdf_path)
        if not layout_info:
            raise ValueError("Failed to analyze document layout")
        
        structured_document = []
        
        for page_elements in layout_info['page_structure']:
            for element in page_elements:
                element_type = classify_element_type(element, layout_info['font_hierarchy'])
                
                structured_element = {
                    'type': element_type,
                    'text': element['text'],
                    'alignment': element['alignment'],
                    'spacing_before': element['spacing_before'],
                    'is_bold': element['is_bold'],
                    'font_size': element['font_size'],
                    'page': element['page']
                }
                
                structured_document.append(structured_element)
        
        return structured_document
    
    except Exception as e:
        logger.error(f"Structured extraction failed: {e}")
        return []


# ============= TRANSLATION WITH STRUCTURE =============
def translate_structured_elements(structured_doc, src_lang, tgt_lang, tokenizer, model):
    """Translate while preserving structure metadata"""
    translated_elements = []
    
    for element in tqdm(structured_doc, desc="Translating document"):
        if not element['text'].strip():
            translated_elements.append(element)
            continue
        
        # Translate the text
        translated_text = translate_chunk(
            element['text'], 
            src_lang, 
            tgt_lang, 
            tokenizer, 
            model
        )
        
        # Create new element with translated text but same structure
        translated_element = {
            **element,  # Copy all metadata
            'text': translated_text,
            'original_text': element['text']  # Keep original for reference
        }
        
        translated_elements.append(translated_element)
        
        # Periodic cache clearing
        if len(translated_elements) % 10 == 0:
            clear_cpu_cache()
    
    return translated_elements


# ============= FORMAT OUTPUT =============
def format_structured_output(structured_elements, target_language):
    """Format document preserving original structure and spacing"""
    output_lines = []
    prev_spacing = 0
    
    for i, element in enumerate(structured_elements):
        elem_type = element['type']
        text = element['text']
        spacing = element.get('spacing_before', 0)
        alignment = element.get('alignment', 'left')
        
        # Add vertical spacing based on original document
        if spacing > 30:
            output_lines.append('\n\n\n')  # Large gap
        elif spacing > 15:
            output_lines.append('\n\n')    # Medium gap
        elif spacing > 5 and prev_spacing > 0:
            output_lines.append('\n')      # Small gap
        
        # Format based on element type
        if elem_type == 'title':
            output_lines.append(f'<center><h1>{text}</h1></center>\n')
        
        elif elem_type == 'heading':
            output_lines.append(f'<h2>{text}</h2>\n')
        
        elif elem_type == 'subheading':
            output_lines.append(f'<h3>{text}</h3>\n')
        
        elif elem_type == 'case_number':
            output_lines.append(f'<p class="case-number">{text}</p>\n')
        
        elif elem_type == 'party_names':
            output_lines.append(f'<p class="party-names">{text}</p>\n')
        
        elif elem_type == 'numbered_section':
            output_lines.append(f'<p class="numbered">{text}</p>\n')
        
        elif elem_type == 'bullet_item':
            text_clean = re.sub(r'^[•\-\*●○]\s*', '', text)
            output_lines.append(f'<li>{text_clean}</li>\n')
        
        elif elem_type == 'lettered_item':
            output_lines.append(f'<li class="lettered">{text}</li>\n')
        
        elif elem_type == 'parenthetical_item':
            output_lines.append(f'<li class="parenthetical">{text}</li>\n')
        
        elif elem_type == 'indented_paragraph':
            output_lines.append(f'<p class="indented">{text}</p>\n')
        
        else:  # Regular paragraph
            if alignment == 'center':
                output_lines.append(f'<p class="center">{text}</p>\n')
            else:
                output_lines.append(f'<p>{text}</p>\n')
        
        prev_spacing = spacing
    
    return ''.join(output_lines)


# ============= PDF CREATION =============
def create_styled_pdf(html_content, output_path):
    """Create PDF with proper legal document styling"""
    
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            @page {{
                size: A4;
                margin: 2.5cm 2cm 2.5cm 2cm;
            }}
            
            body {{
                font-family: 'DejaVu Serif', 'Times New Roman', serif;
                font-size: 12pt;
                line-height: 1.6;
                color: #000;
            }}
            
            h1 {{
                font-size: 14pt;
                font-weight: bold;
                text-align: center;
                margin: 0;
                padding: 0;
                line-height: 1.4;
            }}
            
            h2 {{
                font-size: 13pt;
                font-weight: bold;
                margin: 20pt 0 10pt 0;
                text-align: left;
            }}
            
            h3 {{
                font-size: 12pt;
                font-weight: bold;
                margin: 15pt 0 8pt 0;
            }}
            
            p {{
                margin: 0;
                padding: 0;
                text-align: justify;
                line-height: 1.5;
            }}
            
            p.center {{
                text-align: center;
                font-weight: bold;
            }}
            
            p.case-number {{
                text-align: center;
                font-size: 11pt;
                margin: 5pt 0;
            }}
            
            p.party-names {{
                text-align: center;
                font-weight: bold;
                margin: 10pt 0;
            }}
            
            p.numbered {{
                margin-left: 0;
                text-indent: 0;
                margin-top: 10pt;
            }}
            
            p.indented {{
                margin-left: 40pt;
                text-indent: 0;
            }}
            
            li {{
                margin-left: 30pt;
                margin-bottom: 5pt;
            }}
            
            li.lettered {{
                list-style-type: lower-alpha;
            }}
            
            li.parenthetical {{
                list-style-type: none;
            }}
            
            center {{
                display: block;
                text-align: center;
            }}
            
            h1, h2, h3 {{
                page-break-after: avoid;
                page-break-inside: avoid;
            }}
            
            p {{
                orphans: 3;
                widows: 3;
            }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
    options = {
        'encoding': 'UTF-8',
        'page-size': 'A4',
        'margin-top': '25mm',
        'margin-right': '20mm',
        'margin-bottom': '25mm',
        'margin-left': '20mm',
        'enable-local-file-access': None,
        'no-outline': None,
        'quiet': ''
    }
    
    try:
        pdfkit.from_string(full_html, output_path, options=options)
        logger.info(f"✅ Formatted PDF created: {output_path}")
    except Exception as e:
        logger.error(f"PDF creation failed: {e}")
        try:
            pdfkit.from_string(full_html, output_path)
            logger.info(f"✅ PDF created (fallback mode): {output_path}")
        except Exception as e2:
            logger.error(f"Fallback PDF creation also failed: {e2}")


# ============= MAIN TRANSLATION FUNCTION =============
def translate_pdf_document(pdf_path, job_id=None):
    """
    Main translation function with structure preservation
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
        
        # Extract structured document
        logger.info("Extracting document structure...")
        structured_doc = extract_structured_document(pdf_path)
        
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
        
        # Translate with structure preservation
        logger.info("Translating document...")
        translated_elements = translate_structured_elements(
            structured_doc, src_language, tgt_language, tokenizer, model
        )
        
        if job_id:
            translation_progress[job_id] = {'progress': 80, 'step': 'Formatting output...'}
        
        # Format output
        logger.info("Formatting output...")
        original_formatted = format_structured_output(structured_doc, src_language)
        translated_formatted = format_structured_output(translated_elements, tgt_language)
        
        # Create PDF
        output_pdf = "translated_output.pdf"
        create_styled_pdf(translated_formatted, output_pdf)
        
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