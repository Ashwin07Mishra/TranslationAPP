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
from typing import List, Dict, Any, Tuple
import json

# Docling imports for advanced structure detection
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import PdfFormatOption

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
        en_hi_model_name = "ai4bharat/indictrans2-en-indic-dist-200M"
        
        for attempt in range(max_retries):
            try:
                clear_all_cache()
                tokenizers['en_hi'] = AutoTokenizer.from_pretrained(en_hi_model_name, trust_remote_code=True)
                models['en_hi'] = AutoModelForSeq2SeqLM.from_pretrained(
                    en_hi_model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float32,
                ).to(DEVICE)
                logger.info("English to Hindi model loaded successfully")
                break
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed to load English to Hindi model: {e}")
                if attempt == max_retries - 1:
                    logger.error("All attempts failed for English to Hindi model")
                    raise
                time.sleep(2)
        
        # Load Hindi to English model
        logger.info("Loading Hindi to English model...")
        hi_en_model_name = "ai4bharat/indictrans2-indic-en-dist-200M"
        
        for attempt in range(max_retries):
            try:
                clear_all_cache()
                tokenizers['hi_en'] = AutoTokenizer.from_pretrained(hi_en_model_name, trust_remote_code=True)
                models['hi_en'] = AutoModelForSeq2SeqLM.from_pretrained(
                    hi_en_model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float32,
                ).to(DEVICE)
                logger.info("Hindi to English model loaded successfully")
                break
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed to load Hindi to English model: {e}")
                if attempt == max_retries - 1:
                    logger.error("All attempts failed for Hindi to English model")
                    raise
                time.sleep(2)
        
        return models, tokenizers
    except Exception as e:
        logger.error(f"Model initialization failed: {e}")
        clear_all_cache()
        raise

def initialize_docling_converter():
    """Initialize Docling converter with optimized settings for CPU"""
    try:
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True
        
        pdf_format_option = PdfFormatOption(pipeline_options=pipeline_options)
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: pdf_format_option,
            }
        )
        
        logger.info("Docling converter initialized successfully")
        return converter
    except Exception as e:
        logger.error(f"Failed to initialize Docling converter: {e}")
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

def extract_structure_with_docling(pdf_path: str) -> Dict[str, Any]:
    """Extract document structure using Docling's advanced PDF understanding"""
    try:
        logger.info("Initializing Docling converter...")
        converter = initialize_docling_converter()
        
        logger.info(f"Processing PDF with Docling: {pdf_path}")
        result = converter.convert(pdf_path)
        
        doc = result.document
        
        logger.info(f"Document type: {type(doc)}")
        
        try:
            num_pages = doc.num_pages() if callable(doc.num_pages) else doc.num_pages
        except:
            num_pages = len(doc.pages) if hasattr(doc, 'pages') else 1
        
        logger.info(f"Document has {num_pages} pages")
        
        structured_elements = []
        
        # Method 1: Access texts directly
        try:
            if hasattr(doc, 'texts') and doc.texts is not None and len(doc.texts) > 0:
                logger.info(f"Found {len(doc.texts)} text items in doc.texts")
                
                for idx, text_item in enumerate(doc.texts):
                    text_content = ""
                    
                    if hasattr(text_item, 'text'):
                        text_content = str(text_item.text).strip()
                    elif hasattr(text_item, 'content'):
                        text_content = str(text_item.content).strip()
                    else:
                        try:
                            text_content = str(text_item).strip()
                        except:
                            continue
                    
                    if not text_content or len(text_content) < 3:
                        continue
                    
                    page_num = 1
                    if hasattr(text_item, 'prov') and text_item.prov:
                        try:
                            if isinstance(text_item.prov, list) and len(text_item.prov) > 0:
                                prov_item = text_item.prov[0]
                                if hasattr(prov_item, 'page_no'):
                                    page_num = prov_item.page_no
                                elif isinstance(prov_item, dict):
                                    page_num = prov_item.get('page_no', 1)
                        except:
                            pass
                    
                    label = "text"
                    if hasattr(text_item, 'label'):
                        label = str(text_item.label)
                    
                    element = {
                        'type': 'paragraph',
                        'text': text_content,
                        'level': 0,
                        'page': page_num,
                        'original_type': label
                    }
                    
                    if 'title' in label.lower() or 'heading' in label.lower():
                        element['type'] = 'heading'
                        element['level'] = 1
                    elif 'header' in label.lower():
                        element['type'] = 'header'
                    elif 'footer' in label.lower():
                        element['type'] = 'footer'
                    
                    structured_elements.append(element)
                
                logger.info(f"Extracted {len(structured_elements)} elements from doc.texts")
        
        except Exception as e1:
            logger.warning(f"doc.texts access failed: {e1}")
        
        # Method 2: Export to markdown if no texts found
        if not structured_elements:
            try:
                logger.info("Exporting document to markdown...")
                markdown_text = doc.export_to_markdown()
                
                if markdown_text and markdown_text.strip():
                    logger.info(f"Markdown export successful, length: {len(markdown_text)}")
                    
                    lines = markdown_text.split('\n')
                    current_para = []
                    page_num = 1
                    
                    for line in lines:
                        stripped_line = line.strip()
                        
                        if not stripped_line:
                            if current_para:
                                para_text = ' '.join(current_para).strip()
                                if para_text and len(para_text) >= 3:
                                    structured_elements.append({
                                        'type': 'paragraph',
                                        'text': para_text,
                                        'level': 0,
                                        'page': page_num,
                                        'original_type': 'text'
                                    })
                                current_para = []
                        else:
                            if stripped_line.startswith('#'):
                                if current_para:
                                    para_text = ' '.join(current_para).strip()
                                    if para_text and len(para_text) >= 3:
                                        structured_elements.append({
                                            'type': 'paragraph',
                                            'text': para_text,
                                            'level': 0,
                                            'page': page_num,
                                            'original_type': 'text'
                                        })
                                    current_para = []
                                
                                heading_level = len(stripped_line) - len(stripped_line.lstrip('#'))
                                heading_text = stripped_line.lstrip('#').strip()
                                if heading_text and len(heading_text) >= 3:
                                    structured_elements.append({
                                        'type': 'heading',
                                        'text': heading_text,
                                        'level': heading_level,
                                        'page': page_num,
                                        'original_type': 'heading'
                                    })
                            elif stripped_line.startswith('---'):
                                if current_para:
                                    para_text = ' '.join(current_para).strip()
                                    if para_text and len(para_text) >= 3:
                                        structured_elements.append({
                                            'type': 'paragraph',
                                            'text': para_text,
                                            'level': 0,
                                            'page': page_num,
                                            'original_type': 'text'
                                        })
                                    current_para = []
                                page_num += 1
                            else:
                                clean_line = stripped_line
                                clean_line = re.sub(r'\*\*(.*?)\*\*', r'\1', clean_line)
                                clean_line = re.sub(r'\*(.*?)\*', r'\1', clean_line)
                                clean_line = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', clean_line)
                                clean_line = re.sub(r'`(.*?)`', r'\1', clean_line)
                                
                                if clean_line.strip():
                                    current_para.append(clean_line)
                    
                    if current_para:
                        para_text = ' '.join(current_para).strip()
                        if para_text and len(para_text) >= 3:
                            structured_elements.append({
                                'type': 'paragraph',
                                'text': para_text,
                                'level': 0,
                                'page': page_num,
                                'original_type': 'text'
                            })
                    
                    logger.info(f"Extracted {len(structured_elements)} elements from markdown")
                else:
                    raise ValueError("Markdown export is empty")
            
            except Exception as e2:
                logger.warning(f"Markdown export failed: {e2}, trying text export...")
        
        # Method 3: Plain text export
        if not structured_elements:
            try:
                logger.info("Trying plain text export...")
                plain_text = doc.export_to_text()
                
                if plain_text and plain_text.strip():
                    logger.info(f"Plain text export successful, length: {len(plain_text)}")
                    
                    paragraphs = re.split(r'\n\s*\n', plain_text)
                    
                    for para in paragraphs:
                        para = para.strip()
                        if para and len(para) >= 3:
                            para = re.sub(r'\s+', ' ', para)
                            structured_elements.append({
                                'type': 'paragraph',
                                'text': para,
                                'level': 0,
                                'page': 1,
                                'original_type': 'text'
                            })
                    
                    logger.info(f"Extracted {len(structured_elements)} elements from plain text")
                else:
                    raise ValueError("Plain text export is empty")
            
            except Exception as e3:
                logger.error(f"Text export failed: {e3}")
        
        if not structured_elements:
            error_msg = (
                f"Failed to extract any text from PDF '{pdf_path}'. "
                f"The document may be image-only (OCR is disabled), empty, corrupted, or password-protected."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        doc_title = ""
        if hasattr(doc, 'name') and doc.name:
            doc_title = str(doc.name)
        
        pages_count = max(elem['page'] for elem in structured_elements) if structured_elements else 1
        
        logger.info(f"Successfully extracted {len(structured_elements)} structured elements across {pages_count} page(s)")
        
        if structured_elements:
            sample = structured_elements[0]['text'][:150] if len(structured_elements[0]['text']) > 150 else structured_elements[0]['text']
            logger.info(f"Sample text: {sample}...")
        
        return {
            'elements': structured_elements,
            'metadata': {
                'title': doc_title,
                'pages': pages_count,
                'total_elements': len(structured_elements)
            },
            'raw_docling_doc': doc
        }
    
    except Exception as e:
        logger.error(f"Docling structure extraction failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def enhance_structure_analysis(structured_data: Dict[str, Any]) -> Dict[str, Any]:
    """Enhance structure analysis with additional legal document patterns"""
    elements = structured_data['elements']
    enhanced_elements = []
    
    current_section = None
    
    for element in elements:
        text = element['text'].strip()
        
        if not text:
            continue
        
        enhanced_element = element.copy()
        
        if any(keyword in text.lower() for keyword in ['supreme court', 'high court', 'अदालत', 'न्यायालय']):
            enhanced_element['type'] = 'court_header'
            enhanced_element['level'] = 0
        
        elif re.search(r'(civil appeal|petition|s\.l\.p\.|writ petition)', text.lower()):
            enhanced_element['type'] = 'case_citation'
        
        elif re.match(r'^j\s*u\s*d\s*g\s*m\s*e\s*n\s*t', text.lower().replace(' ', '')):
            enhanced_element['type'] = 'judgment_header'
            enhanced_element['level'] = 1
        
        elif re.search(r'(j\.|justice|judge|जज)', text.lower()):
            enhanced_element['type'] = 'authority'
        
        elif re.match(r'^\d+\.\s+', text):
            number = re.match(r'^(\d+)\.\s+', text).group(1)
            enhanced_element['type'] = 'main_paragraph'
            enhanced_element['number'] = int(number)
            enhanced_element['level'] = 2
            current_section = number
        
        elif re.match(r'^[A-Z]\.\s+', text):
            letter = re.match(r'^([A-Z])\.\s+', text).group(1)
            enhanced_element['type'] = 'sub_paragraph'
            enhanced_element['letter'] = letter
            enhanced_element['level'] = 3
            enhanced_element['parent_section'] = current_section
        
        elif re.match(r'^\([a-z]\)\s+', text):
            subletter = re.match(r'^\(([a-z])\)\s+', text).group(1)
            enhanced_element['type'] = 'sub_sub_paragraph'
            enhanced_element['subletter'] = subletter
            enhanced_element['level'] = 4
            enhanced_element['parent_section'] = current_section
        
        elif re.match(r'^[ivxlc]+\.\s+', text.lower()):
            enhanced_element['type'] = 'roman_paragraph'
            enhanced_element['level'] = 3
        
        elif len(text) < 150 and (
            text.isupper() or 
            any(keyword in text.lower() for keyword in ['background', 'facts', 'arguments', 'conclusion', 'order'])
        ):
            enhanced_element['type'] = 'section_heading'
            enhanced_element['level'] = 2
        
        enhanced_elements.append(enhanced_element)
    
    structured_data['elements'] = enhanced_elements
    return structured_data

def clean_hindi_text(text):
    """Remove extra spaces and fix Hindi punctuation"""
    if not text:
        return ""
    
    try:
        text = re.sub(r'\s+', ' ', text)
        text = text.replace('।।', '।')
        text = text.replace('॥', '।')
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
        text = re.sub(r'[\u0900-\u097F]+', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text, flags=re.IGNORECASE)
        text = re.sub(r'([.!?])\1+', r'\1', text)
        text = re.sub(r'\s+([.!?,:;])', r'\1', text)
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
        text = re.sub(r'\s+[।]+\s+', ' ', text)
        return text.strip()
    except Exception as e:
        logger.warning(f"English text cleaning failed: {e}")
        return text

def translate_chunk_safe(chunk, src_lang, tgt_lang, tokenizer, model):
    """
    Ultra-safe translation with complete error isolation
    Returns tuple: (translated_text, success_flag)
    """
    if not chunk or not chunk.strip():
        return "", False
    
    chunk = chunk.strip()
    
    if len(chunk) < 3:
        return chunk, True
    
    try:
        if src_lang == "hin_Deva":
            chunk = clean_hindi_text(chunk)
        
        batch = ip.preprocess_batch([chunk], src_lang=src_lang, tgt_lang=tgt_lang)
        if not batch or not batch[0]:
            return chunk, False
        
        inputs = tokenizer(
            batch,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )
        
        if "attention_mask" not in inputs or inputs["attention_mask"] is None:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        
        if inputs["input_ids"] is None or inputs["attention_mask"] is None:
            return chunk, False
        
        if inputs["input_ids"].shape[0] == 0 or inputs["attention_mask"].shape[0] == 0:
            return chunk, False
        
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        with torch.no_grad():
            try:
                generated_tokens = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=150,
                    num_beams=2,
                    early_stopping=True,
                )
            except Exception as gen_error:
                logger.debug(f"Generation failed: {gen_error}")
                return chunk, False
        
        if generated_tokens is None or generated_tokens.numel() == 0:
            return chunk, False
        
        translated = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        if not translated or not translated.strip():
            return chunk, False
        
        translated = ip.postprocess_batch([translated], lang=tgt_lang)[0]
        
        if tgt_lang == "eng_Latn":
            translated = clean_english_output(translated)
        
        return translated, True
        
    except Exception as e:
        logger.debug(f"Translation error: {e}")
        return chunk, False

def translate_structured_elements_robust(structured_data: Dict[str, Any], src_language: str, tgt_language: str, tokenizer, model) -> Dict[str, Any]:
    """Robust translation with proper error handling and statistics"""
    elements = structured_data['elements']
    translated_elements = []
    
    stats = {
        'total': len(elements),
        'successful': 0,
        'failed': 0,
        'skipped': 0
    }
    
    logger.info(f"Starting translation of {len(elements)} elements...")
    
    for i, element in enumerate(tqdm(elements, desc="Translating")):
        translated_element = element.copy()
        original_text = element['text'].strip()
        
        if not original_text:
            translated_element['text'] = ""
            translated_elements.append(translated_element)
            stats['skipped'] += 1
            continue
        
        if len(original_text) < 3:
            translated_element['text'] = original_text
            translated_elements.append(translated_element)
            stats['skipped'] += 1
            continue
        
        translated_text, success = translate_chunk_safe(
            original_text,
            src_language,
            tgt_language,
            tokenizer,
            model
        )
        
        if success and translated_text:
            translated_element['text'] = translated_text
            stats['successful'] += 1
        else:
            translated_element['text'] = original_text
            translated_element['translation_failed'] = True
            stats['failed'] += 1
        
        translated_elements.append(translated_element)
        
        if i % 10 == 0 and i > 0:
            clear_cpu_cache()
    
    logger.info(f"Translation complete:")
    logger.info(f"  Total: {stats['total']}")
    logger.info(f"  Successful: {stats['successful']}")
    logger.info(f"  Failed (kept original): {stats['failed']}")
    logger.info(f"  Skipped (too short/empty): {stats['skipped']}")
    
    translated_data = structured_data.copy()
    translated_data['elements'] = translated_elements
    translated_data['metadata']['translated'] = True
    translated_data['metadata']['source_language'] = src_language
    translated_data['metadata']['target_language'] = tgt_language
    translated_data['metadata']['translation_stats'] = stats
    
    return translated_data

def reconstruct_formatted_document(translated_data: Dict[str, Any]) -> str:
    """Reconstruct document with perfect structure preservation like original PDF"""
    elements = translated_data['elements']
    formatted_lines = []
    
    if translated_data['metadata'].get('title'):
        formatted_lines.append(f"# {translated_data['metadata']['title']}\n\n")
    
    current_section = None
    
    for element in elements:
        text = element['text'].strip()
        if not text:
            continue
        
        element_type = element['type']
        level = element.get('level', 0)
        
        if element_type == 'court_header':
            formatted_lines.append(f"**{text}**\n\n")
        
        elif element_type == 'case_citation':
            formatted_lines.append(f"*{text}*\n\n")
        
        elif element_type == 'judgment_header':
            formatted_lines.append(f"## {text}\n\n")
        
        elif element_type == 'authority':
            formatted_lines.append(f"**{text}**\n\n")
        
        elif element_type == 'section_heading':
            formatted_lines.append(f"### {text}\n\n")
        
        elif element_type == 'main_paragraph':
            number = element.get('number', '')
            formatted_lines.append(f"{number}. {text}\n\n")
            current_section = number
        
        elif element_type == 'sub_paragraph':
            letter = element.get('letter', '')
            formatted_lines.append(f"   {letter}. {text}\n\n")
        
        elif element_type == 'sub_sub_paragraph':
            subletter = element.get('subletter', '')
            formatted_lines.append(f"      ({subletter}) {text}\n\n")
        
        elif element_type == 'roman_paragraph':
            formatted_lines.append(f"   {text}\n\n")
        
        elif element_type == 'list_item':
            formatted_lines.append(f"• {text}\n\n")
        
        elif element_type == 'table':
            formatted_lines.append(f"```\n{text}\n```\n\n")
        
        elif element_type == 'header':
            formatted_lines.append(f"---\n**{text}**\n---\n\n")
        
        elif element_type == 'footer':
            formatted_lines.append(f"---\n*{text}*\n---\n\n")
        
        else:
            formatted_lines.append(f"{text}\n\n")
    
    final_document = "".join(formatted_lines)
    final_document = re.sub(r'\n\n\n+', '\n\n', final_document)
    
    return final_document.strip()

def export_to_json(structured_data: Dict[str, Any], output_path: str):
    """Export structured data to JSON for analysis"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(structured_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Structured data exported to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to export JSON: {e}")

def get_document_statistics(structured_data: Dict[str, Any]) -> Dict[str, Any]:
    """Get detailed document statistics"""
    elements = structured_data['elements']
    
    stats = {
        'total_elements': len(elements),
        'element_types': {},
        'pages': len(set(elem['page'] for elem in elements)),
        'total_words': 0,
        'total_characters': 0
    }
    
    for element in elements:
        element_type = element['type']
        stats['element_types'][element_type] = stats['element_types'].get(element_type, 0) + 1
        
        text = element['text']
        stats['total_words'] += len(text.split())
        stats['total_characters'] += len(text)
    
    return stats

translation_progress = {}

def translate_pdf_document(pdf_path, job_id=None):
    """
    Main translation function with Docling structure preservation
    Returns: dict with original_text, translated_text, source_language, target_language, metadata
    """
    models = None
    tokenizers = None
    
    try:
        clear_all_cache()
        
        logger.info("Loading IndicTrans translation models...")
        models, tokenizers = initialize_models()
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if job_id:
            translation_progress[job_id] = {'progress': 10, 'step': 'Extracting document structure...'}
        
        logger.info("Extracting document structure with Docling...")
        structured_data = extract_structure_with_docling(pdf_path)
        
        if not structured_data['elements']:
            raise ValueError("No elements extracted from the document")
        
        if job_id:
            translation_progress[job_id] = {'progress': 30, 'step': 'Analyzing document structure...'}
        
        logger.info("Enhancing structure analysis...")
        structured_data = enhance_structure_analysis(structured_data)
        
        # Sample first 10 non-empty elements for language detection
        sample_text = " ".join([
            elem['text'] for elem in structured_data['elements'][:10] 
            if elem['text'].strip()
        ])
        
        if not sample_text:
            raise ValueError("No valid text available for language detection")
        
        src_language = detect_language(sample_text)
        tgt_language = "hin_Deva" if src_language == "eng_Latn" else "eng_Latn"
        
        logger.info(f"Detected source: {src_language}, target: {tgt_language}")
        
        model_key = 'en_hi' if src_language == "eng_Latn" else 'hi_en'
        model = models[model_key]
        tokenizer = tokenizers[model_key]
        
        if job_id:
            translation_progress[job_id] = {'progress': 40, 'step': f'Translating {len(structured_data["elements"])} elements...'}
        
        logger.info("Starting robust translation...")
        translated_data = translate_structured_elements_robust(
            structured_data,
            src_language,
            tgt_language,
            tokenizer,
            model
        )
        
        if job_id:
            translation_progress[job_id] = {'progress': 90, 'step': 'Formatting document...'}
        
        logger.info("Reconstructing formatted documents...")
        final_translation = reconstruct_formatted_document(translated_data)
        
        original_data = structured_data.copy()
        original_data['metadata']['translated'] = False
        original_formatted = reconstruct_formatted_document(original_data)
        
        clear_all_cache()
        
        if job_id:
            translation_progress[job_id] = {'progress': 100, 'step': 'Complete!'}
        
        return {
            'original_text': original_formatted,
            'translated_text': final_translation,
            'source_language': src_language,
            'target_language': tgt_language,
            'metadata': translated_data['metadata']
        }
    
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        clear_all_cache()
        
        return {
            'original_text': "",
            'translated_text': f"Translation failed: {str(e)}",
            'source_language': "unknown",
            'target_language': "unknown",
            'metadata': {}
        }
    
    finally:
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
            logger.warning(f"Cleanup failed: {e}")