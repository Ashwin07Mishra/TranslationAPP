"""
Translation service for PDF documents with structure preservation.
Entity protection DISABLED to prevent marker artifacts in output.
"""

import sys
import os
import re
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor
import gc
import logging
import unicodedata

# Import the fixed formatter
from .formatter import (
    extract_structured_document,
    format_structured_output,
    format_document
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

translation_progress = {}


def clear_all_cache():
    """Clear GPU and CPU cache to free memory"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass
    gc.collect()


def detect_language(text):
    """Simple character-based language detection for Hindi vs English"""
    if not text:
        return "eng_Latn"
    
    # Count Devanagari characters
    dev_chars = len(re.findall(r"[\u0900-\u097F]", text))
    total_chars = len(re.sub(r"\s", "", text))
    
    if total_chars == 0:
        return "eng_Latn"
    
    # If more than 20% Devanagari, it's Hindi
    return "hin_Deva" if (dev_chars / total_chars) > 0.2 else "eng_Latn"


def initialize_models():
    """Load translation models for both directions"""
    models, tokenizers = {}, {}
    
    try:
        logger.info("Loading translation models...")
        clear_all_cache()
        
        # Load English to Hindi model
        en_hi_model = "ai4bharat/indictrans2-en-indic-dist-200M"
        tokenizers["en_hi"] = AutoTokenizer.from_pretrained(en_hi_model, trust_remote_code=True)
        models["en_hi"] = AutoModelForSeq2SeqLM.from_pretrained(
            en_hi_model,
            trust_remote_code=True,
            torch_dtype=torch.float32 if DEVICE == "cpu" else torch.float16
        ).to(DEVICE)
        logger.info("✅ EN→HI model ready")
        
        clear_all_cache()
        
        # Load Hindi to English model
        hi_en_model = "ai4bharat/indictrans2-indic-en-dist-200M"
        tokenizers["hi_en"] = AutoTokenizer.from_pretrained(hi_en_model, trust_remote_code=True)
        models["hi_en"] = AutoModelForSeq2SeqLM.from_pretrained(
            hi_en_model,
            trust_remote_code=True,
            torch_dtype=torch.float32 if DEVICE == "cpu" else torch.float16
        ).to(DEVICE)
        logger.info("✅ HI→EN model ready")
        
        return models, tokenizers
        
    except Exception as e:
        logger.error(f"❌ Failed to load models: {e}")
        clear_all_cache()
        raise


# Initialize IndicProcessor
try:
    ip = IndicProcessor(inference=True)
    logger.info("✅ IndicProcessor ready")
except Exception as e:
    logger.error(f"❌ IndicProcessor initialization failed: {e}")
    sys.exit(1)


def translate_text_run(run, src_lang, tgt_lang, tokenizer, model):
    """
    Translate single text run WITHOUT entity protection.
    Entity protection disabled to prevent XENTX marker artifacts.
    """
    text = run.get('text', '')
    
    if not text or len(text.strip()) < 3:
        return text
    
    try:
        # ENTITY PROTECTION DISABLED - translating directly
        # This prevents XENTX markers from appearing in output
        
        # Preprocess text for translation
        batch = ip.preprocess_batch([text], src_lang=src_lang, tgt_lang=tgt_lang)
        
        # Tokenize
        inputs = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt"
        ).to(DEVICE)
        
        # Generate translation
        with torch.no_grad():
            tokens = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=256,
                num_beams=3,
                early_stopping=True,
                no_repeat_ngram_size=2,
                repetition_penalty=1.2,
            )
        
        # Decode translation
        translated = tokenizer.decode(tokens[0], skip_special_tokens=True)
        translated = ip.postprocess_batch([translated], lang=tgt_lang)[0]
        
        # ENTITY RESTORATION DISABLED - not needed since protection is off
        
        # Clean up spacing and normalize Unicode
        translated = re.sub(r'\s+', ' ', translated)
        translated = unicodedata.normalize('NFC', translated)
        
        return translated.strip()
    
    except Exception as e:
        logger.error(f"Translation error for text '{text[:30]}...': {e}")
        return text


def translate_text_runs(text_runs, src_lang, tgt_lang, tokenizer, model, job_id=None):
    """Translate all text runs with progress tracking"""
    translated_runs = []
    total = len(text_runs)
    
    for idx, run in enumerate(text_runs):
        # Update progress
        if job_id:
            progress = 25 + int(50 * (idx / total))
            translation_progress[job_id] = {
                "progress": progress,
                "step": f"Translating {idx + 1}/{total}"
            }
        
        # Translate this text run
        translated_text = translate_text_run(run, src_lang, tgt_lang, tokenizer, model)
        
        # Create new run with translated text
        new_run = run.copy()
        new_run['text'] = translated_text
        translated_runs.append(new_run)
    
    return translated_runs


def translate_pdf_document(pdf_path, job_id=None, output_path=None):
    """
    Main translation pipeline for PDF documents.
    
    Args:
        pdf_path: Path to input PDF file
        job_id: Optional job ID for progress tracking
        output_path: Explicit output path for the translated PDF
        
    Returns:
        Dictionary with translation results and metadata
    """
    models = tokenizers = None
    
    try:
        clear_all_cache()
        logger.info(f"🚀 Starting translation: {pdf_path}")
        
        # Update progress: Loading models
        if job_id:
            translation_progress[job_id] = {"progress": 10, "step": "Loading models..."}
        
        models, tokenizers = initialize_models()
        
        # Verify input file exists
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        # Update progress: Analyzing document
        if job_id:
            translation_progress[job_id] = {"progress": 20, "step": "Analyzing document..."}
        
        # Extract structured content from PDF
        text_runs, layout_info = extract_structured_document(pdf_path)
        
        if not text_runs:
            raise ValueError("No content extracted from PDF")
        
        logger.info(f"📊 Extracted {len(text_runs)} text runs from document")
        
        # ENTITY PROTECTION LEARNING DISABLED
        # protector.learn_from_document(text_runs) - commented out to prevent markers
        
        # Detect source language from sample text
        sample_texts = [r['text'] for r in text_runs[:30]]
        sample_text = " ".join(sample_texts)
        src_lang = detect_language(sample_text)
        tgt_lang = "hin_Deva" if src_lang == "eng_Latn" else "eng_Latn"
        
        logger.info(f"🌐 Translation: {src_lang} → {tgt_lang}")
        
        # Select appropriate model
        model_key = "en_hi" if src_lang == "eng_Latn" else "hi_en"
        model = models.get(model_key)
        tokenizer = tokenizers.get(model_key)
        
        if model is None or tokenizer is None:
            raise RuntimeError("Required model/tokenizer not loaded")
        
        # Update progress: Translating
        if job_id:
            translation_progress[job_id] = {"progress": 25, "step": "Translating..."}
        
        logger.info("🔄 Translating text runs...")
        translated_runs = translate_text_runs(
            text_runs,
            src_lang,
            tgt_lang,
            tokenizer,
            model,
            job_id
        )
        
        # Update progress: Creating PDF
        if job_id:
            translation_progress[job_id] = {"progress": 75, "step": "Creating PDF..."}
        
        logger.info("🎨 Recreating PDF with translations...")
        
        # Generate HTML previews
        original_html = format_structured_output(text_runs, src_lang)
        
        # Determine output path
        if output_path is None:
            input_dir = os.path.dirname(os.path.abspath(pdf_path)) or "."
            if job_id:
                output_pdf_path = os.path.join(input_dir, f"{job_id}_translated_output.pdf")
            else:
                output_pdf_path = os.path.join(input_dir, "translated_output.pdf")
        else:
            output_pdf_path = output_path
        
        logger.info(f"📝 Output will be saved to: {output_pdf_path}")
        
        # Create formatted PDF with translations
        result = format_document(
            pdf_path=pdf_path,
            structured_elements=translated_runs,
            target_language=tgt_lang,
            output_pdf=output_pdf_path
        )
        
        translated_html = format_structured_output(translated_runs, tgt_lang)
        translated_pdf_path = result.get('pdf_path', '')
        
        # Verify output file was created
        if not os.path.exists(translated_pdf_path):
            raise FileNotFoundError(f"Output PDF was not created at: {translated_pdf_path}")
        
        # Update progress: Complete
        if job_id:
            translation_progress[job_id] = {"progress": 100, "step": "Complete"}
        
        clear_all_cache()
        
        logger.info(f"✅ Translation complete: {translated_pdf_path}")
        
        return {
            "original_text": original_html,
            "translated_text": translated_html,
            "source_language": src_lang,
            "target_language": tgt_lang,
            "translated_pdf_path": translated_pdf_path,
            "element_count": len(text_runs)
        }
    
    except Exception as e:
        logger.error(f"❌ Translation failed: {e}")
        import traceback
        traceback.print_exc()
        
        if job_id:
            translation_progress[job_id] = {"progress": 0, "step": f"Error: {str(e)}"}
        
        clear_all_cache()
        
        return {
            "original_text": "",
            "translated_text": f"Translation failed: {str(e)}",
            "source_language": "unknown",
            "target_language": "unknown",
            "translated_pdf_path": "",
            "element_count": 0
        }
    
    finally:
        # Clean up models and tokenizers to free memory
        try:
            if models:
                for key, m in list(models.items()):
                    try:
                        m.cpu()
                        del m
                    except Exception:
                        pass
                models.clear()
        except Exception:
            pass
        
        try:
            if tokenizers:
                tokenizers.clear()
        except Exception:
            pass
        
        clear_all_cache()
