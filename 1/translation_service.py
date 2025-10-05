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

def hindi_sentence_tokenizer(paragraph):
    """Split Hindi text at sentence boundaries (।!?)"""
    if not paragraph:
        return []
    
    try:
        # Clean the paragraph first
        paragraph = clean_hindi_text(paragraph)
        # Split on Hindi sentence endings
        sentences = re.split(r'(?<=[।!?])\s+', paragraph)
        # Also handle periods in mixed text
        final_sentences = []
        for sent in sentences:
            if '.' in sent and not sent.endswith('।'):
                sub_sents = re.split(r'(?<=[.])\s+', sent)
                final_sentences.extend(sub_sents)
            else:
                final_sentences.append(sent)
        
        return [s.strip() for s in final_sentences if s.strip()]
    except Exception as e:
        logger.warning(f"Hindi sentence tokenization failed: {e}")
        return [paragraph] if paragraph else []

def english_sentence_tokenizer(paragraph):
    """Split English text using SpaCy sentence detection"""
    if not paragraph:
        return []
    
    try:
        doc = nlp(paragraph)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    except Exception as e:
        logger.warning(f"English sentence tokenization failed: {e}")
        # Fallback to simple period splitting
        sentences = re.split(r'(?<=[.!?])\s+', paragraph)
        return [s.strip() for s in sentences if s.strip()]

def extract_text(pdf_path):
    """Extract all text from PDF file with improved error recovery"""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    all_text = []
    max_retries = 2
    
    for attempt in range(max_retries):
        try:
            with pdfplumber.open(pdf_path) as pdf:
                logger.info(f"Processing PDF with {len(pdf.pages)} pages (attempt {attempt + 1})")
                
                for i, page in enumerate(pdf.pages):
                    try:
                        text = page.extract_text()
                        if text:
                            lines = text.split("\n")
                            page_text = "\n".join(lines)
                            all_text.append(page_text)
                            logger.debug(f"Extracted text from page {i+1}")
                        else:
                            logger.warning(f"No text found on page {i+1}")
                    except Exception as e:
                        logger.warning(f"Failed to extract text from page {i+1}: {e}")
                        # Try alternative extraction method
                        try:
                            reader = PdfReader(pdf_path)
                            if i < len(reader.pages):
                                alt_text = reader.pages[i].extract_text()
                                if alt_text:
                                    all_text.append(alt_text)
                                    logger.info(f"Alternative extraction succeeded for page {i+1}")
                        except Exception as alt_e:
                            logger.warning(f"Alternative extraction also failed for page {i+1}: {alt_e}")
                        continue
            
            # If we got here without exception, break out of retry loop
            break
        except Exception as e:
            logger.error(f"PDF extraction attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(1)  # Wait before retry
    
    if not all_text:
        raise ValueError("No text could be extracted from the PDF")
    
    return "\n\n".join(all_text)

def reconstruct_paragraphs(text, src_language):
    """Rebuild proper paragraphs from extracted text"""
    if not text:
        return []
    
    try:
        raw_paragraphs = text.split("\n\n")
        cleaned_paragraphs = []
        
        # Choose punctuation marks based on language
        end_punctuation = ('।', '!', '?', ':') if src_language == "hin_Deva" else ('.', '!', '?', ':')
        
        for raw_para in raw_paragraphs:
            lines = raw_para.split("\n")
            merged = ""
            
            for i, line in enumerate(lines):
                stripped = line.strip()
                if not stripped:
                    continue
                
                # Join lines that don't end with punctuation
                if merged and not merged.endswith(end_punctuation):
                    merged += " " + stripped
                else:
                    merged += (" " + stripped if merged else stripped)
            
            if merged.strip():
                # Clean based on source language
                if src_language == "hin_Deva":
                    merged = clean_hindi_text(merged.strip())
                cleaned_paragraphs.append(merged)
        
        return cleaned_paragraphs
    except Exception as e:
        logger.error(f"Paragraph reconstruction failed: {e}")
        return [text]  # Return original text as single paragraph

def smart_chunking_with_context(paragraph, src_language, tokenizer, max_tokens=180):
    """Break paragraph into chunks while keeping context from previous sentences"""
    if not paragraph or not paragraph.strip():
        return []
    
    try:
        # Choose sentence splitter based on language
        if src_language == "hin_Deva":
            sentences = hindi_sentence_tokenizer(paragraph)
        else:
            sentences = english_sentence_tokenizer(paragraph)
        
        if len(sentences) <= 1:
            return [(paragraph, paragraph)]  # Single sentence needs no chunking
        
        chunk_groups = []  # Groups of sentences that fit together
        current_group = []
        
        for sentence in sentences:
            # Check if adding this sentence exceeds token limit
            test_group = current_group + [sentence]
            test_chunk = " ".join(test_group)
            
            try:
                if get_token_count(test_chunk, tokenizer, ip, src_language) <= max_tokens:
                    current_group.append(sentence)
                else:
                    # Save current group and start new one
                    if current_group:
                        chunk_groups.append(current_group[:])
                    
                    # Handle sentences that are too long by themselves
                    if get_token_count(sentence, tokenizer, ip, src_language) > max_tokens:
                        word_chunks = split_long_sentence(sentence, tokenizer, src_language, max_tokens)
                        for word_chunk in word_chunks:
                            chunk_groups.append([word_chunk])
                        current_group = []
                    else:
                        current_group = [sentence]
            except Exception as e:
                logger.warning(f"Token counting failed for sentence: {e}")
                # Fallback: assume sentence is okay
                current_group.append(sentence)
        
        # Add remaining sentences
        if current_group:
            chunk_groups.append(current_group)
        
        # Create context chunks: include previous sentence for context
        context_chunks = []
        for i, group in enumerate(chunk_groups):
            context_sentences = []
            # Add previous sentence for context if available
            if i > 0 and chunk_groups[i-1]:
                context_sentences.extend(chunk_groups[i-1][-1:])
            # Add current group
            context_sentences.extend(group)
            
            # Create context chunk and output portion
            context_chunk = " ".join(context_sentences)
            output_chunk = " ".join(group)  # Only new sentences for output
            context_chunks.append((context_chunk, output_chunk))
        
        return context_chunks
    except Exception as e:
        logger.error(f"Smart chunking failed: {e}")
        return [(paragraph, paragraph)]  # Return original as fallback

def split_long_sentence(sentence, tokenizer, src_language, max_tokens):
    """Split overly long sentences by words"""
    if not sentence:
        return []
    
    try:
        words = sentence.split()
        word_chunks = []
        current_words = []
        
        for word in words:
            test_text = " ".join(current_words + [word])
            try:
                if get_token_count(test_text, tokenizer, ip, src_language) <= max_tokens:
                    current_words.append(word)
                else:
                    if current_words:
                        word_chunks.append(" ".join(current_words))
                    current_words = [word]
            except Exception as e:
                logger.warning(f"Token counting failed for word splitting: {e}")
                current_words.append(word)  # Continue anyway
        
        if current_words:
            word_chunks.append(" ".join(current_words))
        
        return word_chunks if word_chunks else [sentence]
    except Exception as e:
        logger.error(f"Long sentence splitting failed: {e}")
        return [sentence]

def translate_chunk(chunk, src_lang, tgt_lang, tokenizer, model):
    """Translate a text chunk using the AI model with improved error recovery"""
    if not chunk or not chunk.strip():
        return ""
    
    max_retries = 2
    for attempt in range(max_retries):
        try:
            # Clear cache before translation if this is a retry
            if attempt > 0:
                clear_gpu_cache()
            
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

def map_translation_to_output(full_translation, context_chunk, output_chunk, src_language):
    """Extract only the new translation part (not the context part)"""
    if not full_translation:
        return ""
    
    try:
        if context_chunk == output_chunk:
            return full_translation
        
        # Get sentence lists for both chunks
        if src_language == "hin_Deva":
            context_sentences = hindi_sentence_tokenizer(context_chunk)
            output_sentences = hindi_sentence_tokenizer(output_chunk)
            translated_sentences = hindi_sentence_tokenizer(full_translation)
        else:
            context_sentences = english_sentence_tokenizer(context_chunk)
            output_sentences = english_sentence_tokenizer(output_chunk)
            translated_sentences = english_sentence_tokenizer(full_translation)
        
        if len(output_sentences) < len(context_sentences):
            # Try to get the last N sentences where N = number of new sentences
            if len(translated_sentences) >= len(output_sentences):
                relevant_translation = " ".join(translated_sentences[-len(output_sentences):])
                return relevant_translation
        
        # If extraction fails, return full translation
        return full_translation
    except Exception as e:
        logger.warning(f"Translation mapping failed: {e}")
        return full_translation

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

def post_process_document(text, target_language):
    """Final cleanup of the complete translated document"""
    if not text:
        return ""
    
    try:
        if target_language == "eng_Latn":
            # Clean English output
            text = clean_english_output(text)
            
            # Fix paragraph spacing
            text = re.sub(r'\n\n+', '\n\n', text)
            
            # Capitalize first letter of sentences
            sentences = text.split('. ')
            capitalized_sentences = []
            for sent in sentences:
                sent = sent.strip()
                if sent:
                    sent = sent[0].upper() + sent[1:] if len(sent) > 1 else sent.upper()
                    capitalized_sentences.append(sent)
            text = '. '.join(capitalized_sentences)
            
            # Add final punctuation if missing
            if not text.endswith(('.', '!', '?')):
                text += '.'
        
        return text
    except Exception as e:
        logger.warning(f"Document post-processing failed: {e}")
        return text

translation_progress = {}

def translate_pdf_document(pdf_path, job_id=None):
    """
    Translate PDF document using the provided path with improved error recovery
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
        
        # Use the provided pdf_path (don't override it!)
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        logger.info("Extracting text from PDF...")
        text = extract_text(pdf_path)
        
        # Detect document language
        logger.info("Detecting document language...")
        src_language = detect_language(text)
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
        
        # Reconstruct paragraphs
        logger.info("Reconstructing paragraphs...")
        paragraphs = reconstruct_paragraphs(text, src_language)
        logger.info(f"Found {len(paragraphs)} paragraphs")
        
        if job_id:
            translation_progress[job_id] = {'progress': 0, 'step': 'Starting translation...'}
        
        # Translate paragraphs using smart chunking
        translated_paragraphs = []
        successful_translations = 0
        failed_paragraphs = 0
        
        for i, para in tqdm(enumerate(paragraphs), total=len(paragraphs), desc="Translating paragraphs"):
            if not para.strip():
                continue
            
            # Calculate progress percentage
            progress_percent = int((i / len(paragraphs)) * 80) + 10  # 10-90% range
            
            # Update progress storage
            if job_id:
                translation_progress[job_id] = {
                    'progress': progress_percent,
                    'step': f'Translating paragraph {i+1} of {len(paragraphs)}'
                }
            
            # Clear cache periodically during translation
            if i % 10 == 0:
                clear_gpu_cache()
            
            try:
                # Break paragraph into context-aware chunks
                chunk_pairs = smart_chunking_with_context(para, src_language, tokenizer, max_tokens=180)
                translated_outputs = []
                
                for context_chunk, output_chunk in chunk_pairs:
                    if context_chunk.strip():
                        # Translate using full context
                        full_translation = translate_chunk(context_chunk, src_language, tgt_language, tokenizer, model)
                        
                        if full_translation.strip():
                            # Extract only the new part
                            output_translation = map_translation_to_output(
                                full_translation, context_chunk, output_chunk, src_language
                            )
                            
                            if output_translation.strip():
                                translated_outputs.append(output_translation)
                
                if translated_outputs:
                    translated_paragraphs.append(" ".join(translated_outputs))
                    successful_translations += 1
                else:
                    failed_paragraphs += 1
                    logger.warning(f"No translation output for paragraph {i+1}")
                    
            except Exception as e:
                failed_paragraphs += 1
                logger.error(f"Failed to translate paragraph {i+1}: {e}")
                continue
        
        logger.info(f"Translation completed: {successful_translations} successful, {failed_paragraphs} failed")
        
        # Handle case where no paragraphs were translated
        if not translated_paragraphs:
            logger.error("No paragraphs were successfully translated")
            # Return a minimal translation result instead of None
            return {
                'original_text': text,
                'translated_text': "Translation failed - no content could be processed",
                'source_language': src_language,
                'target_language': tgt_language
            }
        
        # Join and clean final document
        logger.info("Post-processing final document...")
        final_translation = "\n\n".join(translated_paragraphs)
        final_translation = post_process_document(final_translation, tgt_language)
        
        # Final cache clear
        clear_all_cache()
        
        # Return the required format for Flask app
        return {
            'original_text': text,
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