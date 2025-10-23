import os
import fitz
import re
import unicodedata
from collections import defaultdict


class RobustFontDetector:
    """Detects and manages fonts for different scripts with size optimization"""
    def __init__(self):
        self.english_font = "helv"
        self.hindi_font_path = None
        self._register_unicode_fonts()
    
    def _register_unicode_fonts(self):
        """Find Hindi fonts - prefers Regular over Variable for smaller file size"""
        try:
            custom_font_dir = os.path.expanduser("~/Downloads/Noto_Sans_Devanagari")
            
            # Check for fonts in order of preference (Regular is much smaller)
            font_paths = [
                # Prefer Regular weight (smaller file size)
                os.path.join(custom_font_dir, "NotoSansDevanagari-Regular.ttf"),
                # Fallback to variable font (larger)
                os.path.join(custom_font_dir, "NotoSansDevanagari-VariableFont_wdth,wght.ttf"),
                # System fonts
                "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
                "/usr/share/fonts/truetype/noto/NotoSansDevanagari[wght].ttf",
            ]
            
            for font_path in font_paths:
                if os.path.exists(font_path):
                    self.hindi_font_path = font_path
                    font_size_mb = os.path.getsize(font_path) / (1024 * 1024)
                    print(f"✅ Found Hindi font: {font_path} ({font_size_mb:.2f} MB)")
                    return
            
            print("⚠️ No Hindi font found - boxes will appear!")
        except Exception as e:
            print(f"⚠️ Font detection error: {e}")
    
    def get_font_path(self):
        """Returns the detected Hindi font path"""
        return self.hindi_font_path
    
    def get_font_dir(self):
        """Returns directory containing the font"""
        if self.hindi_font_path:
            return os.path.dirname(self.hindi_font_path)
        return None


class IntelligentDocumentAnalyzer:
    """Analyzes document structure and extracts text runs with better layout detection"""
    
    def __init__(self, pdf_path):
        self.doc = fitz.open(pdf_path)
        self.page_count = len(self.doc)
        self.text_runs = []
        self.repeating_elements = {}
        
        # Detect page dimensions from first page
        first_page = self.doc[0]
        self.page_width = first_page.rect.width
        self.page_height = first_page.rect.height
        print(f"📏 Page size: {self.page_width} x {self.page_height}")
    
    def analyze(self):
        """Complete analysis"""
        print("🔍 Analyzing document...")
        self._find_repeating_elements()
        self._extract_text_runs()
        print(f"✓ Extracted {len(self.text_runs)} text runs")
        print(f"✓ Filtered {len(self.repeating_elements)} repeating elements")
        return self.text_runs
    
    def _find_repeating_elements(self):
        """Find headers/footers by repetition"""
        position_map = defaultdict(list)
        
        for page_num, page in enumerate(self.doc):
            blocks = page.get_text("blocks")
            page_height = page.rect.height
            
            for block in blocks:
                if block[6] == 0:  # text block
                    text = block[4].strip()
                    y_pos = block[1]
                    y_percent = int((y_pos / page_height) * 100)
                    
                    # Create key with text and relative position
                    key = (text[:50], y_percent)
                    position_map[key].append(page_num)
        
        # Elements appearing on 40%+ of pages are likely headers/footers
        threshold = max(3, int(self.page_count * 0.4))
        for (text, y_percent), pages in position_map.items():
            if len(pages) >= threshold:
                self.repeating_elements[(text, y_percent)] = pages
    
    def _is_repeating_element(self, text, y_pos, page_height):
        """Check if text is a repeating header/footer"""
        y_percent = int((y_pos / page_height) * 100)
        key = (text[:50], y_percent)
        return key in self.repeating_elements
    
    def _detect_alignment(self, bbox, page_width):
        """Better alignment detection"""
        x0, y0, x1, y1 = bbox
        text_center = (x0 + x1) / 2
        text_width = x1 - x0
        
        # Define margins (10% on each side)
        left_margin = page_width * 0.1
        right_margin = page_width * 0.9
        
        # Center detection: text center is in middle third
        center_start = page_width * 0.33
        center_end = page_width * 0.67
        
        if center_start < text_center < center_end and text_width < page_width * 0.6:
            return "center"
        elif x0 > right_margin - page_width * 0.05:
            return "right"
        else:
            return "left"
    
    def _extract_text_runs(self):
        """Extract all text with formatting info"""
        for page_num, page in enumerate(self.doc):
            page_dict = page.get_text("dict")
            page_height = page.rect.height
            page_width = page.rect.width
            
            for block in page_dict["blocks"]:
                if block["type"] != 0:  # skip image blocks
                    continue
                
                for line in block["lines"]:
                    # Process full line for better spacing
                    line_text = ""
                    line_spans = []
                    
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if text:
                            line_text += text + " "
                            line_spans.append(span)
                    
                    if not line_text.strip() or len(line_text.strip()) < 2:
                        continue
                    
                    # Use first span for position/formatting
                    if not line_spans:
                        continue
                    
                    first_span = line_spans[0]
                    bbox = first_span["bbox"]
                    y_pos = bbox[1]
                    
                    # Skip repeating elements
                    if self._is_repeating_element(line_text.strip(), y_pos, page_height):
                        continue
                    
                    font_size = first_span["size"]
                    flags = first_span["flags"]
                    font_name = first_span.get("font", "")
                    
                    is_bold = bool(flags & 16) or "Bold" in font_name
                    is_italic = bool(flags & 2) or "Italic" in font_name
                    
                    # Clean text
                    text = self._clean_text(line_text.strip())
                    if not text:
                        continue
                    
                    # Detect alignment
                    alignment = self._detect_alignment(bbox, page_width)
                    
                    # Calculate proper spacing
                    line_spacing = font_size * 1.2  # 120% of font size
                    
                    self.text_runs.append({
                        'text': text,
                        'page': page_num + 1,
                        'x': bbox[0],
                        'y': y_pos,
                        'font_size': round(font_size, 1),
                        'is_bold': is_bold,
                        'is_italic': is_italic,
                        'font_name': font_name,
                        'align': alignment,
                        'line_spacing': line_spacing,
                        'page_width': page_width,
                        'page_height': page_height
                    })
    
    def _clean_text(self, text):
        """Clean and normalize text"""
        text = text.replace('\ufffd', '')
        text = re.sub(r'�+', '', text)
        text = unicodedata.normalize('NFC', text)
        return text.strip()
    
    def close(self):
        """Close document"""
        self.doc.close()


class PDFRecreator:
    """Recreates a PDF with translated text using insert_htmlbox for proper Hindi rendering"""
    
    def __init__(self, output_path):
        self.output_path = output_path
        self.font_detector = RobustFontDetector()
    
    def create_pdf(self, text_runs, target_language='hin_Deva'):
        """Create optimized PDF with proper font embedding and compression"""
        print(f"📄 Creating PDF: {self.output_path}")
        
        if not text_runs:
            print("⚠️ No text runs to process")
            return False
        
        # Get page dimensions from first run
        first_run = text_runs[0]
        page_width = first_run.get('page_width', 595)
        page_height = first_run.get('page_height', 842)
        
        doc = fitz.open()
        is_hindi = "Deva" in target_language or "hin" in target_language
        
        # Group by page
        pages = defaultdict(list)
        for run in text_runs:
            pages[run['page']].append(run)
        
        # Configure font for Hindi
        font_path = self.font_detector.get_font_path()
        font_dir = self.font_detector.get_font_dir()
        
        if not font_path or not os.path.exists(font_path):
            print("❌ ERROR: Hindi font not found! Text will appear as boxes.")
            print(f"   Expected font at: ~/Downloads/Noto_Sans_Devanagari/")
            return False
        
        # CSS configuration for Hindi
        if is_hindi:
            font_filename = os.path.basename(font_path)
            
            # Check if variable font or regular
            if "Variable" in font_filename:
                css = f"""
                @font-face {{
                    font-family: 'DevanagariFont';
                    src: url('{font_filename}');
                    font-weight: 100 900;
                    font-stretch: 75% 125%;
                }}
                * {{
                    font-family: 'DevanagariFont', 'Noto Sans Devanagari', sans-serif;
                    line-height: 1.5;
                }}
                """
            else:
                css = f"""
                @font-face {{
                    font-family: 'DevanagariFont';
                    src: url('{font_filename}');
                }}
                * {{
                    font-family: 'DevanagariFont', 'Noto Sans Devanagari', sans-serif;
                    line-height: 1.5;
                }}
                """
            
            archive = fitz.Archive(font_dir)
            print(f"✅ Using font: {font_path}")
            print(f"✅ Font archive: {font_dir}")
        else:
            css = "* {font-family: sans-serif; line-height: 1.5;}"
            archive = None
        
        # Create each page
        for page_num in sorted(pages.keys()):
            page = doc.new_page(width=page_width, height=page_height)
            
            # Sort runs by Y position
            sorted_runs = sorted(pages[page_num], key=lambda r: r.get('y', 0))
            
            for run in sorted_runs:
                text = run.get('text', '')
                if not text:
                    continue
                
                x = run.get('x', 10)
                y = run.get('y', 10)
                font_size = run.get('font_size', 12)
                is_bold = run.get('is_bold', False)
                alignment = run.get('align', 'left')
                
                # Create HTML with proper styling
                style_parts = [f"font-size: {font_size}px"]
                
                if is_bold:
                    style_parts.append("font-weight: bold")
                
                # Map alignment
                if alignment == "center":
                    style_parts.append("text-align: center")
                elif alignment == "right":
                    style_parts.append("text-align: right")
                else:
                    style_parts.append("text-align: left")
                
                style_str = "; ".join(style_parts)
                
                # Escape HTML special characters
                text_escaped = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                html_text = f'<div style="{style_str}">{text_escaped}</div>'
                
                # Calculate rectangle for text box
                estimated_width = len(text) * font_size * 0.6
                line_height = font_size * 1.8
                
                # Adjust rectangle based on alignment
                if alignment == "center":
                    text_rect = fitz.Rect(
                        page_width * 0.1,
                        y - font_size,
                        page_width * 0.9,
                        y + line_height
                    )
                elif alignment == "right":
                    text_rect = fitz.Rect(
                        max(10, page_width * 0.3),
                        y - font_size,
                        page_width - 20,
                        y + line_height
                    )
                else:  # left
                    text_rect = fitz.Rect(
                        x,
                        y - font_size,
                        min(page_width - 20, x + estimated_width + 100),
                        y + line_height
                    )
                
                try:
                    # Use insert_htmlbox for proper text shaping (CRITICAL for Hindi)
                    page.insert_htmlbox(
                        text_rect,
                        html_text,
                        css=css,
                        archive=archive
                    )
                except Exception as e:
                    print(f"⚠️ Failed to insert text: {text[:30]}... Error: {e}")
        
        # Ensure output directory exists
        outdir = os.path.dirname(self.output_path)
        if outdir and not os.path.exists(outdir):
            os.makedirs(outdir, exist_ok=True)
        
        try:
            print("📦 Optimizing PDF (subsetting fonts, removing unused data)...")
            
            # CRITICAL: Subset fonts BEFORE saving to drastically reduce file size
            # This removes all unused glyphs from embedded fonts
            doc.subset_fonts(verbose=True)
            
            # Additional optimization: remove unused objects and compress
            doc.scrub()  # Remove unused/duplicate objects
            
            # Get size before optimization
            temp_path = self.output_path + ".temp"
            doc.save(temp_path)
            size_before = os.path.getsize(temp_path) / (1024 * 1024)
            os.remove(temp_path)
            
            # Save with maximum compression
            doc.save(
                self.output_path,
                garbage=4,           # Maximum garbage collection (removes all unused objects)
                deflate=True,        # Compress all streams
                clean=True,          # Clean up PDF structure
                linear=False         # Don't linearize (saves time)
            )
            
            doc.close()
            
            # Check final size
            final_size = os.path.getsize(self.output_path)
            final_size_mb = final_size / (1024 * 1024)
            final_size_kb = final_size / 1024
            
            print(f"✅ PDF created successfully: {self.output_path}")
            if final_size_mb > 1:
                print(f"📏 Final size: {final_size_mb:.2f} MB (reduced from ~{size_before:.2f} MB)")
            else:
                print(f"📏 Final size: {final_size_kb:.2f} KB")
            
            # Warn if file is still large
            if final_size_mb > 50:
                print(f"⚠️ Warning: PDF is still large ({final_size_mb:.2f} MB)")
                print("   Consider using a Regular font instead of Variable font")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to save PDF: {e}")
            import traceback
            traceback.print_exc()
            doc.close()
            return False


def extract_structured_document(pdf_path=None, existing_layout_info=None):
    """Extract document structure"""
    analyzer = IntelligentDocumentAnalyzer(pdf_path)
    text_runs = analyzer.analyze()
    
    layout_info = {
        'page_count': analyzer.page_count,
        'repeating_elements': len(analyzer.repeating_elements),
        'page_width': analyzer.page_width,
        'page_height': analyzer.page_height
    }
    
    analyzer.close()
    return text_runs, layout_info


def merge_translated_text(original_runs, translated_texts):
    """Merge translated text into the same structure as original_runs"""
    merged = []
    for i, run in enumerate(original_runs):
        if i < len(translated_texts):
            merged.append({**run, "text": translated_texts[i]})
        else:
            merged.append(run)
    return merged


def format_document(pdf_path=None, structured_elements=None, target_language='eng_Latn',
                   output_pdf='output.pdf', existing_layout_info=None, translated_texts=None):
    """Create a formatted, structure-preserving PDF"""
    
    # If structured_elements is not provided, extract from pdf_path
    if structured_elements is None:
        structured_elements, layout_info = extract_structured_document(pdf_path)
    
    # If translated_texts provided and are simple strings, merge them
    if translated_texts:
        if len(translated_texts) > 0 and isinstance(translated_texts[0], str):
            structured_elements = merge_translated_text(structured_elements, translated_texts)
        else:
            structured_elements = translated_texts
    
    recreator = PDFRecreator(output_pdf)
    success = recreator.create_pdf(structured_elements, target_language)
    
    return {
        'pdf_path': output_pdf if success else '',
        'success': success
    }


def format_structured_output(structured_elements, target_language, font_statistics=None):
    """Create HTML output for preview"""
    html_parts = ['<div class="document-preview">']
    
    current_page = None
    for element in structured_elements:
        page = element.get('page', 1)
        
        # Page separator
        if page != current_page:
            if current_page is not None:
                html_parts.append('</div>')  # Close previous page
            html_parts.append(f'<div class="page" data-page="{page}">')
            html_parts.append(f'<div class="page-number">Page {page}</div>')
            current_page = page
        
        # Text element
        text = element.get('text', '')
        is_bold = element.get('is_bold', False)
        font_size = element.get('font_size', 12)
        alignment = element.get('align', 'left')
        
        style = f"font-size: {font_size}px; text-align: {alignment};"
        
        if is_bold:
            html_parts.append(f'<p style="{style}"><strong>{text}</strong></p>')
        else:
            html_parts.append(f'<p style="{style}">{text}</p>')
    
    if current_page is not None:
        html_parts.append('</div>')  # Close last page
    
    html_parts.append('</div>')
    
    return '\n'.join(html_parts)
