# rag_core.py — Fully‑Working Advanced RAG with PyQt6 + Thread Cleanup
import os, sys, threading, pickle, datetime, re
from typing import List, Dict, Optional, Any, Tuple

# Add parent directory to sys.path to find retrieval_engine.py
# This ensures that modules in the root of the workspace can be imported
current_script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(current_script_dir)
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

import requests, pdfplumber, docx, pytesseract
from PIL import Image
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Import semantic chunking and multimodal processing
from core.chunking import SemanticChunker, ChunkManifest, DocumentChunk, ChunkMetadata
from core.image_processor import ImageProcessor, ImageProcessingConfig, MultimodalChunker

from PyQt6.QtGui import (
    QFont, QPalette, QLinearGradient, QColor, QBrush,
    QRadialGradient, QConicalGradient, QFontDatabase,
    QPainter, QPen, QPainterPath, QIcon, QPixmap
)
from PyQt6.QtWidgets import (
    QApplication, QTabWidget, QWidget, QSplitter, QListWidget,
    QTreeWidget, QTreeWidgetItem, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QFileDialog, QProgressBar,
    QMessageBox, QCheckBox, QTextEdit, QScrollArea, QFrame,
    QGroupBox, QGraphicsDropShadowEffect, QDialog, QGridLayout,
    QSpinBox, QColorDialog, QSlider, QDoubleSpinBox, QLineEdit,
    QRadioButton, QButtonGroup, QFontComboBox, QToolButton,
    QDockWidget, QMainWindow, QMenuBar, QMenu, QStatusBar,
    QToolBar, QSizePolicy, QSpacerItem, QInputDialog
)
from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal, QTimer, QPropertyAnimation,
    QEasingCurve, QPoint, QSize, QRect, QObject, pyqtProperty,
    QSettings
)
from collections import Counter, defaultdict # Add these if not already present
import math # Add this if not already present

# Import new imports
from core.embedding import VectorIndexManager, IndexEntry # Make sure IndexEntry is imported if Hydroid uses it directly (it does via VIM.search_index)
from retrieval_engine import HydroidRetrievalEngine, SimpleBM25 # Import new engine and also SimpleBM25 if it was removed from here

# --- Animated Gradient + Styling ---
def set_futuristic_style(app: QApplication):
    """Set the futuristic style for the application."""
    app.setStyleSheet("""
        QWidget { 
            background: #121212; 
            color: #EEE; 
            font-family: -apple-system, BlinkMacSystemFont, '.SFNSText-Regular', sans-serif; 
        }
        QPushButton { 
            background: #1F1F1F; 
            border: 2px solid #2D2D2D; 
            border-radius: 8px; 
            padding: 8px;
            min-width: 100px;
        }
        QPushButton:hover { 
            border-color: #00FFAA;
            background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0, stop: 0 #1F1F1F, stop: 1 #2F2F2F);
            margin: -1px;
            border-width: 3px;
        }
        QPushButton:pressed {
            background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0, stop: 0 #2F2F2F, stop: 1 #1F1F1F);
        }
        QPushButton#askBtn {
            background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0, stop: 0 #FF4500, stop: 1 #FFD700);
            color: #000; 
            font-weight: bold; 
            font-size: 16px;
            border: none;
            padding: 12px;
        }
        QPushButton#askBtn:hover {
            background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0, stop: 0 #FFD700, stop: 1 #FF4500);
            margin: -2px;
            border: 2px solid #00FFAA;
            border-radius: 10px;
        }
        QPushButton#askBtn:pressed {
            background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0, stop: 0 #FF4500, stop: 1 #FFD700);
            margin: 0px;
            border: none;
        }
        QTabWidget::pane { 
            border: 1px solid #2D2D2D; 
            border-radius: 6px; 
        }
        QTabBar::tab { 
            background: #1F1F1F; 
            padding: 8px; 
            border: 1px solid #2D2D2D;
            border-radius: 4px;
        }
        QTabBar::tab:selected { 
            background: #272727; 
            border-bottom: 2px solid #00FFAA;
        }
        QListWidget, QTreeWidget, QComboBox, QTextEdit, QProgressBar {
            background: #1E1E1E;
            border: 1px solid #2D2D2D;
            border-radius: 4px;
            padding: 4px;
        }
        QProgressBar::chunk { 
            background: #00FFAA;
            border-radius: 4px;
        }
        QCheckBox { 
            spacing: 6px; 
        }
        QScrollBar:vertical {
            border: none;
            background: #1E1E1E;
            width: 10px;
            margin: 0px;
        }
        QScrollBar::handle:vertical {
            background: #2D2D2D;
            min-height: 20px;
            border-radius: 5px;
        }
        QScrollBar::handle:vertical:hover {
            background: #00FFAA;
        }
    """)


class GradientWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.angle = 0
        self.setAutoFillBackground(True)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_gradient)
        self.timer.start(50)

    def update_gradient(self):
        self.angle = (self.angle + 1) % 360
        grad = QLinearGradient(0, 0, self.width(), self.height())
        grad.setColorAt(0, QColor.fromHsv(self.angle, 180, 50))
        grad.setColorAt(1, QColor.fromHsv((self.angle + 60) % 360, 180, 50))
        pal = self.palette()
        pal.setBrush(QPalette.ColorRole.Window, QBrush(grad))
        self.setPalette(pal)


# --- Config ---
class Config:
    FOLDER_PATH = os.path.expanduser("~/Desktop/cpp_notes")
    MODEL_PATH = "rag_notes_doc2vec.model"
    CACHE_PATH = "rag_cache.pkl"
    SESSION_PATH = "rag_history.pkl"
    CHUNK_MANIFEST_PATH = "rag_chunk_manifest.json"
    TOP_KEYWORDS = 20
    
    # Chunking parameters
    CHUNK_SIZE = 512          # Target chunk size in tokens
    CHUNK_OVERLAP = 50        # Overlap between chunks in tokens
    DETECT_SECTIONS = True    # Whether to detect sections by headings
    
    # Image processing parameters
    OCR_ENABLED = True
    CAPTIONING_ENABLED = True
    MAX_IMAGE_SIZE = (800, 800)  # Maximum image size for processing
    
    PROMPT_TMPL = """You are a helpful assistant. Only use the context below.

Context:
{context}

User Question: {query}
"""
    MODEL_OPTS = {
        "CodeLLaMA-13B": {
            "url": "http://127.0.0.1:1234/v1/completions",
            "params": {"temperature": 0.2, "max_tokens": 512, "top_p": 0.9}
        }
    }


# --- Text Extraction & Section Parsing ---
class SmartParser:
    @staticmethod
    def clean_text(text: str) -> str:
        """Advanced text cleaning and normalization."""
        import unicodedata

        # --- Preserve Code Blocks --- 
        code_blocks = []
        def preserve_code(match):
            code_blocks.append(match.group(0))
            return f"\n__CODE_BLOCK_{len(code_blocks) - 1}__\n"
        text = re.sub(r'(?:^|\n)(?: {4}[^\n]*\n?)+', preserve_code, text) # Indented
        text = re.sub(r'```(?:[a-zA-Z]+\n)?[\s\S]+?```', preserve_code, text) # Fenced

        # --- Preserve Tables --- 
        tables = []
        def preserve_table(match):
            tables.append(match.group(0))
            return f"\n__TABLE_BLOCK_{len(tables) - 1}__\n"
        table_pattern = r'(?:^|\n)(\|.*?\|\s*\n\|[-| :]+\|(?:\s*\n\|.*?\|)*)'
        text = re.sub(table_pattern, preserve_table, text, flags=re.MULTILINE)
        
        # --- Normalize Whitespace --- 
        text = text.replace('\r\n', '\n').replace('\r', '\n') # Normalize line endings
        text = re.sub(r'[ \t]+', ' ', text) # Collapse spaces/tabs within lines
        text = re.sub(r'^[ \t]+|[ \t]+$', '', text, flags=re.MULTILINE) # Trim lines
        text = re.sub(r'\n{3,}', '\n\n', text) # Collapse multiple blank lines
        text = text.strip() # Remove leading/trailing whitespace from the whole string

        # --- Normalize Punctuation --- 
        text = re.sub(r'[""‟]', '"', text)  # Double quotes
        text = re.sub(r"[''‛]", "'", text)  # Single quotes/apostrophes
        text = re.sub(r'\.{3,}', '...', text) # Ellipsis
        text = re.sub(r'[-‐‑‒–—―]', '-', text) # Hyphens/dashes

        # --- Remove Remaining Control Characters --- 
        text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C' or char == '\n')

        # --- Restore Preserved Blocks --- 
        for i, block in enumerate(code_blocks):
            # Restore without adding extra newlines if block already has them
            text = text.replace(f"__CODE_BLOCK_{i}__", block.strip('\n'))
        for i, table in enumerate(tables):
            text = text.replace(f"__TABLE_BLOCK_{i}__", table.strip('\n'))

        # One final pass to ensure paragraph spacing after restores
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text

    @staticmethod
    def normalize_technical_content(text: str) -> str:
        """Specifically handle technical content like code, equations, and technical terms."""
        # Preserve inline code
        text = re.sub(r'(?<!`)`([^`\n]+)`(?!`)', r'[CODE]\1[/CODE]', text)

        # Preserve mathematical expressions (both single and double dollar signs)
        text = re.sub(r'\$\$([^$]+)\$\$', r'[MATH]\1[/MATH]', text) # Display math
        text = re.sub(r'(?<!\$)\$([^$\n]+)\$(?!\$)', r'[MATH]\1[/MATH]', text) # Inline math

        # Preserve URLs (improved pattern)
        url_pattern = r'(https?://(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,6}(?:/[^\s<>"()\[\]\\]*)?|www\.[a-zA-Z0-9][a-zA-Z0-9-]{0,61}[a-zA-Z0-9]\.[a-zA-Z]{2,6}(?:/[^\s<>"()\[\]\\]*)?)'
        text = re.sub(url_pattern, r'[URL]\1[/URL]', text)

        # Preserve file paths (improved pattern for Unix/Windows, avoids common trailing punctuation)
        # Matches paths with at least one separator, ending in allowed characters
        path_pattern = r'((?:(?:[a-zA-Z]:)?(?:\\|/))|(?:\.\./|\./))[^\\/:*?"<>|\n\s]+(?:(?:\\|/)[^\\/:*?"<>|\n\s]+)*?[a-zA-Z0-9_.]'
        # Use finditer to apply tags carefully, avoiding nested tags or partial overlaps
        matches = list(re.finditer(path_pattern, text))
        processed_text = ''
        last_end = 0
        for match in matches:
            start, end = match.span()
            # Check if this match is already inside another tag
            if '[/PATH]' in text[last_end:start] and '[PATH]' not in text[last_end:start]: continue
            if '[/URL]' in text[last_end:start] and '[URL]' not in text[last_end:start]: continue
            if '[/CODE]' in text[last_end:start] and '[CODE]' not in text[last_end:start]: continue
            if '[/MATH]' in text[last_end:start] and '[MATH]' not in text[last_end:start]: continue
            
            processed_text += text[last_end:start]
            # Exclude trailing punctuation .,;:!? from the path itself before tagging
            path_str = match.group(0)
            trailing_punct = ''
            if path_str[-1] in '.,;:!?':
                trailing_punct = path_str[-1]
                path_str = path_str[:-1]
                end -= 1
            processed_text += f'[PATH]{path_str}[/PATH]{trailing_punct}'
            last_end = match.end()
        processed_text += text[last_end:]
        text = processed_text

        return text

    @staticmethod
    def extract(path: str) -> str:
        """Enhanced file content extraction with better error handling and format support."""
        ext = os.path.splitext(path)[1].lower()
        try:
            content = ""
            if ext == ".txt" or ext == ".html" or ext == ".htm" or ext == ".md":  # Added Markdown support
                with open(path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                    # For HTML files, try to strip HTML tags
                    if ext == ".html" or ext == ".htm":
                        try:
                            import re
                            # Simple HTML tag stripping - could use BeautifulSoup for better results
                            content = re.sub(r'<style.*?>.*?</style>', '', content, flags=re.DOTALL)
                            content = re.sub(r'<script.*?>.*?</script>', '', content, flags=re.DOTALL)
                            content = re.sub(r'<[^>]*>', ' ', content)
                            content = re.sub(r'&[^;]+;', ' ', content)
                            content = re.sub(r'\s+', ' ', content).strip()
                        except Exception as html_e:
                            print(f"Error stripping HTML tags: {html_e}, proceeding with raw content")
            elif ext == ".docx":
                doc = docx.Document(path)
                # Extract both paragraphs and tables
                content = []
                for element in doc.element.body:
                    if element.tag.endswith('p'):
                        content.append(doc.element.xpath('//w:p')[len(content)].text)
                    elif element.tag.endswith('tbl'):
                        table = doc.element.xpath('//w:tbl')[0]
                        rows = []
                        for row in table.xpath('.//w:tr'):
                            cells = [cell.text for cell in row.xpath('.//w:t')]
                            rows.append('|' + '|'.join(cells) + '|')
                        content.append('\n'.join(rows))
                content = '\n'.join(content)
            elif ext == ".pdf":
                pages = []
                with pdfplumber.open(path) as pdf:
                    for page in pdf.pages:
                        # Try text extraction first
                        text = page.extract_text()
                        if not text or len(text.strip()) < 10:
                            # If text extraction fails, try OCR
                            text = pytesseract.image_to_string(page.to_image().original)
                        # Extract tables if present
                        tables = page.extract_tables()
                        if tables:
                            for table in tables:
                                text += '\n' + '\n'.join(
                                    '|' + '|'.join(str(cell) for cell in row) + '|' for row in table)
                        pages.append(text)
                content = '\n'.join(pages)
            elif ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp"]:
                content = pytesseract.image_to_string(Image.open(path))
            else:
                raise ValueError(f"Unsupported file type: {ext}")

            # Clean and normalize the extracted content
            content = SmartParser.clean_text(content)
            content = SmartParser.normalize_technical_content(content)

            return content

        except Exception as e:
            print(f"Error processing {path}: {str(e)}")
            return ""

    @staticmethod
    def parse_sections(text: str) -> List[str]:
        secs = []
        lines = text.splitlines()

        # Common section patterns
        patterns = [
            r'^SECTION:.*$',  # Section header starting with SECTION:
            r'^[A-Z][A-Z\s]+$',  # All caps lines
            r'^[A-Z][a-z]+(?:\s[A-Z][a-z]+)*:$',  # Title case with colon
            r'^\d+\.\s+[A-Z][a-z]+',  # Numbered sections
            r'^[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\s*:$',  # Title case with optional colon
            r'^[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\s*\([A-Z][a-z]+\)$',  # Title with parenthetical
            # Markdown headers are handled separately below
        ]

        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for Markdown headers first (special handling)
            if re.match(r'^#{1,6}\s+.+$', line):
                # Strip the hash marks and whitespace from Markdown headers
                clean_header = re.sub(r'^#{1,6}\s*', '', line)
                if clean_header and len(clean_header) > 1:  # Ensure we don't get single-character headers
                    secs.append(clean_header)
                    print(f"Detected Markdown header: '{clean_header}'")
                continue
                
            # Check for section definitions
            if line.startswith("SECTION:"):
                secs.append(line)
                continue

            # Check if line matches any section pattern
            if any(re.match(pattern, line) for pattern in patterns):
                secs.append(line)
                continue

            # Additional checks for section headers
            elif (len(line) > 4 and
                  (line.isupper() or
                   line.endswith(':') or
                   line.startswith(('Chapter', 'Section', 'Part', 'Appendix')))):
                secs.append(line)

        # Remove duplicates while preserving order
        seen = set()
        result = [x for x in secs if not (x in seen or seen.add(x))]
        
        # Add debug output
        print(f"SmartParser found {len(result)} sections: {result[:5]}...")
        
        return result

    @staticmethod
    def extract_keywords(text: str, top_n: int = 20) -> List[str]:
        """Extract important keywords from text."""
        # Remove common stop words
        stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
        words = [word.lower() for word in re.findall(r'\w+', text) if word.lower() not in stop_words]

        # Count word frequencies
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1

        # Sort by frequency and return top N
        return sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:top_n]


# --- Document Indexing in Background ---
class DocumentProcessor(QThread):
    progress = pyqtSignal(int, int)
    complete = pyqtSignal(
        list, object, list, list,  # docs, model, names, sections
        object, dict,              # (tfidf_vec,tfidf_mat), keywords_map
        list                       # chunks (NEW)
    )

    def __init__(self, folder: str):
        super().__init__()
        self.folder = folder
        # Initialize chunkers
        self.semantic_chunker = SemanticChunker(
            target_chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            detect_sections=Config.DETECT_SECTIONS
        )
        # Initialize image processor
        self.image_config = ImageProcessingConfig(
            enable_ocr=Config.OCR_ENABLED,
            enable_captioning=Config.CAPTIONING_ENABLED,
            max_image_size=Config.MAX_IMAGE_SIZE
        )
        self.image_processor = ImageProcessor(self.image_config)
        # Initialize multimodal chunker
        self.multimodal_chunker = MultimodalChunker(
            base_chunker=self.semantic_chunker,
            image_processor=self.image_processor
        )
        # Initialize chunk manifest
        self.manifest = ChunkManifest(Config.CHUNK_MANIFEST_PATH)

    def run(self):
        files = [f for f in os.listdir(self.folder)
                 if f.lower().endswith((".txt", ".docx", ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".md", ".html", ".htm"))]
        docs, names, secs = [], [], []
        all_chunks = []  # Store all chunks
        
        print(f"Processing {len(files)} files from {self.folder}")
        
        for i, f in enumerate(files, 1):
            path = os.path.join(self.folder, f)
            doc_name = f
            
            is_image_file_type = f.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp"))
            
            print(f"Processing file {i}/{len(files)}: {f} ({os.path.getsize(path) / 1024:.1f} KB)")

            current_chunks_for_file = []
            combined_text_for_doc_model = ""

            if is_image_file_type:
                # Use the specialized image_processor for images
                try:
                    print(f"  Using image processor for {f}")
                    current_chunks_for_file = self.image_processor.process_image(
                        image_path=path,
                        doc_name=doc_name,
                        doc_path=path,
                        section_title=f"Image: {doc_name}",
                        chunker=self.semantic_chunker
                    )
                    if current_chunks_for_file:
                        # Combine text from OCR/caption for the document-level model
                        combined_text_for_doc_model = "\n\n".join(
                            f"{chunk.metadata.chunk_type.capitalize()}: {chunk.text}" 
                            for chunk in current_chunks_for_file if chunk.text and chunk.text.strip()
                        )
                        print(f"  ✓ Generated {len(current_chunks_for_file)} image chunks with {len(combined_text_for_doc_model)} chars of text")
                    else:
                        print(f"  ⚠ No chunks generated for image {f}")
                except Exception as img_e:
                    print(f"  ❌ Error processing image {f}: {str(img_e)}")
            else:
                # For non-image files (txt, pdf, docx, etc.), use SmartParser
                try:
                    print(f"  Using SmartParser.extract for {f}")
                    extracted_text_content = SmartParser.extract(path)
                    if extracted_text_content and extracted_text_content.strip():
                        print(f"  ✓ Extracted {len(extracted_text_content)} chars from {f}")
                        combined_text_for_doc_model = extracted_text_content
                        
                        # Get section titles before chunking - helps with accurate section tagging
                        section_titles = SmartParser.parse_sections(extracted_text_content)
                        print(f"  ✓ Found {len(section_titles)} section titles in {f}")
                        
                        current_chunks_for_file = self.semantic_chunker.chunk_document(
                            text=extracted_text_content,
                            doc_name=doc_name,
                            doc_path=path
                        )
                        print(f"  ✓ Generated {len(current_chunks_for_file)} text chunks from {f}")
                        
                        # Categorize and tag chunks by type (code, table, etc.)
                        for chunk in current_chunks_for_file:
                            # Detect code blocks
                            if "```" in chunk.text or any(line.startswith("    ") for line in chunk.text.split("\n")):
                                chunk.metadata.chunk_type = "code" 
                            # Detect tables
                            elif "|" in chunk.text and "-|-" in chunk.text.replace(" ", ""):
                                chunk.metadata.chunk_type = "table"
                            # Detect equations
                            elif "$" in chunk.text and any(symbol in chunk.text for symbol in ["\\sum", "\\int", "\\frac"]):
                                chunk.metadata.chunk_type = "math"
                    else:
                        print(f"  ⚠ No content extracted from {f}")
                        self.progress.emit(i, len(files))
                        continue 
                except Exception as text_e:
                    print(f"  ❌ Error extracting text from {f}: {str(text_e)}")
                    self.progress.emit(i, len(files))
                    continue

            if current_chunks_for_file:
                for chunk in current_chunks_for_file:
                    # Make sure chunk has a hash before adding
                    if not chunk.metadata.content_hash:
                        chunk.update_hash()
                    
                    # Apply section priority weights for certain section titles
                    if chunk.metadata.section_title and any(priority_section in chunk.metadata.section_title.lower() 
                               for priority_section in ["summary", "conclusion", "abstract", "key", "important"]):
                        # Tag this chunk as high priority (used later by the retrieval engine for boosting)
                        chunk.metadata.priority = "high"
                    
                    self.manifest.add_chunk(chunk)
                
                all_chunks.extend(current_chunks_for_file)

                # Add to traditional model data if we have content for it
                if combined_text_for_doc_model and combined_text_for_doc_model.strip():
                    docs.append(combined_text_for_doc_model)
                    names.append(doc_name)
                    # Extract section titles from chunks for this document
                    section_titles_for_doc = list(set([
                        c.metadata.section_title for c in current_chunks_for_file 
                        if c.metadata.section_title and c.metadata.section_title.strip()
                    ]))
                    secs.append(section_titles_for_doc if section_titles_for_doc else [f"Content from {doc_name}"])
                else:
                    # If no combined_text but we have chunks (e.g. image processed but yielded no text for combined model)
                    # We might still want to count it as a "processed" file if chunks exist.
                    # However, `docs` list for Doc2Vec/TFIDF needs text.
                    print(f"  ⚠ No combined text for doc model from {doc_name} despite chunks existing")
            
            else: # No chunks were generated
                if combined_text_for_doc_model and combined_text_for_doc_model.strip(): 
                    # This case might happen if SmartParser extracted text but chunking failed or yielded nothing.
                    # Still add to `docs` for document-level models if text exists.
                    docs.append(combined_text_for_doc_model)
                    names.append(doc_name)
                    secs.append([f"Content from {doc_name}"]) # Default section
                    print(f"  ⚠ Content extracted for {doc_name} but no chunks generated. Added to document model.")
                else:
                    print(f"  ❌ No content extracted or chunks generated for {path}. Completely skipping.")

            self.progress.emit(i, len(files))

        # Save the manifest
        print(f"Saving manifest with {len(self.manifest.chunks)} chunks...")
        self.manifest.save()
        print(f"Processing complete: {len(all_chunks)} total chunks across {len(docs)} documents")

        model = None
        if docs: # Ensure docs is not empty before training
            tagged = [TaggedDocument(d.split(), [i]) for i, d in enumerate(docs)]
            model = Doc2Vec(vector_size=100, window=5, min_count=1, workers=4)
            model.build_vocab(tagged)
            model.train(tagged, epochs=60, total_examples=model.corpus_count)
            model.save(Config.MODEL_PATH)
        else:
            print("Warning: No documents processed to train Doc2Vec model.")

        vec = TfidfVectorizer(stop_words="english", max_features=Config.TOP_KEYWORDS)
        mat = vec.fit_transform(docs) if docs else None # Handle empty docs

        kw_map: Dict[str, List[int]] = {}
        if mat is not None and docs: # Check if mat is not None and docs is not empty
            feats = vec.get_feature_names_out()
            csc = mat.tocsc()
            for j, term in enumerate(feats):
                kw_map[term] = csc[:, j].nonzero()[0].tolist()

        self.complete.emit(docs, model, names, secs, (vec, mat), kw_map, all_chunks)


# --- LLM Client ---
class LLMClient:
    @staticmethod
    def call_model(name: str, prompt: str, offline=False) -> str:
        if offline:
            return "Offline mode – AI disabled."
        cfg = Config.MODEL_OPTS.get(name)
        try:
            r = requests.post(cfg["url"], json={"prompt": prompt, **cfg["params"]}, timeout=60)
            r.raise_for_status()
            return r.json()["choices"][0]["text"]
        except Exception as e:
            return f"[ERROR] {e}"


# --- Result Card UI ---
class ResultCard(QFrame):
    """A card widget to display search results."""

    def __init__(self, title: str, snippet: str, hybrid_score: float,
                 keywords: List[str], sections: List[str],
                 doc2vec_score: float, tfidf_score: float,
                 query_terms: List[str],
                 chunk_type: str = "text",
                 section_title: str = "",
                 image_path: str = "") -> None:
        """Initialize the result card.

        Args:
            title: The title of the result
            snippet: The text snippet to display
            hybrid_score: Combined relevance score
            keywords: List of relevant keywords
            sections: List of document sections
            doc2vec_score: Doc2Vec similarity score
            tfidf_score: TF-IDF similarity score
            query_terms: List of terms from the query
            chunk_type: Type of chunk (text, ocr, caption, etc.)
            section_title: Full section title (if available)
            image_path: Path to the image (for image chunks)
        """
        super().__init__()
        self.setStyleSheet(
            "background:#1E1E1E;border:1px solid #2D2D2D;border-radius:8px;margin:4px;padding:4px;"
        )
        self.setMinimumHeight(200)  # Set minimum height for each card
        
        ly = QVBoxLayout(self)
        ly.setContentsMargins(12, 12, 12, 12)  # Larger margins
        ly.setSpacing(8)  # More space between elements

        # Main header with title and score
        title_layout = QHBoxLayout()
        
        hdr = QLabel(f"<b>{title}</b>")
        hdr.setFont(QFont(None, 13))  # Larger font
        hdr.setStyleSheet("color: #00FFAA;")
        title_layout.addWidget(hdr)
        
        score_lbl = QLabel(f"<span style='color:#FF9500'>({hybrid_score:.2f})</span>")
        score_lbl.setFont(QFont(None, 12))
        title_layout.addWidget(score_lbl)
        
        title_layout.addStretch()
        ly.addLayout(title_layout)

        # Display section title if available
        if section_title:
            section_lbl = QLabel(f"<i>Section:</i> <b>{section_title}</b>")
            section_lbl.setStyleSheet("color: #00D0FF; font-size: 11pt;")
            ly.addWidget(section_lbl)

        # Display chunk type if it's not plain text
        if chunk_type and chunk_type != "text":
            chunk_type_lbl = QLabel(f"<i>Type:</i> <b>{chunk_type.upper()}</b>")
            chunk_type_lbl.setStyleSheet("color: #FFA500; font-size: 11pt;")
            ly.addWidget(chunk_type_lbl)

        # Display the image if this is an image chunk and we have a path
        if image_path and (chunk_type == "ocr" or chunk_type == "caption"):
            try:
                # Create a label to hold the image
                img_label = QLabel()
                img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                
                # Load and resize the image
                pixmap = QPixmap(image_path)
                if not pixmap.isNull():
                    # Resize to fit within the card (max width 400px, max height 300px)
                    if pixmap.width() > 400 or pixmap.height() > 300:
                        pixmap = pixmap.scaled(400, 300, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                    img_label.setPixmap(pixmap)
                    img_label.setToolTip(f"Image: {os.path.basename(image_path)}")
                    
                    # Add the image to the layout
                    ly.addWidget(img_label)
                    
                    # Add a separator
                    separator = QFrame()
                    separator.setFrameShape(QFrame.Shape.HLine)
                    separator.setFrameShadow(QFrame.Shadow.Sunken)
                    separator.setStyleSheet("background-color: #333333;")
                    ly.addWidget(separator)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")

        # Keywords section with improved styling
        if keywords:
            keywords_layout = QHBoxLayout()
            keywords_layout.addWidget(QLabel("<i>Keywords:</i>"))
            
            # Create pills for keywords
            keyword_widget = QWidget()
            keyword_layout = QHBoxLayout(keyword_widget)
            keyword_layout.setContentsMargins(0, 0, 0, 0)
            keyword_layout.setSpacing(6)
            
            for kw in keywords[:5]:
                kw_pill = QPushButton(str(kw))
                kw_pill.setStyleSheet("""
                    QPushButton {
                        background-color: #333333;
                        color: #00FFAA;
                        border-radius: 10px;
                        padding: 3px 8px;
                        font-size: 10px;
                        border: none;
                    }
                    QPushButton:hover {
                        background-color: #444444;
                    }
                """)
                kw_pill.setCursor(Qt.CursorShape.PointingHandCursor)
                keyword_layout.addWidget(kw_pill)
            
            keyword_layout.addStretch()
            keywords_layout.addWidget(keyword_widget)
            ly.addLayout(keywords_layout)
        
        # Scores in smaller text
        scores_lbl = QLabel(f"Doc2Vec: {doc2vec_score:.2f}, TF‑IDF: {tfidf_score:.2f}")
        scores_lbl.setStyleSheet("color: #888888; font-size: 9pt;")
        ly.addWidget(scores_lbl)

        # Highlight query terms in the snippet with a more visually appealing container
        snippet_container = QFrame()
        snippet_container.setStyleSheet("background-color: #252525; border-radius: 6px; padding: 8px;")
        snippet_layout = QVBoxLayout(snippet_container)
        snippet_layout.setContentsMargins(8, 8, 8, 8)
        
        html = snippet
        for t in set(query_terms):
            if t.strip():  # Skip empty terms
                html = re.sub(
                    rf"\b{re.escape(t)}\b",
                    f"<span style='background:#00FFAA;color:#000;padding:0px 2px;border-radius:2px;'>{t}</span>",
                    html,
                    flags=re.IGNORECASE
                )
        
        lbl = QLabel(html)
        lbl.setWordWrap(True)
        lbl.setStyleSheet("font-size: 11pt; line-height: 1.4;")  # Larger text with better line height
        lbl.setTextFormat(Qt.TextFormat.RichText)
        snippet_layout.addWidget(lbl)
        
        ly.addWidget(snippet_container)


# --- Style Editor ---
class StyleEditor(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent

        # Initialize all instance attributes
        # Gradient attributes
        self.gradient_type = None
        self.gradient_stops = []

        # Font attributes
        self.font_family = None
        self.font_size = None
        self.font_weight = None
        self.font_style = None
        self.font_underline = None
        self.font_strikeout = None
        self.font_kerning = None

        # Animation attributes
        self.anim_type = None
        self.anim_duration = None
        self.anim_easing = None
        self.anim_delay = None
        self.anim_loop = None
        self.anim_reverse = None

        # Neon effect attributes
        self.neon_enabled = None
        self.neon_color = None
        self.neon_blur = None
        self.neon_spread = None
        self.neon_pulse = None
        self.neon_pulse_speed = None

        # Layout attributes
        self.margin = None
        self.spacing = None
        self.alignment = None
        self.padding = None
        self.border_radius = None
        self.border_width = None

        # Other UI elements
        self.preset_list = None
        self.color_pickers = {}

        # Initialize UI
        self.setup_ui()

    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)

        # Create tabs for different styling categories
        tabs = QTabWidget()

        # Colors tab
        colors_tab = QWidget()
        colors_layout = QVBoxLayout(colors_tab)

        # Basic colors
        basic_colors = QGroupBox("Basic Colors")
        basic_layout = QGridLayout()
        colors = {
            "Background": "#121212",
            "Text": "#EEE",
            "Button": "#1F1F1F",
            "Border": "#2D2D2D",
            "Accent": "#00FFAA",
            "Highlight": "#FF4500",
            "Selection": "#00FFAA",
            "Link": "#00FFFF"
        }

        row = 0
        for name, default in colors.items():
            label = QLabel(name)
            picker = QPushButton()
            picker.setStyleSheet(f"background-color: {default}; border: none;")
            picker.clicked.connect(lambda checked, p=picker, n=name: self.pick_color(p, n))
            self.color_pickers[name] = picker

            basic_layout.addWidget(label, row // 2, (row % 2) * 2)
            basic_layout.addWidget(picker, row // 2, (row % 2) * 2 + 1)
            row += 1

        basic_colors.setLayout(basic_layout)
        colors_layout.addWidget(basic_colors)

        # Gradient editor
        gradient_group = QGroupBox("Gradient Editor")
        gradient_layout = QVBoxLayout()

        self.gradient_type = QComboBox()
        self.gradient_type.addItems(["Linear", "Radial", "Conical"])
        gradient_layout.addWidget(self.gradient_type)

        # Container for dynamic gradient stops
        self.gradient_stop_layout = QVBoxLayout()
        gradient_layout.addLayout(self.gradient_stop_layout)

        # Button to add new gradient stops
        add_stop_btn = QPushButton("Add Gradient Stop")
        add_stop_btn.clicked.connect(lambda: self.add_gradient_stop(self.gradient_stop_layout))
        gradient_layout.addWidget(add_stop_btn)

        gradient_group.setLayout(gradient_layout)
        colors_layout.addWidget(gradient_group)

        colors_tab.setLayout(colors_layout)
        tabs.addTab(colors_tab, "Colors")

        # Fonts tab
        fonts_tab = QWidget()
        fonts_layout = QVBoxLayout(fonts_tab)

        # Font family
        font_family_group = QGroupBox("Font Family")
        font_family_layout = QVBoxLayout()
        self.font_family = QFontComboBox()
        font_family_layout.addWidget(self.font_family)
        font_family_group.setLayout(font_family_layout)
        fonts_layout.addWidget(font_family_group)

        # Font properties
        font_props_group = QGroupBox("Font Properties")
        font_props_layout = QGridLayout()

        self.font_size = QSpinBox()
        self.font_size.setRange(8, 72)
        self.font_size.setValue(12)

        self.font_weight = QComboBox()
        self.font_weight.addItems(["Normal", "Bold", "Light", "Black"])

        self.font_style = QComboBox()
        self.font_style.addItems(["Normal", "Italic", "Oblique"])

        self.font_underline = QCheckBox("Underline")
        self.font_strikeout = QCheckBox("Strikeout")
        self.font_kerning = QCheckBox("Kerning")

        font_props_layout.addWidget(QLabel("Size:"), 0, 0)
        font_props_layout.addWidget(self.font_size, 0, 1)
        font_props_layout.addWidget(QLabel("Weight:"), 1, 0)
        font_props_layout.addWidget(self.font_weight, 1, 1)
        font_props_layout.addWidget(QLabel("Style:"), 2, 0)
        font_props_layout.addWidget(self.font_style, 2, 1)
        font_props_layout.addWidget(self.font_underline, 3, 0, 1, 2)
        font_props_layout.addWidget(self.font_strikeout, 4, 0, 1, 2)
        font_props_layout.addWidget(self.font_kerning, 5, 0, 1, 2)

        font_props_group.setLayout(font_props_layout)
        fonts_layout.addWidget(font_props_group)

        fonts_tab.setLayout(fonts_layout)
        tabs.addTab(fonts_tab, "Fonts")

        # Animations tab
        anim_tab = QWidget()
        anim_layout = QVBoxLayout(anim_tab)

        # Animation properties
        anim_props_group = QGroupBox("Animation Properties")
        anim_props_layout = QGridLayout()

        self.anim_type = QComboBox()
        self.anim_type.addItems([
            "Fade", "Slide", "Scale", "Rotation",
            "Bounce", "Elastic", "Back"
        ])

        self.anim_duration = QSpinBox()
        self.anim_duration.setRange(100, 5000)
        self.anim_duration.setValue(1000)

        self.anim_easing = QComboBox()
        self.anim_easing.addItems([
            "Linear", "InQuad", "OutQuad", "InOutQuad",
            "InCubic", "OutCubic", "InOutCubic"
        ])

        self.anim_delay = QSpinBox()
        self.anim_delay.setRange(0, 2000)
        self.anim_delay.setValue(0)

        self.anim_loop = QCheckBox("Loop")
        self.anim_reverse = QCheckBox("Reverse")

        anim_props_layout.addWidget(QLabel("Type:"), 0, 0)
        anim_props_layout.addWidget(self.anim_type, 0, 1)
        anim_props_layout.addWidget(QLabel("Duration:"), 1, 0)
        anim_props_layout.addWidget(self.anim_duration, 1, 1)
        anim_props_layout.addWidget(QLabel("Easing:"), 2, 0)
        anim_props_layout.addWidget(self.anim_easing, 2, 1)
        anim_props_layout.addWidget(QLabel("Delay:"), 3, 0)
        anim_props_layout.addWidget(self.anim_delay, 3, 1)
        anim_props_layout.addWidget(self.anim_loop, 4, 0, 1, 2)
        anim_props_layout.addWidget(self.anim_reverse, 5, 0, 1, 2)

        anim_props_group.setLayout(anim_props_layout)
        anim_layout.addWidget(anim_props_group)

        # Neon effects
        neon_group = QGroupBox("Neon Effects")
        neon_layout = QGridLayout()

        self.neon_enabled = QCheckBox("Enable Neon")
        self.neon_color = QPushButton()
        self.neon_color.setStyleSheet("background-color: #00FFAA; border: none;")
        self.neon_color.clicked.connect(lambda: self.pick_color(self.neon_color, "Neon"))

        self.neon_blur = QSpinBox()
        self.neon_blur.setRange(0, 50)
        self.neon_blur.setValue(10)

        self.neon_spread = QSpinBox()
        self.neon_spread.setRange(0, 50)
        self.neon_spread.setValue(5)

        self.neon_pulse = QCheckBox("Pulse Effect")
        self.neon_pulse_speed = QSpinBox()
        self.neon_pulse_speed.setRange(1, 10)
        self.neon_pulse_speed.setValue(3)

        neon_layout.addWidget(self.neon_enabled, 0, 0, 1, 2)
        neon_layout.addWidget(QLabel("Color:"), 1, 0)
        neon_layout.addWidget(self.neon_color, 1, 1)
        neon_layout.addWidget(QLabel("Blur:"), 2, 0)
        neon_layout.addWidget(self.neon_blur, 2, 1)
        neon_layout.addWidget(QLabel("Spread:"), 3, 0)
        neon_layout.addWidget(self.neon_spread, 3, 1)
        neon_layout.addWidget(self.neon_pulse, 4, 0, 1, 2)
        neon_layout.addWidget(QLabel("Pulse Speed:"), 5, 0)
        neon_layout.addWidget(self.neon_pulse_speed, 5, 1)

        neon_group.setLayout(neon_layout)
        anim_layout.addWidget(neon_group)

        anim_tab.setLayout(anim_layout)
        tabs.addTab(anim_tab, "Effects")

        # Layout tab
        layout_tab = QWidget()
        layout_layout = QVBoxLayout(layout_tab)

        # Layout properties
        layout_props_group = QGroupBox("Layout Properties")
        layout_props_layout = QGridLayout()

        self.margin = QSpinBox()
        self.margin.setRange(0, 50)
        self.margin.setValue(8)

        self.spacing = QSpinBox()
        self.spacing.setRange(0, 50)
        self.spacing.setValue(6)

        self.alignment = QComboBox()
        self.alignment.addItems(["Left", "Center", "Right"])

        self.padding = QSpinBox()
        self.padding.setRange(0, 50)
        self.padding.setValue(8)

        self.border_radius = QSpinBox()
        self.border_radius.setRange(0, 50)
        self.border_radius.setValue(6)

        self.border_width = QSpinBox()
        self.border_width.setRange(0, 10)
        self.border_width.setValue(1)

        layout_props_layout.addWidget(QLabel("Margin:"), 0, 0)
        layout_props_layout.addWidget(self.margin, 0, 1)
        layout_props_layout.addWidget(QLabel("Spacing:"), 1, 0)
        layout_props_layout.addWidget(self.spacing, 1, 1)
        layout_props_layout.addWidget(QLabel("Alignment:"), 2, 0)
        layout_props_layout.addWidget(self.alignment, 2, 1)
        layout_props_layout.addWidget(QLabel("Padding:"), 3, 0)
        layout_props_layout.addWidget(self.padding, 3, 1)
        layout_props_layout.addWidget(QLabel("Border Radius:"), 4, 0)
        layout_props_layout.addWidget(self.border_radius, 4, 1)
        layout_props_layout.addWidget(QLabel("Border Width:"), 5, 0)
        layout_props_layout.addWidget(self.border_width, 5, 1)

        layout_props_group.setLayout(layout_props_layout)
        layout_layout.addWidget(layout_props_group)

        layout_tab.setLayout(layout_layout)
        tabs.addTab(layout_tab, "Layout")

        layout.addWidget(tabs)

        # Presets
        presets_group = QGroupBox("Style Presets")
        presets_layout = QHBoxLayout()

        self.preset_list = QListWidget()
        self.load_presets()
        presets_layout.addWidget(self.preset_list)

        preset_buttons = QVBoxLayout()
        save_btn = QPushButton("Save Preset")
        save_btn.clicked.connect(self.save_preset)
        load_btn = QPushButton("Load Preset")
        load_btn.clicked.connect(self.load_preset)
        delete_btn = QPushButton("Delete Preset")
        delete_btn.clicked.connect(self.delete_preset)

        preset_buttons.addWidget(save_btn)
        preset_buttons.addWidget(load_btn)
        preset_buttons.addWidget(delete_btn)
        preset_buttons.addStretch()

        presets_layout.addLayout(preset_buttons)
        presets_group.setLayout(presets_layout)
        layout.addWidget(presets_group)

        # Apply button
        apply_btn = QPushButton("Apply Style")
        apply_btn.clicked.connect(self.apply_style)
        layout.addWidget(apply_btn)

    def _create_preset_dict(self) -> dict:
        """Create a dictionary of current preset values."""
        return {
            'colors': {k: v.styleSheet().split(":")[1].strip(";")
                       for k, v in self.color_pickers.items()},
            'font': {
                'family': self.font_family.currentFont().family(),
                'size': self.font_size.value(),
                'weight': self.font_weight.currentText(),
                'style': self.font_style.currentText(),
                'underline': self.font_underline.isChecked(),
                'strikeout': self.font_strikeout.isChecked(),
                'kerning': self.font_kerning.isChecked()
            },
            'animations': {
                'type': self.anim_type.currentText(),
                'duration': self.anim_duration.value(),
                'easing': self.anim_easing.currentText(),
                'delay': self.anim_delay.value(),
                'loop': self.anim_loop.isChecked(),
                'reverse': self.anim_reverse.isChecked()
            },
            'neon': {
                'enabled': self.neon_enabled.isChecked(),
                'color': self.neon_color.styleSheet().split(":")[1].strip(";"),
                'blur': self.neon_blur.value(),
                'spread': self.neon_spread.value(),
                'pulse': self.neon_pulse.isChecked(),
                'pulse_speed': self.neon_pulse_speed.value()
            },
            'layout': {
                'margin': self.margin.value(),
                'spacing': self.spacing.value(),
                'alignment': self.alignment.currentText(),
                'padding': self.padding.value(),
                'border_radius': self.border_radius.value(),
                'border_width': self.border_width.value()
            }
        }

    def _apply_preset_dict(self, preset: dict) -> None:
        """Apply a preset dictionary to the UI elements."""
        if not preset:
            return

        # Load colors
        for k, v in preset['colors'].items():
            if k in self.color_pickers:
                self.color_pickers[k].setStyleSheet(f"background-color: {v}; border: none;")

        # Load font
        self.font_family.setCurrentFont(QFont(preset['font']['family']))
        self.font_size.setValue(preset['font']['size'])
        self.font_weight.setCurrentText(preset['font']['weight'])
        self.font_style.setCurrentText(preset['font']['style'])
        self.font_underline.setChecked(preset['font']['underline'])
        self.font_strikeout.setChecked(preset['font']['strikeout'])
        self.font_kerning.setChecked(preset['font']['kerning'])

        # Load animations
        self.anim_type.setCurrentText(preset['animations']['type'])
        self.anim_duration.setValue(preset['animations']['duration'])
        self.anim_easing.setCurrentText(preset['animations']['easing'])
        self.anim_delay.setValue(preset['animations']['delay'])
        self.anim_loop.setChecked(preset['animations']['loop'])
        self.anim_reverse.setChecked(preset['animations']['reverse'])

        # Load neon
        self.neon_enabled.setChecked(preset['neon']['enabled'])
        self.neon_color.setStyleSheet(f"background-color: {preset['neon']['color']}; border: none;")
        self.neon_blur.setValue(preset['neon']['blur'])
        self.neon_spread.setValue(preset['neon']['spread'])
        self.neon_pulse.setChecked(preset['neon']['pulse'])
        self.neon_pulse_speed.setValue(preset['neon']['pulse_speed'])

        # Load layout
        self.margin.setValue(preset['layout']['margin'])
        self.spacing.setValue(preset['layout']['spacing'])
        self.alignment.setCurrentText(preset['layout']['alignment'])
        self.padding.setValue(preset['layout']['padding'])
        self.border_radius.setValue(preset['layout']['border_radius'])
        self.border_width.setValue(preset['layout']['border_width'])

    def add_gradient_stop(self, layout):
        """Add a new gradient stop to the gradient editor."""
        stop_widget = QWidget()
        stop_layout = QHBoxLayout(stop_widget)

        color_btn = QPushButton()
        color_btn.setStyleSheet("background-color: #FFFFFF; border: none;")
        color_btn.clicked.connect(lambda: self.pick_color(color_btn, "Gradient Stop"))

        pos_spin = QDoubleSpinBox()
        pos_spin.setRange(0.0, 1.0)
        pos_spin.setSingleStep(0.1)
        pos_spin.setValue(0.0)

        stop_layout.addWidget(color_btn)
        stop_layout.addWidget(pos_spin)
        stop_layout.addWidget(QPushButton("×", clicked=lambda: self.remove_gradient_stop(stop_widget)))

        layout.addWidget(stop_widget)
        self.gradient_stops.append((color_btn, pos_spin))

    def remove_gradient_stop(self, widget):
        """Remove a gradient stop from the gradient editor."""
        for stop in self.gradient_stops[:]:
            if stop[0].parent() == widget:
                self.gradient_stops.remove(stop)
                # Clean up the widgets properly
                stop[0].deleteLater()
                stop[1].deleteLater()
        widget.deleteLater()
        # Force update the gradient preview
        self.update_gradient_preview()

    def pick_color(self, button: QPushButton, name: str) -> None:
        """Pick a color for a button."""
        color = QColorDialog.getColor(QColor(button.styleSheet().split(":")[1].strip(";")))
        if color.isValid():
            button.setStyleSheet(f"background-color: {color.name()}; border: none;")

    def save_preset(self):
        """Save the current style settings as a preset."""
        name, ok = QInputDialog.getText(self, "Save Preset", "Enter preset name:")
        if ok and name:
            settings = QSettings("RAGApp", "StylePresets")
            settings.setValue(f"presets/{name}", self._create_preset_dict())
            self.load_presets()

    def load_preset(self):
        """Load a selected preset."""
        if self.preset_list.currentItem():
            name = self.preset_list.currentItem().text()
            settings = QSettings("RAGApp", "StylePresets")
            preset = settings.value(f"presets/{name}")
            if preset:
                self._apply_preset_dict(preset)

    def delete_preset(self):
        """Delete a selected preset."""
        if self.preset_list.currentItem():
            name = self.preset_list.currentItem().text()
            settings = QSettings("RAGApp", "StylePresets")
            settings.remove(f"presets/{name}")
            self.load_presets()

    def load_presets(self):
        """Load all available presets into the preset list."""
        self.preset_list.clear()
        settings = QSettings("RAGApp", "StylePresets")
        presets = settings.value("presets", {})
        if presets:
            for name in presets.keys():
                self.preset_list.addItem(name)

    def apply_style(self):
        """Apply the current style settings to the application."""
        # Create comprehensive style sheet
        style = f"""
            QWidget {{ 
                background:{self.color_pickers["Background"].styleSheet().split(":")[1].strip(";")}; 
                color:{self.color_pickers["Text"].styleSheet().split(":")[1].strip(";")}; 
                font-family:'{self.font_family.currentFont().family()}';
                font-size:{self.font_size.value()}px;
                font-weight:{self.font_weight.currentText().lower()};
                font-style:{self.font_style.currentText().lower()};
                {f"text-decoration: underline;" if self.font_underline.isChecked() else ""}
                {f"text-decoration: line-through;" if self.font_strikeout.isChecked() else ""}
                {f"font-kerning: normal;" if self.font_kerning.isChecked() else ""}
            }}
            QPushButton {{ 
                background:{self.color_pickers["Button"].styleSheet().split(":")[1].strip(";")}; 
                border:{self.border_width.value()}px solid {self.color_pickers["Border"].styleSheet().split(":")[1].strip(";")}; 
                border-radius:{self.border_radius.value()}px; 
                padding:{self.padding.value()}px;
                min-width: 100px;
            }}
            QPushButton:hover {{ 
                border-color:{self.color_pickers["Accent"].styleSheet().split(":")[1].strip(";")};
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0, 
                    stop:0 {self.color_pickers["Highlight"].styleSheet().split(":")[1].strip(";")},
                    stop:1 #FFD700);
                margin: -2px;  /* Simulate scale effect without transform */
                border-width: 2px;
            }}
            QPushButton#askBtn {{
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #FF4500, stop: 1 #FFD700);
                color:#000; 
                font-weight:bold; 
                font-size:16px;
                border: none;
            }}
            QPushButton#askBtn:hover {{
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0, 
                    stop:0 {self.color_pickers["Highlight"].styleSheet().split(":")[1].strip(";")},
                    stop:1 #FFD700);
                transform: scale(1.05);
                transition: transform 0.2s ease;
            }}
        """
        QApplication.instance().setStyleSheet(style)

        # Apply animations
        if self.anim_type.currentText() == "Fade":
            # Add fade animation
            pass
        elif self.anim_type.currentText() == "Slide":
            # Add slide animation
            pass
        # Add more animation types...

        # Apply neon effects via Qt graphics effect
        from PyQt6.QtWidgets import QGraphicsDropShadowEffect
        from PyQt6.QtGui import QColor

        if self.neon_enabled.isChecked():
            # Create the neon glow effect
            glow = QGraphicsDropShadowEffect()
            glow.setOffset(0, 0)
            glow.setBlurRadius(self.neon_blur.value())
            glow.setColor(QColor(self.neon_color.styleSheet().split(":")[1].strip(";")))
            # Apply to all buttons in the application
            for widget in QApplication.instance().allWidgets():
                if isinstance(widget, QPushButton):
                    widget.setGraphicsEffect(glow)
        else:
            # Remove any existing glow effects
            for widget in QApplication.instance().allWidgets():
                if isinstance(widget, QPushButton):
                    widget.setGraphicsEffect(None)

    def update_gradient_preview(self):
        """Update the gradient preview after changes."""
        if not self.gradient_stops:
            return

        # Create gradient string based on type
        grad_type = self.gradient_type.currentText().lower()
        stops = []
        for btn, pos in self.gradient_stops:
            color = btn.styleSheet().split(":")[1].strip(";")
            stops.append(f"stop: {pos.value()} {color}")

        if grad_type == "linear":
            grad_str = f"qlineargradient(x1:0,y1:0,x2:1,y2:0, {', '.join(stops)})"
        elif grad_type == "radial":
            grad_str = f"qradialgradient(cx:0.5,cy:0.5,radius:0.5,fx:0.5,fy:0.5, {', '.join(stops)})"
        else:  # conical
            grad_str = f"qconicalgradient(cx:0.5,cy:0.5,angle:0, {', '.join(stops)})"

        # Apply to preview if it exists
        if hasattr(self, 'gradient_preview'):
            self.gradient_preview.setStyleSheet(f"background: {grad_str};")


# --- Main Application Window ---
class RAGApp(GradientWindow):
    def __init__(self):
        super().__init__()
        self.docs = []
        self.names = []
        self.sections = []
        self.keywords_map = {}
        self.model = None # Doc2Vec model
        self.tfidf_vec = None
        self.tfidf_mat = None
        self.history = []
        self.chunks: List[DocumentChunk] = []
        self.current_manifest: Optional[ChunkManifest] = None # To hold the latest manifest

        try:
            self.vector_index_manager = VectorIndexManager(index_dir="vector_index_chunks")
        except ImportError as e:
            print(f"CRITICAL: Could not initialize VectorIndexManager: {e}")
            QMessageBox.critical(self, "Dependency Error", f"Could not load VectorIndexManager: {e}. Ensure FAISS and sentence-transformers are installed.")
            self.vector_index_manager = None
            # Potentially exit or disable search functionalities

        # Initialize HydroidRetrievalEngine
        # It needs a live manifest. We'll give it the path, and it can load/reload it.
        # Or, RAGApp loads it and passes the object. For now, let's pass the object and update it.
        if self.vector_index_manager: # Only init engine if VIM is available
            # Create an initial empty manifest or load if exists, to pass to engine
            # This manifest will be updated in _on_indexed
            self.current_manifest = ChunkManifest(Config.CHUNK_MANIFEST_PATH) 
            try:
                self.current_manifest.load() # Try to load existing manifest
            except FileNotFoundError:
                print("No existing chunk manifest found. A new one will be created after processing.")
            except Exception as e_manifest_load:
                print(f"Error loading initial manifest: {e_manifest_load}. Proceeding with empty/new.")
                self.current_manifest.chunks.clear() # Ensure it's empty if load failed badly

            self.hydroid_engine = HydroidRetrievalEngine(
                vector_index_manager=self.vector_index_manager,
                chunk_manifest=self.current_manifest, # Pass the manifest object
                debug_dir="./retrieval_debug" 
            )
        else:
            self.hydroid_engine = None
            QMessageBox.warning(self, "Engine Warning", "Hydroid Retrieval Engine could not be initialized due to missing Vector Index Manager.")

        self._build_ui()
        # ... rest of __init__
        # Remove self.bm25_ranker_chunks = SimpleBM25() - engine handles its own BM25

    def _build_ui(self):
        set_futuristic_style(QApplication.instance())

        # Replace 'Segoe UI' with system default sans-serif font
        default_font = QApplication.font().family()
        QApplication.setFont(QFont(default_font))

        tabs = QTabWidget()
        tabs.setTabPosition(QTabWidget.TabPosition.North)

        # --- Tab 1: Combined RAG + AI Search ---
        w1 = QWidget()
        L1 = QVBoxLayout(w1)

        # Top query section
        query_section = QHBoxLayout()
        self.query = QTextEdit()
        self.query.setPlaceholderText("Enter query…")
        self.query.setMaximumHeight(100)
        query_section.addWidget(self.query, 3)

        ctrl = QVBoxLayout()
        self.ai_cb = QCheckBox("Enable AI")
        self.ai_cb.setChecked(True)  # Enable AI by default
        ctrl.addWidget(self.ai_cb)
        self.model_cb = QComboBox()
        self.model_cb.addItems(Config.MODEL_OPTS.keys())
        ctrl.addWidget(self.model_cb)

        self.ask_btn = QPushButton("🔥 Ask RAG + AI 🔥")
        self.ask_btn.setObjectName("askBtn")
        self.ask_btn.clicked.connect(self._do_query)
        ctrl.addWidget(self.ask_btn)
        ctrl.addStretch()
        query_section.addLayout(ctrl)
        L1.addLayout(query_section)

        # Main content area
        content_split = QSplitter(Qt.Orientation.Horizontal)

        # Left sidebar for document navigation
        left = QVBoxLayout()
        btn_folder = QPushButton("📂 Select Folder")
        btn_folder.clicked.connect(self._select_folder)
        btn_refresh = QPushButton("🔄 Process Folder")
        btn_refresh.clicked.connect(self._process)
        left.addWidget(btn_folder)
        left.addWidget(btn_refresh)
        
        # Add status message label
        self.status_msg = QLabel("Ready to process documents")
        left.addWidget(self.status_msg)
        
        left.addSpacing(12)

        left.addWidget(QLabel("📂 Documents"))
        self.doc_list = QListWidget()
        self.doc_list.itemClicked.connect(self._on_doc)
        left.addWidget(self.doc_list, 1)

        left.addWidget(QLabel("🧠 Sections"))
        self.sec_tree = QTreeWidget()
        self.sec_tree.setHeaderHidden(True)
        self.sec_tree.itemClicked.connect(self._on_section)
        left.addWidget(self.sec_tree, 1)

        left.addWidget(QLabel("🔑 Keywords"))
        self.kw_combo = QComboBox()
        self.kw_combo.currentTextChanged.connect(self._on_kw)
        left.addWidget(self.kw_combo)

        left_box = QGroupBox()
        left_box.setLayout(left)
        content_split.addWidget(left_box)

        # Right side with results
        right_split = QSplitter(Qt.Orientation.Vertical)

        # RAG Results section
        rag_group = QGroupBox("📖 RAG Results")
        rag_layout = QVBoxLayout(rag_group)
        self.r_scroll = QScrollArea()
        cont1 = QWidget()
        self.r_layout = QVBoxLayout(cont1)
        self.r_layout.setSpacing(16)  # Add more spacing between cards
        self.r_scroll.setWidgetResizable(True)
        self.r_scroll.setWidget(cont1)
        self.r_scroll.setMinimumHeight(500)  # Set minimum height for results
        rag_layout.addWidget(self.r_scroll)

        # Show Parsed button
        self.btn_parsed = QPushButton("Show Parsed Details")
        self.btn_parsed.setEnabled(False)
        self.btn_parsed.clicked.connect(self._show_parsed_details)
        rag_layout.addWidget(self.btn_parsed, alignment=Qt.AlignmentFlag.AlignRight)
        right_split.addWidget(rag_group)

        # AI Response section
        ai_group = QGroupBox("🤖 AI Response")
        ai_layout = QVBoxLayout(ai_group)
        self.ai_scroll = QScrollArea()
        cont2 = QWidget()
        self.ai_layout = QVBoxLayout(cont2)
        self.ai_scroll.setWidgetResizable(True)
        self.ai_scroll.setWidget(cont2)
        ai_layout.addWidget(self.ai_scroll)
        right_split.addWidget(ai_group)

        # Set initial sizes for the split
        right_split.setSizes([600, 200])  # More space for RAG results, less for AI responses
        content_split.addWidget(right_split)
        content_split.setSizes([250, 950])  # Less space for nav, more for content

        L1.addWidget(content_split)
        tabs.addTab(w1, "🔍 RAG + AI Search")

        # --- Tab 2: Style Editor ---
        w2 = QWidget()
        L2 = QVBoxLayout(w2)
        style_editor = StyleEditor(self)
        L2.addWidget(style_editor)
        tabs.addTab(w2, "🎨 Style Editor")

        # Main layout
        main = QVBoxLayout(self)
        main.addWidget(tabs)
        self.progress = QProgressBar()
        main.addWidget(self.progress)

    def _select_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Select Folder")
        if d:
            Config.FOLDER_PATH = d
            self._process()

    def _load_cache(self):
        """Try to load indexed data from cache."""
        try:
            if os.path.exists(Config.CACHE_PATH):
                with open(Config.CACHE_PATH, 'rb') as f:
                    try:
                        # Try to load with chunks (new format)
                        data = pickle.load(f)
                        if len(data) == 7:  # Complete data with chunks
                            docs, model, names, secs, tfidf, km, chunks = data
                            self.chunks = chunks
                        else:  # Old format without chunks
                            docs, model, names, secs, tfidf, km = data
                            self.chunks = []  # Initialize empty chunks list
                            
                        # Set up the rest of the data
                        self._on_indexed(docs, model, names, secs, tfidf, km, self.chunks)
                        return True
                    except ValueError:
                        # Fall back to old cache format for compatibility
                        f.seek(0)
                        data = pickle.load(f)
                        if isinstance(data, dict):
                            self.docs = data.get("docs", [])
                            self.names = data.get("names", [])
                            self.sections = data.get("sections", [])
                            self.keyword_map = data.get("keywords_map", {})
                            self.chunks = []  # Initialize empty chunks list
                            
                            # Load model separately
                            if os.path.exists(Config.MODEL_PATH):
                                self.model = Doc2Vec.load(Config.MODEL_PATH)
                            
                            # Try to reconstruct TF-IDF data
                            vec = TfidfVectorizer(stop_words="english", max_features=Config.TOP_KEYWORDS)
                            mat = vec.fit_transform(self.docs) if self.docs else None
                            self.rhino_tfidf_vec, self.rhino_tfidf_mat = vec, mat
                            
                            # Set up the UI
                            self.doc_list.clear()
                            self.doc_list.addItems(self.names)
                            
                            self.sec_tree.clear()
                            for i, name in enumerate(self.names):
                                p = QTreeWidgetItem(self.sec_tree, [name])
                                if i < len(self.sections):
                                    for s in self.sections[i]:
                                        QTreeWidgetItem(p, [s])
                            
                            self.kw_combo.clear()
                            self.kw_combo.addItems(sorted(self.keyword_map.keys()))
                            
                            self.query.setEnabled(True)
                            self.ask_btn.setEnabled(True)
                            return True
        except Exception as e:
            print(f"Error loading cache: {e}")
        
        return False

    def _process(self):
        if not os.path.isdir(Config.FOLDER_PATH):
            QMessageBox.warning(self, "Error", "Invalid folder")
            return

        # start indexing
        self.processor = DocumentProcessor(Config.FOLDER_PATH)
        self.processor.finished.connect(self.processor.deleteLater)  # auto‑cleanup
        self.processor.progress.connect(lambda c, t: self.progress.setValue(int(c / t * 100)))
        self.processor.complete.connect(self._on_indexed)
        self.processor.start()

    def _on_indexed(self, docs, model, names, secs, tfidf, km, processed_chunks):
        self.docs, self.model = docs, model
        self.names, self.sections = names, secs
        self.rhino_tfidf_vec, self.rhino_tfidf_mat = tfidf
        self.keyword_map = km
        self.chunks = processed_chunks
        print(f"RAGApp._on_indexed: Received {len(self.chunks)} processed chunks.")

        # The DocumentProcessor (self.processor) should have saved its manifest.
        # We need to reload it here to ensure RAGApp has the latest version for the HydroidEngine.
        if self.current_manifest is not None: # self.current_manifest was initialized in __init__
            try:
                self.current_manifest.load() # Reload from disk where DocumentProcessor saved it
                print(f"RAGApp._on_indexed: Reloaded manifest with {len(self.current_manifest.chunks)} chunks for HydroidEngine.")
            except Exception as e:
                print(f"RAGApp._on_indexed: Error reloading manifest: {e}. Engine might use stale manifest data.")
                # If reload fails, current_manifest still holds what it had (potentially empty or from last successful load).
        else:
            # This case should ideally not happen if __init__ sets it up.
            print("RAGApp._on_indexed: self.current_manifest is None. Cannot provide to engine.")

        if self.vector_index_manager and self.chunks and self.current_manifest:
            print(f"RAGApp._on_indexed: Updating VectorIndexManager with manifest containing {len(self.current_manifest.chunks)} entries...")
            # ---- BEGIN DEBUG PRINT ----
            if self.current_manifest.chunks:
                first_item_type = type(list(self.current_manifest.chunks.values())[0])
                print(f"RAGApp._on_indexed: Type of first item in self.current_manifest.chunks.values(): {first_item_type}")
                if isinstance(list(self.current_manifest.chunks.values())[0], str):
                    print(f"RAGApp._on_indexed: First item is a STRING: {list(self.current_manifest.chunks.values())[0][:100]}...")
                elif isinstance(list(self.current_manifest.chunks.values())[0], DocumentChunk):
                    print(f"RAGApp._on_indexed: First item is a DocumentChunk. Text: {list(self.current_manifest.chunks.values())[0].text[:100]}...")
                else:
                    print(f"RAGApp._on_indexed: First item is an unexpected type: {first_item_type}")
            else:
                print("RAGApp._on_indexed: self.current_manifest.chunks is empty before VIM update.")
            # ---- END DEBUG PRINT ----
            try:
                num_added, num_unchanged, num_skipped = self.vector_index_manager.update_from_manifest(self.current_manifest)
                print(f"VectorIndexManager updated: {num_added} added, {num_unchanged} unchanged, {num_skipped} skipped.")
                self.vector_index_manager._save_index()
            except AttributeError as ae:
                print(f"ATTRIBUTE ERROR during VIM update: {ae}. This usually means an object was expected but a str was found (or vice-versa).")
                QMessageBox.warning(self, "Indexing Error", f"Could not update vector index (AttributeError): {ae}")
            except Exception as e:
                print(f"Error updating VectorIndexManager: {e}")
                QMessageBox.warning(self, "Indexing Error", f"Could not update vector index: {e}")
        
        if self.hydroid_engine and self.chunks:
             # Pass all *currently processed* chunks to fit BM25 inside the engine.
             # HydroidEngine uses the manifest primarily for getting chunk details by hash during search.
             # The BM25 part of HydroidEngine needs the actual chunk texts.
            print(f"RAGApp._on_indexed: Updating HydroidEngine corpus with {len(self.chunks)} chunks.")
            self.hydroid_engine.update_corpus(self.chunks)
        elif not self.chunks:
            print("No chunks to update Hydroid Engine with.")
        elif not self.hydroid_engine:
            print("Hydroid Engine not initialized. Skipping corpus update.")

        # ... (rest of UI updates like populating lists, saving main cache, etc.)
        # Populate file list
        self.doc_list.clear()
        for n in names:
            self.doc_list.addItem(n)
        
        self.kw_combo.clear()
        self.kw_combo.addItems(sorted(km.keys())) # Ensure km is not None
        
        self.sec_tree.clear()
        for i, name_item in enumerate(names):
            p = QTreeWidgetItem(self.sec_tree, [name_item])
            if i < len(secs):
                for s in secs[i]:
                    QTreeWidgetItem(p, [s])
        
        # Cache the main RAGApp data (docs, model, names, etc.)
        # The vector index and chunk manifest are saved by their respective managers/processors
        with open(Config.CACHE_PATH, 'wb') as f:
            pickle.dump((docs, model, names, secs, tfidf, km, self.chunks), f) # Save self.chunks
        
        self.query.setEnabled(True)
        self.ask_btn.setEnabled(True)
        self.query.setPlaceholderText("Ask a question...")
        
        self.status_msg.setText(
            f"Indexed {len(docs)} documents, {len(self.chunks)} chunks, "
            f"{sum(len(d.split()) for d in docs if d)} tokens, {len(km)} keywords"
        )
        for child in self.findChildren(QPushButton):
            if child.text() == "🔄 Process Folder": # Or initial text
                child.setText("🔄 Re-process Folder")
                child.setEnabled(True)
                break
        self.progress.setValue(100)


    def _do_query(self) -> None:
        q = self.query.toPlainText().strip()
        if not q:
            QMessageBox.warning(self, "Empty Query", "Please enter a search query.")
            return

        # Initialize ranked_chunks_with_scores before any search happens
        ranked_chunks_with_scores = []

        if not self.hydroid_engine:
            QMessageBox.critical(self, "Search Error", "Hydroid Retrieval Engine is not initialized.")
            self.status_msg.setText("Error: Retrieval Engine not available.")
            return
        if not self.chunks:
             QMessageBox.information(self, "Not Ready", "No chunks available for searching. Please process a folder first.")
             self.status_msg.setText("No chunks processed. Select and process a folder.")
             return

        while self.r_layout.count():
            item = self.r_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()

        self.status_msg.setText(f"Searching with Hydroid Engine for: {q}...")
        QApplication.processEvents()

        try:
            # Use the HydroidRetrievalEngine for searching
            enable_debug_logging = True # Set to True for debug logs
            
            # New HydroidRetrievalEngine API
            search_results = self.hydroid_engine.search(
                query=q,
                top_k=5,               # Display top 5 results
                semantic_weight=0.7,   # Weight for semantic search
                keyword_weight=0.3,    # Weight for keyword search
                debug_log=enable_debug_logging
            )

            if search_results:
                self.status_msg.setText(f"Displaying top {len(search_results)} hybrid results for '{q}'.")
                
                # Convert to format expected by _show_parsed_details
                ranked_chunks_with_scores = [(result["combined_score"], result["chunk"]) for result in search_results]
                self.last_top_chunks_for_details = ranked_chunks_with_scores
                
                for result in search_results:
                    chunk_obj = result["chunk"]
                    doc_name = chunk_obj.metadata.doc_name
                    section_title = chunk_obj.metadata.section_title
                    chunk_type = chunk_obj.metadata.chunk_type
                    image_path = getattr(chunk_obj.metadata, 'image_path', "")
                    combined_score = result["combined_score"]
                    semantic_score = result["semantic_score"]
                    keyword_score = result["keyword_score"]

                    # Get keywords from parent document for context (current approach)
                    doc_idx = self.names.index(doc_name) if doc_name in self.names else -1
                    keywords_for_card = []
                    if doc_idx != -1 and self.rhino_tfidf_mat is not None and self.rhino_tfidf_vec is not None:
                        try:
                            doc_tfidf_scores = self.rhino_tfidf_mat[doc_idx].toarray()[0]
                            feature_names = self.rhino_tfidf_vec.get_feature_names_out()
                            top_indices = doc_tfidf_scores.argsort()[-5:][::-1]
                            keywords_for_card = [feature_names[i] for i in top_indices if doc_tfidf_scores[i] > 0]
                        except Exception as e_kw:
                            print(f"Error extracting card keywords: {e_kw}")
                    
                    card = ResultCard(
                        title=doc_name,
                        snippet=chunk_obj.text[:400].replace("\n", " ") + "…",
                        hybrid_score=combined_score,
                        keywords=keywords_for_card,
                        sections=[section_title] if section_title else [],
                        doc2vec_score=semantic_score, # Now using semantic score directly
                        tfidf_score=keyword_score,    # Now using keyword score directly
                        query_terms=q.split(),
                        chunk_type=chunk_type,
                        section_title=section_title,
                        image_path=image_path
                    )
                    self.r_layout.addWidget(card)
            else:
                self.status_msg.setText(f"No relevant chunks found for '{q}' using Hydroid Engine.")

            self.btn_parsed.setEnabled(len(search_results) > 0)

            # AI Context Generation
            if self.ai_cb.isChecked():
                ai_context_parts = []
                if ranked_chunks_with_scores:
                    for score_val_ctx, chunk_obj_ctx in ranked_chunks_with_scores[:3]: # Top 3 for context
                        doc_name_ctx = chunk_obj_ctx.metadata.doc_name
                        section_ctx = chunk_obj_ctx.metadata.section_title
                        section_info_ctx = f" (Section: {section_ctx})" if section_ctx else ""
                        ai_context_parts.append(f"--- From: {doc_name_ctx}{section_info_ctx} ---\n{chunk_obj_ctx.text[:700]}")
                    ai_final_context = "\n\n".join(ai_context_parts)
                else:
                    ai_final_context = "No relevant information found in documents for your query."
                
                prompt_for_ai = Config.PROMPT_TMPL.format(context=ai_final_context, query=q)
                threading.Thread(target=self._ai_call,
                                 args=(self.model_cb.currentText(), prompt_for_ai),
                                 daemon=True).start()
        except Exception as e_query:
            print(f"Error during Hydroid Engine query processing: {e_query}")
            QMessageBox.critical(self, "Query Error", f"An error occurred during search: {e_query}")
            self.status_msg.setText(f"Error processing query: {e_query}")

    def _on_doc(self, item):
        i = self.names.index(item.text())
        self.r_scroll.verticalScrollBar().setValue(i * 150)

    def _on_section(self, item, _):
        if not item.parent():
            return
        doc, sec = item.parent().text(0), item.text(0)
        i = self.names.index(doc)
        m = re.search(rf"{re.escape(sec)}(.+?)(\n\n|\Z)", self.docs[i], re.DOTALL)
        txt = m.group(0) if m else sec
        QMessageBox.information(self, doc, txt[:1000])

    def _on_kw(self, kw):
        """Handle keyword selection to filter results."""
        if not kw:
            return
        
        # Clear RAG pane
        while self.r_layout.count():
            item = self.r_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Filter the chunks by keyword
        matching_chunks = []
        if hasattr(self, 'chunks') and self.chunks:
            for chunk in self.chunks:
                # Check if the keyword is in the chunk text
                if kw.lower() in chunk.text.lower():
                    matching_chunks.append(chunk)
        
        # If we found matching chunks, display them
        if matching_chunks:
            self.status_msg.setText(f"Found {len(matching_chunks)} chunks containing '{kw}'")
            
            # Sort chunks by document name for better organization
            matching_chunks.sort(key=lambda x: x.metadata.doc_name)
            
            # Display each matching chunk
            for i, chunk in enumerate(matching_chunks):
                doc_name = chunk.metadata.doc_name
                section_title = chunk.metadata.section_title
                chunk_type = chunk.metadata.chunk_type
                image_path = chunk.metadata.image_path if hasattr(chunk.metadata, 'image_path') else ""
                
                # Create simplified keywords from chunk text
                words = chunk.text.lower().split()
                word_freq = {}
                for word in words:
                    if len(word) > 3:  # Skip short words
                        word_freq[word] = word_freq.get(word, 0) + 1
                topk = sorted(word_freq.items(), key=lambda x: -x[1])[:5]
                topk = [k for k, _ in topk]
                
                # Create result card with highlighted keyword
                card = ResultCard(
                    title=doc_name,
                    snippet=chunk.text[:300].replace("\n", " ") + "…",
                    hybrid_score=1.0,  # Fixed score for keyword matches
                    keywords=topk,
                    sections=[],
                    doc2vec_score=0,
                    tfidf_score=0,
                    query_terms=[kw],  # Highlight the keyword
                    chunk_type=chunk_type,
                    section_title=section_title,
                    image_path=image_path
                )
                self.r_layout.addWidget(card)
        else:
            # If no chunks found, display documents from original keyword mapping
            self.doc_list.clear()
            docs_found = False
            for i in self.keyword_map.get(kw, []):
                self.doc_list.addItem(self.names[i])
                docs_found = True
            
            if docs_found:
                self.status_msg.setText(f"Found documents containing '{kw}'")
            else:
                self.status_msg.setText(f"No documents or chunks found containing '{kw}'")
            
        # Make sure the parsed details button is enabled
        self.btn_parsed.setEnabled(len(matching_chunks) > 0)

    def _show_parsed_details(self):
        if not hasattr(self, "last_top_chunks_for_details") or not self.last_top_chunks_for_details:
            QMessageBox.information(self, "No Details", "No search results to show details for.")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Parsed Hybrid Search Contexts")
        dlg.resize(800, 700) # Increased size for more info
        main_layout = QVBoxLayout(dlg)

        # Create a scroll area for the content
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        content_widget = QWidget()
        scroll_area.setWidget(content_widget)
        layout = QVBoxLayout(content_widget) # Layout for the scrollable content

        # Option to show debug JSON content if available (and HydroidEngine stores it)
        # For now, display chunk details directly.
        # We need a way to get the normalized vector/BM25 scores for each displayed chunk.
        # This might require the HydroidEngine to return this, or RAGApp to recalculate/retrieve from debug log.

        if (hasattr(self.hydroid_engine, 'last_query_debug_data_for_ui') and 
            self.hydroid_engine.last_query_debug_data_for_ui):
            debug_info = self.hydroid_engine.last_query_debug_data_for_ui
            query_info_label = QLabel(f"<b>Query:</b> {debug_info.get('query', 'N/A')}")
            layout.addWidget(query_info_label)
            
            for chunk_data in debug_info.get('fused_results', []):
                chunk_frame = QFrame()
                chunk_frame.setFrameShape(QFrame.Shape.StyledPanel)
                chunk_frame.setFrameShadow(QFrame.Shadow.Raised)
                frame_layout = QVBoxLayout(chunk_frame)

                # Retrieve full chunk from manifest using hash for complete details if needed
                full_chunk = self.current_manifest.get_chunk(chunk_data['hash'])
                if not full_chunk:
                    frame_layout.addWidget(QLabel(f"Could not retrieve full chunk for hash: {chunk_data['hash']}"))
                    layout.addWidget(chunk_frame)
                    continue

                header_text = f"<b>Doc:</b> {full_chunk.metadata.doc_name} (Chunk Hash: {chunk_data['hash']})"
                frame_layout.addWidget(QLabel(header_text))
                if full_chunk.metadata.section_title:
                    frame_layout.addWidget(QLabel(f"<i>Section:</i> {full_chunk.metadata.section_title}"))
                frame_layout.addWidget(QLabel(f"<i>Type:</i> {full_chunk.metadata.chunk_type.upper()}"))
                
                scores_text = f"<b>Fused:</b> {chunk_data['fused_score']:.4f} | Norm Vec: {chunk_data['norm_vector']:.4f} | Norm BM25: {chunk_data['norm_bm25']:.4f}"
                frame_layout.addWidget(QLabel(scores_text))

                text_preview = QTextEdit()
                text_preview.setReadOnly(True)
                # Highlight query terms in the preview
                html_content = full_chunk.text
                current_query_text = self.query.toPlainText().strip()
                for term in set(current_query_text.split()):
                    if term.strip():
                        html_content = re.sub(rf"\b{re.escape(term)}\b", 
                                              f"<span style='background:#00FFAA;color:#000'>{term}</span>", 
                                              html_content, flags=re.IGNORECASE)
                text_preview.setHtml(html_content)
                text_preview.setMaximumHeight(250) # Limit height for preview
                frame_layout.addWidget(text_preview)
                layout.addWidget(chunk_frame)
        else: # Fallback if detailed debug data not available from engine
            for fused_score, chunk_obj in self.last_top_chunks_for_details:
                chunk_frame = QFrame()
                chunk_frame.setFrameShape(QFrame.Shape.StyledPanel)
                chunk_frame.setFrameShadow(QFrame.Shadow.Raised)
                frame_layout = QVBoxLayout(chunk_frame)

                header_text = f"<b>Doc:</b> {chunk_obj.metadata.doc_name} (Score: {fused_score:.4f})"
                frame_layout.addWidget(QLabel(header_text))
                if chunk_obj.metadata.section_title:
                    frame_layout.addWidget(QLabel(f"<i>Section:</i> {chunk_obj.metadata.section_title}"))
                frame_layout.addWidget(QLabel(f"<i>Type:</i> {chunk_obj.metadata.chunk_type.upper()}"))

                # Here we don't have easy access to normalized components without more changes to HydroidEngine.search return
                frame_layout.addWidget(QLabel("(Normalized component scores not directly available in this view)"))

                text_preview = QTextEdit()
                text_preview.setReadOnly(True)
                html_content = chunk_obj.text
                current_query_text = self.query.toPlainText().strip()
                for term in set(current_query_text.split()):
                    if term.strip():
                        html_content = re.sub(rf"\b{re.escape(term)}\b", 
                                              f"<span style='background:#00FFAA;color:#000'>{term}</span>", 
                                              html_content, flags=re.IGNORECASE)
                text_preview.setHtml(html_content)
                text_preview.setMaximumHeight(250)
                frame_layout.addWidget(text_preview)
                layout.addWidget(chunk_frame)

        layout.addStretch() # Push content to the top
        main_layout.addWidget(scroll_area)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        main_layout.addWidget(close_btn, alignment=Qt.AlignmentFlag.AlignRight)
        
        dlg.exec()

    def _ai_call(self, name: str, prompt: str) -> None:
        """Call the AI model and display the response.

        Args:
            name: The name of the AI model to use
            prompt: The prompt to send to the model
        """
        try:
            ans = LLMClient.call_model(name, prompt, offline=False)
            if not ans:
                ans = "No response from AI model."

            card = ResultCard("AI Answer", ans, 0, [], [], 0, 0, [])
            self.ai_layout.addWidget(card)

            # Ensure the layout is updated before scrolling
            self.ai_layout.update()

            # Use a timer to scroll after the widget is properly added and laid out
            QTimer.singleShot(100, lambda: self.ai_scroll.verticalScrollBar().setValue(
                self.ai_scroll.verticalScrollBar().maximum()
            ))
        except Exception as e:
            error_card = ResultCard("AI Error", f"Error getting AI response: {str(e)}", 0, [], [], 0, 0, [])
            self.ai_layout.addWidget(error_card)

    def closeEvent(self, event):
        """Clean up resources before closing."""
        # Stop document processor if running
        if hasattr(self, 'processor') and self.processor.isRunning():
            self.processor.quit()
            self.processor.wait()

        # Stop any running AI threads
        for thread in threading.enumerate():
            if thread != threading.current_thread() and thread.daemon:
                thread.join(timeout=0.5)

        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = RAGApp()
    win.resize(1200, 800)
    win.show()
    sys.exit(app.exec())



