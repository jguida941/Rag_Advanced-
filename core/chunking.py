#!/usr/bin/env python3
"""
chunking.py - Semantic document chunking for RAG systems

This module provides advanced chunking functionality:
1. Semantic document splitting that respects section boundaries
2. Chunk overlap for better context preservation
3. Token counting for accurate LLM input sizing
4. Image handling for multimodal documents
5. Metadata tracking for document provenance
"""

import json
import os
import re
import hashlib
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple, Set, Union
import uuid

# Load transformers tokenizer if available
try:
    from transformers import AutoTokenizer
    default_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    HAVE_TRANSFORMERS = True
except ImportError:
    HAVE_TRANSFORMERS = False
    print("Warning: transformers library not found. Using fallback token counting.")

# Load image processing libraries if available
try:
    import pytesseract
    from PIL import Image
    import numpy as np
    HAVE_OCR = True
except ImportError:
    HAVE_OCR = False
    print("Warning: pytesseract, PIL, or numpy not found. Image processing will be disabled.")

@dataclass
class ChunkMetadata:
    """Metadata for a document chunk."""
    doc_name: str
    doc_path: str
    chunk_index: int
    section_title: str = ""
    start_char_idx: int = 0
    end_char_idx: int = 0
    start_line: int = 0
    end_line: int = 0
    token_count: int = 0
    content_hash: str = ""
    chunk_type: str = "text"  # "text", "image", "code", etc.
    image_path: str = ""
    priority: str = "normal"  # "high", "normal", "low"
    created_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

@dataclass
class DocumentChunk:
    """A chunk of a document with its metadata."""
    text: str
    metadata: ChunkMetadata
    
    def update_hash(self):
        """Update the content hash based on the text."""
        hash_obj = hashlib.sha256(self.text.encode())
        self.metadata.content_hash = hash_obj.hexdigest()
        return self.metadata.content_hash
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "text": self.text,
            "metadata": self.metadata.to_dict()
        }

class ChunkManifest:
    """Manages a collection of document chunks and provides serialization."""
    
    def __init__(self, manifest_path: str):
        """Initialize chunk manifest with path to store/load from."""
        self.manifest_path = manifest_path
        self.chunks: Dict[str, DocumentChunk] = {}  # hash -> chunk
        self.doc_to_chunks: Dict[str, List[str]] = {}  # doc_path -> [hash1, hash2, ...]
        
        # Load existing manifest if it exists
        if os.path.exists(manifest_path):
            self.load()
    
    def add_chunk(self, chunk: DocumentChunk) -> str:
        """Add a chunk to the manifest."""
        # Ensure hash is up to date
        chunk_hash = chunk.update_hash()
        
        # Add to chunks dictionary
        self.chunks[chunk_hash] = chunk
        
        # Update document to chunks mapping
        doc_path = chunk.metadata.doc_path
        if doc_path not in self.doc_to_chunks:
            self.doc_to_chunks[doc_path] = []
        
        if chunk_hash not in self.doc_to_chunks[doc_path]:
            self.doc_to_chunks[doc_path].append(chunk_hash)
        
        return chunk_hash
    
    def add_chunks(self, chunks: List[DocumentChunk]) -> List[str]:
        """Add multiple chunks to the manifest."""
        return [self.add_chunk(chunk) for chunk in chunks]
    
    def get_chunk(self, chunk_hash: str) -> Optional[DocumentChunk]:
        """Get a chunk by its hash."""
        return self.chunks.get(chunk_hash)
    
    def get_chunks_for_doc(self, doc_path: str) -> List[DocumentChunk]:
        """Get all chunks for a document."""
        if doc_path not in self.doc_to_chunks:
            return []
        
        return [self.chunks[chunk_hash] for chunk_hash in self.doc_to_chunks[doc_path] 
                if chunk_hash in self.chunks]
    
    def remove_doc(self, doc_path: str):
        """Remove all chunks for a document."""
        if doc_path not in self.doc_to_chunks:
            return
        
        # Remove each chunk
        for chunk_hash in self.doc_to_chunks[doc_path]:
            if chunk_hash in self.chunks:
                del self.chunks[chunk_hash]
        
        # Remove from doc mapping
        del self.doc_to_chunks[doc_path]
    
    def save(self):
        """Save the manifest to disk."""
        # Convert to serializable format
        # Make sure we're serializing DocumentChunk objects properly
        chunk_dict = {}
        for h, c in self.chunks.items():
            if isinstance(c, DocumentChunk):
                chunk_dict[h] = c.to_dict()
            else:
                print(f"Warning: Expected DocumentChunk for hash {h}, but found {type(c)}. Converting to dict representation.")
                # Try to convert to dict if it's a string or other format
                if isinstance(c, str):
                    print(f"Error: Cannot serialize string as DocumentChunk: {c[:100]}...")
                    continue
                try:
                    chunk_dict[h] = c.to_dict()
                except AttributeError:
                    print(f"Error: Object for hash {h} has no to_dict method. Skipping.")
                    continue
        
        data = {
            "chunks": chunk_dict,
            "doc_to_chunks": self.doc_to_chunks
        }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(self.manifest_path)), exist_ok=True)
        
        # Write to file
        with open(self.manifest_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self):
        """Load the manifest from disk."""
        try:
            with open(self.manifest_path, 'r') as f:
                data = json.load(f)
            
            # Load chunks
            self.chunks = {}
            print(f"[ChunkManifest.load] Attempting to load {len(data.get('chunks', {}))} chunk entries from manifest.") # Debug
            for h, c_dict in data.get("chunks", {}).items():
                print(f"[ChunkManifest.load] Loading chunk dict for hash {h}: {type(c_dict)} -- {str(c_dict)[:200]}") # Debug c_dict
                if not isinstance(c_dict, dict):
                    print(f"[ChunkManifest.load] ERROR: c_dict for hash {h} is NOT a dict, it is {type(c_dict)}. Skipping.")
                    continue
                if "metadata" not in c_dict or "text" not in c_dict:
                    print(f"[ChunkManifest.load] ERROR: c_dict for hash {h} is missing 'metadata' or 'text'. Keys: {c_dict.keys()}. Skipping.")
                    continue
                
                metadata_dict = c_dict["metadata"]
                if not isinstance(metadata_dict, dict):
                    print(f"[ChunkManifest.load] ERROR: metadata for hash {h} is NOT a dict, it is {type(metadata_dict)}. Skipping.")
                    continue

                # Ensure all required fields for ChunkMetadata are present with default fallbacks if necessary
                # This is important if the manifest was saved with an older version of ChunkMetadata
                required_meta_fields = ChunkMetadata.__annotations__.keys()
                for field_name in required_meta_fields:
                    if field_name not in metadata_dict:
                        # Provide a sensible default if a field is missing based on field type
                        print(f"[ChunkManifest.load] Warning: metadata for hash {h} missing field '{field_name}'. Adding default value.")
                        if field_name in ['doc_name', 'doc_path', 'content_hash']:
                            metadata_dict[field_name] = "unknown" 
                        elif field_name in ['section_title', 'chunk_type', 'image_path']:
                            metadata_dict[field_name] = ""
                        elif field_name == 'chunk_index':
                            metadata_dict[field_name] = 0
                        elif field_name == 'token_count':
                            metadata_dict[field_name] = 0
                        elif field_name == 'priority':
                            metadata_dict[field_name] = "normal"
                
                # Fix truncated section titles - look for single letter section titles 
                # that may have been created by older parsers
                if 'section_title' in metadata_dict and len(metadata_dict['section_title'].strip()) == 1:
                    print(f"[ChunkManifest.load] Warning: Found truncated section title '{metadata_dict['section_title']}'. Setting to empty.")
                    metadata_dict['section_title'] = ""
                
                # Also check if section title is "Section: X" format and extract just the X
                if 'section_title' in metadata_dict and metadata_dict['section_title'].startswith('Section: '):
                    print(f"[ChunkManifest.load] Warning: Found 'Section: X' format in section title. Extracting original title.")
                    metadata_dict['section_title'] = metadata_dict['section_title'][9:].strip()
                
                try:
                    metadata = ChunkMetadata(**metadata_dict)
                    chunk = DocumentChunk(text=c_dict["text"], metadata=metadata)
                    self.chunks[h] = chunk
                except TypeError as te_meta:
                    print(f"[ChunkManifest.load] ERROR creating ChunkMetadata for hash {h} due to TypeError: {te_meta}. Metadata was: {metadata_dict}. Skipping.")
                    continue
                except Exception as e_meta:
                    print(f"[ChunkManifest.load] ERROR creating chunk/metadata for hash {h}: {e_meta}. Skipping.")
                    continue
            
            # Load doc to chunks mapping
            self.doc_to_chunks = data.get("doc_to_chunks", {})
            print(f"[ChunkManifest.load] Loaded {len(self.chunks)} DocumentChunk objects.") # Debug
            
        except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
            print(f"Error loading manifest: {e}")
            self.chunks = {}
            self.doc_to_chunks = {}

class SemanticChunker:
    """Semantic document chunker that respects section boundaries and includes image processing."""
    
    def __init__(self, target_chunk_size: int = 512, chunk_overlap: int = 50, 
                 tokenizer=None, detect_sections: bool = True, 
                 max_image_size: Tuple[int, int] = (800, 800)):
        """
        Initialize the semantic chunker.
        
        Args:
            target_chunk_size: Target size for each chunk in tokens
            chunk_overlap: Number of tokens to overlap between chunks
            tokenizer: Tokenizer to use for counting tokens (uses GPT-2 by default)
            detect_sections: Whether to detect sections by headings
            max_image_size: Maximum size for images before resizing
        """
        self.target_chunk_size = target_chunk_size
        self.chunk_overlap = chunk_overlap
        self.detect_sections = detect_sections
        self.max_image_size = max_image_size
        
        # Set up tokenizer
        if tokenizer:
            self.tokenizer = tokenizer
        elif HAVE_TRANSFORMERS:
            self.tokenizer = default_tokenizer
        else:
            self.tokenizer = None
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the tokenizer."""
        if not text:
            return 0
            
        if self.tokenizer:
            return len(self.tokenizer(text)['input_ids'])
        else:
            # Fallback approximation: rough token count
            return len(text.split())
    
    def detect_section_breaks(self, text: str) -> List[Tuple[int, int, str]]:
        """
        Detect section breaks in text based on heading patterns.
        Returns list of (start_idx, end_idx, section_title) tuples.
        """
        if not self.detect_sections:
            return [(0, len(text), "")]
        
        # Match Markdown headers and other section dividers
        # We specifically match markdown headers with multiple ways:
        # 1. Hashed headers: # Title, ## Subtitle, etc.
        # 2. Underlined headers: Title\n=====, Subtitle\n-----, etc.
        section_pattern = re.compile(
            r'^(?:(?P<hashed>#{1,6}\s+.+)|(?P<underlined>[^\n]+\n[-=]+\s*$))',
            re.MULTILINE
        )
        
        sections = []
        last_start = 0
        last_title = ""
        
        for match in section_pattern.finditer(text):
            if last_start > 0:  # Not the first section
                sections.append((last_start, match.start(), last_title))
            
            last_start = match.start()
            
            # Extract the section title based on the type of heading
            if match.group('hashed'):  # # Heading style
                # Remove the leading # and whitespace
                raw_title = match.group('hashed')
                last_title = re.sub(r'^#+\s*', '', raw_title).strip()
                
                # Debug output to help diagnose section detection issues
                print(f"Found Markdown header: '#{raw_title}' -> '{last_title}'")
                
            elif match.group('underlined'):  # Underlined style
                # Only keep the text part before the underline
                raw_title = match.group('underlined')
                if '\n' in raw_title:
                    last_title = raw_title.split('\n')[0].strip()
                else:
                    last_title = raw_title.strip()
                    
                # Debug output
                print(f"Found underlined header: '{raw_title}' -> '{last_title}'")
            else:
                # Should not happen, but for safety
                last_title = match.group(0).strip()
                print(f"Found other header type: '{match.group(0)}' -> '{last_title}'")
        
        # Add the last section
        if last_start < len(text):
            sections.append((last_start, len(text), last_title))
        
        # If no sections were found, treat the whole document as one section
        if not sections:
            sections = [(0, len(text), "")]
            
        return sections
    
    def process_image(self, image_path: str) -> str:
        """
        Process an image using OCR and return the extracted text.
        """
        if not HAVE_OCR or not os.path.exists(image_path):
            return f"[Image: {os.path.basename(image_path)}]"
        
        try:
            # Open the image
            img = Image.open(image_path)
            
            # Resize if needed
            max_w, max_h = self.max_image_size
            if img.width > max_w or img.height > max_h:
                ratio = min(max_w / img.width, max_h / img.height)
                new_size = (int(img.width * ratio), int(img.height * ratio))
                img = img.resize(new_size, Image.LANCZOS)
            
            # Extract text
            text = pytesseract.image_to_string(img)
            
            # If no text was extracted or it's very short
            if len(text.strip()) < 10:
                return f"[Image: {os.path.basename(image_path)}]\n" + text
            
            return text
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return f"[Image: {os.path.basename(image_path)}]"
    
    def process_text(self, text: str, section_title: str = "") -> List[str]:
        """
        Split text into overlapping chunks respecting sentence boundaries.
        """
        if not text:
            return []
            
        # Count tokens to determine if splitting is needed
        token_count = self.count_tokens(text)
        if token_count <= self.target_chunk_size:
            return [text]
        
        # Split on paragraph or sentence boundaries
        splits = []
        
        # First try to split on paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        
        current_chunk = []
        current_tokens = 0
        
        for paragraph in paragraphs:
            paragraph_tokens = self.count_tokens(paragraph)
            
            # If a single paragraph is too large, split by sentences
            if paragraph_tokens > self.target_chunk_size:
                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                for sentence in sentences:
                    sentence_tokens = self.count_tokens(sentence)
                    
                    if current_tokens + sentence_tokens > self.target_chunk_size and current_chunk:
                        splits.append("\n\n".join(current_chunk))
                        
                        # Keep overlap by retaining some of the previous content
                        if self.chunk_overlap > 0 and current_chunk:
                            overlap_text = "\n\n".join(current_chunk[-2:])
                            overlap_tokens = self.count_tokens(overlap_text)
                            if overlap_tokens <= self.chunk_overlap:
                                current_chunk = current_chunk[-2:]
                                current_tokens = overlap_tokens
                            else:
                                current_chunk = [current_chunk[-1]]
                                current_tokens = self.count_tokens(current_chunk[0])
                        else:
                            current_chunk = []
                            current_tokens = 0
                    
                    current_chunk.append(sentence)
                    current_tokens += sentence_tokens
            else:
                # If adding this paragraph would exceed the target, finalize the current chunk
                if current_tokens + paragraph_tokens > self.target_chunk_size and current_chunk:
                    splits.append("\n\n".join(current_chunk))
                    
                    # Keep overlap
                    if self.chunk_overlap > 0 and current_chunk:
                        overlap_text = "\n\n".join(current_chunk[-1:])
                        current_chunk = current_chunk[-1:]
                        current_tokens = self.count_tokens(overlap_text)
                    else:
                        current_chunk = []
                        current_tokens = 0
                
                current_chunk.append(paragraph)
                current_tokens += paragraph_tokens
        
        # Add the last chunk if there's anything left
        if current_chunk:
            splits.append("\n\n".join(current_chunk))
        
        return splits
    
    def chunk_document(self, text: str, doc_name: str, doc_path: str, 
                       section_titles: Optional[List[str]] = None) -> List[DocumentChunk]:
        """
        Split a document into semantic chunks.
        
        Args:
            text: Document text
            doc_name: Document name
            doc_path: Path to the document
            section_titles: Optional explicit section titles
            
        Returns:
            List of DocumentChunk objects
        """
        chunks = []
        
        # Detect sections if not provided
        if section_titles is None:
            sections = self.detect_section_breaks(text)
        else:
            # Assume the whole document is one section with provided titles
            section_length = len(text) // len(section_titles)
            sections = []
            for i, title in enumerate(section_titles):
                start = i * section_length
                end = (i + 1) * section_length if i < len(section_titles) - 1 else len(text)
                sections.append((start, end, title))
        
        # Process each section
        for start_idx, end_idx, section_title in sections:
            section_text = text[start_idx:end_idx]
            
            # Skip empty sections
            if not section_text.strip():
                continue
            
            # Get text chunks
            text_chunks = self.process_text(section_text, section_title)
            
            # Create DocumentChunk objects
            for i, chunk_text in enumerate(text_chunks):
                # Count tokens
                token_count = self.count_tokens(chunk_text)
                
                # Get character indices for this chunk in the original text
                if i == 0:
                    chunk_start = start_idx
                else:
                    # Approximate position - could be improved
                    prev_chunks_length = sum(len(tc) for tc in text_chunks[:i])
                    chunk_start = start_idx + prev_chunks_length
                
                chunk_end = chunk_start + len(chunk_text)
                
                # Create metadata - use full section title instead of truncating
                metadata = ChunkMetadata(
                    doc_name=doc_name,
                    doc_path=doc_path,
                    chunk_index=len(chunks),
                    section_title=section_title,  # Use full section title
                    start_char_idx=chunk_start,
                    end_char_idx=chunk_end,
                    start_line=section_text[:chunk_start-start_idx].count('\n') if chunk_start > start_idx else 0,
                    end_line=section_text[:chunk_end-start_idx].count('\n') if chunk_end > start_idx else 0,
                    token_count=token_count,
                    chunk_type="text"
                )
                
                # Create the chunk
                chunk = DocumentChunk(
                    text=chunk_text,
                    metadata=metadata
                )
                
                # Update hash
                chunk.update_hash()
                
                chunks.append(chunk)
        
        return chunks
    
    def process_document_with_images(self, text: str, doc_name: str, doc_path: str, 
                                    section_titles: Optional[List[str]] = None,
                                    image_paths: Optional[List[str]] = None) -> List[DocumentChunk]:
        """
        Process a document with images and create chunks.
        
        Args:
            text: Document text
            doc_name: Document name
            doc_path: Document path
            section_titles: Optional explicit section titles
            image_paths: List of image paths to process
            
        Returns:
            List of DocumentChunk objects
        """
        chunks = self.chunk_document(text, doc_name, doc_path, section_titles)
        
        # Process images if provided
        if image_paths and HAVE_OCR:
            for i, img_path in enumerate(image_paths):
                # Extract text from image
                image_text = self.process_image(img_path)
                
                # Create image chunk metadata
                metadata = ChunkMetadata(
                    doc_name=doc_name,
                    doc_path=doc_path,
                    chunk_index=len(chunks),
                    section_title=f"Image {i+1}",
                    token_count=self.count_tokens(image_text),
                    chunk_type="image",
                    image_path=img_path
                )
                
                # Create the chunk
                chunk = DocumentChunk(
                    text=image_text,
                    metadata=metadata
                )
                
                # Update hash
                chunk.update_hash()
                
                chunks.append(chunk)
        
        return chunks

# Demo functionality
def main():
    """Run a demo of the chunking functionality."""
    chunker = SemanticChunker(target_chunk_size=100, chunk_overlap=20)
    
    # Sample text with multiple sections
    sample_text = """# Introduction to Python

Python is a versatile programming language that's great for beginners and experts alike.

## Basic Syntax

Python syntax is clean and easy to read. Here's a simple example:

```python
# This is a comment
print("Hello, World!")
```

## Data Structures

Python has several built-in data structures:
- Lists: ordered, mutable collections
- Tuples: ordered, immutable collections
- Dictionaries: key-value mappings

### Working with Lists

Lists are very flexible:

```python
my_list = [1, 2, 3, 4, 5]
my_list.append(6)
print(my_list)  # Outputs: [1, 2, 3, 4, 5, 6]
```

## Functions

Functions help organize code and make it reusable:

```python
def greet(name):
    return f"Hello, {name}!"
    
print(greet("Python User"))
```
"""
    
    # Chunk the document
    chunks = chunker.chunk_document(sample_text, "python_intro.md", "docs/python_intro.md")
    
    # Create a manifest
    manifest = ChunkManifest("sample_manifest.json")
    manifest.add_chunks(chunks)
    manifest.save()
    
    # Display results
    print(f"Split into {len(chunks)} chunks:\n")
    
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}:")
        print(f"Section: {chunk.metadata.section_title}")
        print(f"Type: {chunk.metadata.chunk_type}")
        print(f"Tokens: {chunk.metadata.token_count}")
        print(f"Hash: {chunk.metadata.content_hash[:8]}...")
        print(f"Text: {chunk.text[:50]}...\n")

if __name__ == "__main__":
    main() 