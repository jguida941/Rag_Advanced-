#!/usr/bin/env python3
"""
image_processor.py - Image processing for multimodal document chunking

This module provides:
1. OCR text extraction from images
2. Image captioning using BLIP (or pretrained models)
3. Integration with the chunking system
"""

import os
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

# Try to import image processing libraries
try:
    import pytesseract
    from PIL import Image
    import numpy as np
    HAVE_OCR = True
except ImportError:
    HAVE_OCR = False
    print("Warning: pytesseract or PIL not found. OCR will be disabled.")

# Try to import BLIP for image captioning
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    HAVE_BLIP = True
except ImportError:
    HAVE_BLIP = False
    print("Warning: transformers library not found or BLIP not available. Image captioning will be disabled.")

# Import chunking components
from chunking import DocumentChunk, ChunkMetadata, ChunkManifest, SemanticChunker


@dataclass
class ImageProcessingConfig:
    """Configuration for image processing."""
    enable_ocr: bool = True
    enable_captioning: bool = True
    max_image_size: Tuple[int, int] = (800, 800)
    ocr_language: str = "eng"
    caption_max_length: int = 50
    resize_before_processing: bool = True


class ImageProcessor:
    """Processes images for multimodal chunking."""
    
    def __init__(self, config: Optional[ImageProcessingConfig] = None):
        """Initialize the image processor with given configuration."""
        self.config = config or ImageProcessingConfig()
        self.blip_model = None
        self.blip_processor = None
        
        # Check if we can do OCR
        self.can_do_ocr = HAVE_OCR and self.config.enable_ocr
        
        # Check if we can do captioning
        self.can_do_captioning = HAVE_BLIP and self.config.enable_captioning
        
        # Initialize BLIP model if available and enabled
        if self.can_do_captioning:
            try:
                self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
                print("BLIP captioning model loaded successfully")
            except Exception as e:
                print(f"Error loading BLIP model: {e}")
                self.can_do_captioning = False
    
    def prepare_image(self, image_path: str) -> Optional[Image.Image]:
        """
        Load and prepare an image for processing.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            PIL Image object or None if loading fails
        """
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return None
        
        try:
            # Open the image
            img = Image.open(image_path)
            
            # Resize if needed
            if self.config.resize_before_processing:
                max_w, max_h = self.config.max_image_size
                if img.width > max_w or img.height > max_h:
                    ratio = min(max_w / img.width, max_h / img.height)
                    new_size = (int(img.width * ratio), int(img.height * ratio))
                    img = img.resize(new_size, Image.LANCZOS)
            
            return img
        except Exception as e:
            print(f"Error preparing image {image_path}: {e}")
            return None
    
    def extract_text_ocr(self, image_path: str) -> str:
        """
        Extract text from an image using OCR.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Extracted text or placeholder if OCR fails/unavailable
        """
        if not self.can_do_ocr:
            return f"[Image: {os.path.basename(image_path)}]"
        
        img = self.prepare_image(image_path)
        if img is None:
            return f"[Image: {os.path.basename(image_path)}]"
        
        try:
            # Extract text with pytesseract
            text = pytesseract.image_to_string(img, lang=self.config.ocr_language)
            
            # If no text was extracted or it's very short
            if len(text.strip()) < 10:
                return f"[Image: {os.path.basename(image_path)}]\n" + text.strip()
            
            return text.strip()
        except Exception as e:
            print(f"OCR error on {image_path}: {e}")
            return f"[Image: {os.path.basename(image_path)}]"
    
    def generate_caption(self, image_path: str) -> str:
        """
        Generate a caption for an image using BLIP.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Generated caption or placeholder if captioning fails/unavailable
        """
        if not self.can_do_captioning:
            return f"[Image caption for: {os.path.basename(image_path)}]"
        
        img = self.prepare_image(image_path)
        if img is None:
            return f"[Image caption for: {os.path.basename(image_path)}]"
        
        try:
            # Convert PIL image to RGB if needed
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            # Process image and generate caption
            inputs = self.blip_processor(img, return_tensors="pt")
            outputs = self.blip_model.generate(
                **inputs, 
                max_length=self.config.caption_max_length, 
                num_beams=4, 
                early_stopping=True
            )
            caption = self.blip_processor.decode(outputs[0], skip_special_tokens=True)
            
            return caption.strip()
        except Exception as e:
            print(f"Captioning error on {image_path}: {e}")
            return f"[Image caption for: {os.path.basename(image_path)}]"
    
    def process_image(self, 
                      image_path: str, 
                      doc_name: str, 
                      doc_path: str, 
                      section_title: str = "", 
                      page_number: int = 0,
                      chunker: Optional[SemanticChunker] = None) -> List[DocumentChunk]:
        """
        Process an image and create OCR and caption chunks.
        
        Args:
            image_path: Path to the image file
            doc_name: Document name
            doc_path: Document path
            section_title: Section title for the image
            page_number: Page number (for multi-page documents)
            chunker: Optional chunker to split OCR text into multiple chunks
            
        Returns:
            List of DocumentChunk objects
        """
        chunks = []
        current_time = time.time()
        
        # Extract OCR text if enabled
        if self.can_do_ocr:
            ocr_text = self.extract_text_ocr(image_path)
            
            # If we have a chunker and substantial text, split it into chunks
            if chunker and len(ocr_text.strip()) > chunker.target_chunk_size * 10:  # Only chunk if enough text
                ocr_chunks = chunker.chunk_text(ocr_text, section_title)
                
                for i, chunk_text in enumerate(ocr_chunks):
                    # Create metadata
                    metadata = ChunkMetadata(
                        doc_name=doc_name,
                        doc_path=doc_path,
                        chunk_index=i,
                        section_title=section_title,
                        token_count=chunker.count_tokens(chunk_text),
                        chunk_type="ocr",
                        image_path=image_path,
                        created_at=current_time
                    )
                    
                    # Create chunk
                    chunk = DocumentChunk(text=chunk_text, metadata=metadata)
                    chunk.update_hash()
                    chunks.append(chunk)
            else:
                # Single OCR chunk
                ocr_metadata = ChunkMetadata(
                    doc_name=doc_name,
                    doc_path=doc_path,
                    chunk_index=0,
                    section_title=section_title,
                    token_count=chunker.count_tokens(ocr_text) if chunker else len(ocr_text.split()),
                    chunk_type="ocr",
                    image_path=image_path,
                    created_at=current_time
                )
                
                ocr_chunk = DocumentChunk(text=ocr_text, metadata=ocr_metadata)
                ocr_chunk.update_hash()
                chunks.append(ocr_chunk)
        
        # Generate caption if enabled
        if self.can_do_captioning:
            caption = self.generate_caption(image_path)
            
            caption_metadata = ChunkMetadata(
                doc_name=doc_name,
                doc_path=doc_path,
                chunk_index=len(chunks),
                section_title=section_title,
                token_count=chunker.count_tokens(caption) if chunker else len(caption.split()),
                chunk_type="caption",
                image_path=image_path,
                created_at=current_time
            )
            
            caption_chunk = DocumentChunk(text=caption, metadata=caption_metadata)
            caption_chunk.update_hash()
            chunks.append(caption_chunk)
        
        return chunks


class MultimodalChunker:
    """Extended chunker that handles text and images."""
    
    def __init__(self, 
                 base_chunker: SemanticChunker, 
                 image_processor: Optional[ImageProcessor] = None,
                 image_processing_config: Optional[ImageProcessingConfig] = None):
        """Initialize with base chunker and image processor."""
        self.base_chunker = base_chunker
        self.image_processor = image_processor or ImageProcessor(image_processing_config)
    
    def process_document_with_images(self, 
                                    text: str, 
                                    doc_name: str, 
                                    doc_path: str,
                                    image_paths: List[str],
                                    image_sections: Optional[List[str]] = None) -> List[DocumentChunk]:
        """
        Process a document with embedded images.
        
        Args:
            text: Document text
            doc_name: Document name
            doc_path: Document path
            image_paths: List of image paths
            image_sections: Optional list of section titles for each image
            
        Returns:
            List of DocumentChunk objects
        """
        # Process text chunks
        text_chunks = self.base_chunker.chunk_document(text, doc_name, doc_path)
        
        # Process images
        image_chunks = []
        for i, img_path in enumerate(image_paths):
            # Determine section title for this image
            section = (image_sections[i] if image_sections and i < len(image_sections) 
                      else f"Image {i+1}")
            
            # Process image
            chunks = self.image_processor.process_image(
                image_path=img_path,
                doc_name=doc_name,
                doc_path=doc_path,
                section_title=section,
                page_number=i,
                chunker=self.base_chunker
            )
            
            image_chunks.extend(chunks)
        
        # Combine text and image chunks
        all_chunks = text_chunks + image_chunks
        
        # Sort by position in document if possible
        all_chunks.sort(key=lambda x: (
            x.metadata.chunk_type != "text",  # Text chunks first
            x.metadata.start_char_idx if x.metadata.chunk_type == "text" else float('inf')
        ))
        
        # Update chunk indices
        for i, chunk in enumerate(all_chunks):
            chunk.metadata.chunk_index = i
        
        return all_chunks


def main():
    """Run a demo of the image processing functionality."""
    from chunking import SemanticChunker, ChunkManifest
    
    # Check if we have the required libraries
    if not HAVE_OCR:
        print("Cannot run demo: OCR libraries not installed.")
        print("Please install: pip install pytesseract pillow numpy")
        return
    
    if not HAVE_BLIP:
        print("Warning: BLIP model not available. Only OCR will be demonstrated.")
    
    # Initialize components
    base_chunker = SemanticChunker(target_chunk_size=200, chunk_overlap=50)
    img_processor = ImageProcessor()
    multimodal_chunker = MultimodalChunker(base_chunker, img_processor)
    
    # Create a test image with text (you'll need to provide a real image path)
    test_image_path = "test_image.png"
    
    # Check if the test image exists, otherwise suggest creating one
    if not os.path.exists(test_image_path):
        print(f"Test image not found: {test_image_path}")
        print("Please create a test image with text or provide a valid image path.")
        return
    
    # Process a single image
    print(f"Processing image: {test_image_path}")
    image_chunks = img_processor.process_image(
        test_image_path, 
        "test_document.md", 
        "docs/test_document.md",
        "Test Image",
        chunker=base_chunker
    )
    
    # Create a manifest
    manifest = ChunkManifest("image_test_manifest.json")
    for chunk in image_chunks:
        manifest.add_chunk(chunk)
    manifest.save()
    
    # Display results
    print(f"\nCreated {len(image_chunks)} chunks from image:")
    
    for i, chunk in enumerate(image_chunks):
        print(f"\nChunk {i+1} ({chunk.metadata.chunk_type}):")
        print(f"Hash: {chunk.metadata.content_hash[:8]}...")
        print(f"Text: {chunk.text[:100]}...")
    
    # Example of processing a document with embedded images
    print("\n" + "="*50)
    print("Example of processing a document with images")
    print("="*50)
    
    sample_text = """# Document with Images

## Introduction

This is a sample document that includes images.

## Image Section

The following image shows some test content.

## Conclusion

This demonstrates multimodal chunking.
"""
    
    # Process the document with images
    all_chunks = multimodal_chunker.process_document_with_images(
        sample_text,
        "multimodal_doc.md",
        "docs/multimodal_doc.md",
        [test_image_path],
        ["## Image Section"]
    )
    
    # Display results
    print(f"\nProcessed document with {len(all_chunks)} total chunks:")
    
    for i, chunk in enumerate(all_chunks):
        print(f"\nChunk {i+1} ({chunk.metadata.chunk_type}):")
        print(f"Section: {chunk.metadata.section_title}")
        print(f"Hash: {chunk.metadata.content_hash[:8]}...")
        print(f"Text: {chunk.text[:50]}...")


if __name__ == "__main__":
    main() 