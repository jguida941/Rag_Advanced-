"""
embedding.py - Modern embedding system with FAISS vector indexing

This module implements Phase 2 of the RAG enhancement roadmap, providing:
1. Integration with sentence-transformers for advanced embeddings
2. FAISS vector indexing for fast similarity search
3. Differential embedding based on chunk checksums
"""

import os
import json
import pickle
import hashlib
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field

try:
    import faiss
    HAVE_FAISS = True
except ImportError:
    HAVE_FAISS = False
    print("Warning: faiss library not found. Install with: pip install faiss-cpu (or faiss-gpu)")

try:
    from sentence_transformers import SentenceTransformer
    HAVE_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAVE_SENTENCE_TRANSFORMERS = False
    print("Warning: sentence_transformers library not found. Install with: pip install sentence-transformers")

from chunking import DocumentChunk, ChunkMetadata, ChunkManifest


@dataclass
class IndexEntry:
    """Metadata for an entry in the FAISS index."""
    faiss_id: int              # ID in the FAISS index
    content_hash: str          # Content hash for differential update
    doc_name: str              # Document name
    doc_path: str              # Document path
    chunk_index: int           # Chunk index within document
    embedding_timestamp: str   # When embedding was created/updated


class VectorIndexManager:
    """Manages the FAISS vector index for document chunks."""
    
    def __init__(self, 
                 index_dir: str = "vector_index",
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 index_type: str = "IndexFlatL2"):
        """
        Initialize the vector index manager.
        
        Args:
            index_dir: Directory to store index files
            model_name: Name of the sentence-transformers model to use
            index_type: Type of FAISS index (e.g., "IndexFlatL2", "IndexIVFFlat")
        """
        # Validate required libraries
        if not HAVE_SENTENCE_TRANSFORMERS:
            raise ImportError("sentence_transformers library required for embedding generation")
        if not HAVE_FAISS:
            raise ImportError("faiss library required for vector indexing")
        
        self.index_dir = index_dir
        self.model_name = model_name
        self.index_type = index_type
        
        # Create index directory if it doesn't exist
        os.makedirs(index_dir, exist_ok=True)
        
        # Paths for index files
        self.index_path = os.path.join(index_dir, "vector_index.faiss")
        self.metadata_path = os.path.join(index_dir, "index_metadata.json")
        self.lookup_path = os.path.join(index_dir, "index_lookup.pkl")
        
        # Initialize sentence transformer model
        self._load_model()
        
        # Initialize or load index and metadata
        self._initialize_index()
        
    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            print(f"Loading model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            print(f"Model loaded with embedding dimension: {self.embedding_dim}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def _initialize_index(self):
        """Initialize or load the FAISS index and metadata."""
        # Initialize empty metadata
        self.metadata = {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "index_type": self.index_type,
            "creation_time": None,
            "last_updated": None,
            "num_vectors": 0
        }
        
        # Initialize empty lookup for ID to chunk mapping
        self.id_to_entry: Dict[int, IndexEntry] = {}
        self.hash_to_id: Dict[str, int] = {}
        self.next_id = 0
        
        # Check if index files exist
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path) and os.path.exists(self.lookup_path):
            self._load_existing_index()
        else:
            self._create_new_index()
    
    def _create_new_index(self):
        """Create a new FAISS index."""
        import datetime
        now = datetime.datetime.now().isoformat()
        
        # Create a new FAISS index
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        # Update metadata
        self.metadata["creation_time"] = now
        self.metadata["last_updated"] = now
        self.metadata["num_vectors"] = 0
        
        # Save index and metadata
        self._save_index()
    
    def _load_existing_index(self):
        """Load existing index and metadata from files."""
        import datetime
        
        # Load metadata
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Verify embedding dimension matches current model
        if self.metadata["embedding_dim"] != self.embedding_dim:
            raise ValueError(f"Embedding dimension mismatch: index has {self.metadata['embedding_dim']}, model has {self.embedding_dim}")
        
        # Load FAISS index
        self.index = faiss.read_index(self.index_path)
        
        # Load ID to chunk mapping
        with open(self.lookup_path, 'rb') as f:
            lookup_data = pickle.load(f)
            self.id_to_entry = lookup_data.get("id_to_entry", {})
            self.hash_to_id = lookup_data.get("hash_to_id", {})
            self.next_id = lookup_data.get("next_id", 0)
        
        print(f"Loaded index with {self.metadata['num_vectors']} vectors")
        
        # Update last accessed time
        self.metadata["last_accessed"] = datetime.datetime.now().isoformat()
        self._save_metadata()
    
    def _save_index(self):
        """Save index, metadata, and lookup to files."""
        # Save FAISS index
        faiss.write_index(self.index, self.index_path)
        
        # Save metadata
        self._save_metadata()
        
        # Save lookup
        with open(self.lookup_path, 'wb') as f:
            lookup_data = {
                "id_to_entry": self.id_to_entry,
                "hash_to_id": self.hash_to_id,
                "next_id": self.next_id
            }
            pickle.dump(lookup_data, f)
    
    def _save_metadata(self):
        """Save metadata to file."""
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a text string."""
        embedding = self.model.encode(text)
        return embedding
    
    def add_chunk(self, chunk: DocumentChunk) -> int:
        """
        Add a document chunk to the index.
        
        Args:
            chunk: The document chunk to add
            
        Returns:
            The FAISS ID of the added chunk
        """
        import datetime
        
        # Ensure the chunk has a content hash
        if not chunk.metadata.content_hash:
            chunk.update_hash()
        
        content_hash = chunk.metadata.content_hash
        
        # Check if this chunk is already in the index
        if content_hash in self.hash_to_id:
            return self.hash_to_id[content_hash]
        
        # Generate embedding for the chunk
        embedding = self.embed_text(chunk.text)
        embedding = embedding.reshape(1, -1).astype(np.float32)
        
        # Add to FAISS index
        faiss_id = self.next_id
        self.index.add(embedding)
        
        # Create entry metadata
        entry = IndexEntry(
            faiss_id=faiss_id,
            content_hash=content_hash,
            doc_name=chunk.metadata.doc_name,
            doc_path=chunk.metadata.doc_path,
            chunk_index=chunk.metadata.chunk_index,
            embedding_timestamp=datetime.datetime.now().isoformat()
        )
        
        # Update lookups
        self.id_to_entry[faiss_id] = entry
        self.hash_to_id[content_hash] = faiss_id
        
        # Update metadata
        self.next_id += 1
        self.metadata["num_vectors"] += 1
        self.metadata["last_updated"] = datetime.datetime.now().isoformat()
        
        # Save changes (saving is expensive, so we'll only do it periodically in practice)
        if self.metadata["num_vectors"] % 100 == 0:
            self._save_index()
        else:
            self._save_metadata()  # This is cheaper than saving the full index
        
        return faiss_id
    
    def add_chunks(self, chunks: List[DocumentChunk]) -> List[int]:
        """
        Add multiple document chunks to the index.
        
        Args:
            chunks: The document chunks to add
            
        Returns:
            The FAISS IDs of the added chunks
        """
        import datetime
        
        # First, identify which chunks need to be added
        new_chunks = []
        faiss_ids = []
        
        for chunk in chunks:
            # Ensure the chunk has a content hash
            if not chunk.metadata.content_hash:
                chunk.update_hash()
            
            content_hash = chunk.metadata.content_hash
            
            # Check if this chunk is already in the index
            if content_hash in self.hash_to_id:
                faiss_ids.append(self.hash_to_id[content_hash])
            else:
                new_chunks.append(chunk)
                faiss_ids.append(None)  # Placeholder to be filled after adding
        
        if not new_chunks:
            return faiss_ids  # No new chunks to add
        
        # Generate embeddings for all new chunks at once (more efficient)
        texts = [chunk.text for chunk in new_chunks]
        embeddings = self.model.encode(texts)
        embeddings = embeddings.astype(np.float32)
        
        # Add to FAISS index
        new_faiss_ids = list(range(self.next_id, self.next_id + len(new_chunks)))
        self.index.add(embeddings)
        
        # Create entries and update lookups
        now = datetime.datetime.now().isoformat()
        
        for i, (chunk, faiss_id) in enumerate(zip(new_chunks, new_faiss_ids)):
            content_hash = chunk.metadata.content_hash
            
            # Create entry metadata
            entry = IndexEntry(
                faiss_id=faiss_id,
                content_hash=content_hash,
                doc_name=chunk.metadata.doc_name,
                doc_path=chunk.metadata.doc_path,
                chunk_index=chunk.metadata.chunk_index,
                embedding_timestamp=now
            )
            
            # Update lookups
            self.id_to_entry[faiss_id] = entry
            self.hash_to_id[content_hash] = faiss_id
            
            # Update the output list
            idx = chunks.index(chunk)
            faiss_ids[idx] = faiss_id
        
        # Update metadata
        self.next_id += len(new_chunks)
        self.metadata["num_vectors"] += len(new_chunks)
        self.metadata["last_updated"] = now
        
        # Save changes
        self._save_index()
        
        return faiss_ids
    
    def search_index(self, query: str, top_k: int = 5) -> List[Tuple[int, float, IndexEntry]]:
        """
        Search the index for similar chunks.
        
        Args:
            query: The search query
            top_k: Number of results to return
            
        Returns:
            List of tuples (faiss_id, distance, entry_metadata)
        """
        # Generate query embedding
        query_embedding = self.embed_text(query)
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        
        # Search the index
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Prepare results
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx != -1:  # -1 means no result
                dist = distances[0][i]
                entry = self.id_to_entry.get(int(idx))
                if entry:
                    results.append((int(idx), float(dist), entry))
        
        return results
    
    def update_from_manifest(self, manifest: ChunkManifest) -> Tuple[int, int, int]:
        """
        Update the index from a chunk manifest.
        
        Args:
            manifest: The chunk manifest
            
        Returns:
            Tuple of (num_added, num_unchanged, num_skipped)
        """
        num_added = 0
        num_unchanged = 0
        num_skipped = 0
        
        # Process all chunks in the manifest
        for chunk_hash, chunk in manifest.chunks.items():
            # Verify we have a DocumentChunk object, not a string
            if isinstance(chunk, str):
                print(f"Warning: Expected DocumentChunk object but found string for hash {chunk_hash}. Skipping.")
                num_skipped += 1
                continue
                
            # Skip chunks without text (e.g., some image chunks might be empty)
            if not hasattr(chunk, 'text') or not chunk.text or not chunk.text.strip():
                num_skipped += 1
                continue
            
            # Ensure the chunk has a content hash
            if not chunk.metadata.content_hash:
                chunk.update_hash()
            
            content_hash = chunk.metadata.content_hash
            
            # Check if this chunk is already in the index
            if content_hash in self.hash_to_id:
                num_unchanged += 1
            else:
                try:
                    self.add_chunk(chunk)
                    num_added += 1
                except Exception as e:
                    print(f"Error adding chunk with hash {content_hash}: {e}")
                    num_skipped += 1
        
        # Save index if any changes were made
        if num_added > 0:
            self._save_index()
        
        return num_added, num_unchanged, num_skipped
    
    def get_chunk_by_id(self, faiss_id: int) -> Optional[IndexEntry]:
        """Get chunk entry by FAISS ID."""
        return self.id_to_entry.get(faiss_id)
    
    def search_chunks(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        User-friendly search function returning detailed chunk info.
        
        Args:
            query: The search query
            top_k: Number of results to return
            
        Returns:
            List of dictionaries with search results
        """
        results = self.search_index(query, top_k)
        
        detailed_results = []
        for idx, distance, entry in results:
            detailed_results.append({
                "faiss_id": idx,
                "similarity_score": 1.0 / (1.0 + distance),  # Convert distance to similarity
                "doc_name": entry.doc_name,
                "doc_path": entry.doc_path,
                "chunk_index": entry.chunk_index,
                "content_hash": entry.content_hash
            })
        
        return detailed_results 