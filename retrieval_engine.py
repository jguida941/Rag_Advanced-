#!/usr/bin/env python3
"""
Hydroid Retrieval Engine - Advanced hybrid search system combining:
1. Dense vector search with FAISS
2. Sparse vector search with BM25
3. Context-aware re-ranking
4. Section prioritization

Designed to create a comprehensive search experience across
various document types with per-chunk scoring and result fusion.
"""

import os
import json
import time
import math
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union, Set
from collections import defaultdict, Counter

from core.chunking import DocumentChunk, ChunkManifest, ChunkMetadata
from core.embedding import VectorIndexManager

class SimpleBM25:
    """Simple BM25 implementation for in-memory search."""
    
    def __init__(self, k1=1.5, b=0.75, epsilon=0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.corpus = []
        self.doc_freqs = Counter()
        self.doc_lengths = []
        self.avg_doc_length = 0
        self.idf = {}
        self.vocab_size = 0
        self.n_docs = 0
        
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25."""
        import re
        # Lowercase, remove punctuation, split on whitespace
        return [token.lower() for token in re.findall(r'\w+', text)]
    
    def fit(self, documents: List[str]) -> None:
        """Fit BM25 parameters on a corpus of documents."""
        if not documents:
            print("Warning: Empty document list provided to BM25 fit.")
            self.corpus = []
            self.n_docs = 0
            return
            
        self.corpus = documents
        self.n_docs = len(documents)
        
        # Calculate document frequencies and document lengths
        self.doc_freqs = Counter()
        self.doc_lengths = []
        
        for doc in documents:
            tokens = self.tokenize(doc)
            self.doc_lengths.append(len(tokens))
            self.doc_freqs.update(set(tokens))
        
        self.avg_doc_length = sum(self.doc_lengths) / self.n_docs if self.n_docs > 0 else 0
        
        # Calculate IDF values
        self.idf = {}
        for word, freq in self.doc_freqs.items():
            self.idf[word] = math.log((self.n_docs - freq + 0.5) / (freq + 0.5) + self.epsilon)
        
        self.vocab_size = len(self.doc_freqs)
    
    def _score_one(self, query_tokens: List[str], doc_idx: int) -> float:
        """Score a single document."""
        score = 0.0
        doc_length = self.doc_lengths[doc_idx]
        doc = self.corpus[doc_idx]
        doc_tokens = self.tokenize(doc)
        doc_token_freqs = Counter(doc_tokens)
        
        for token in query_tokens:
            if token in self.idf:
                # Get term frequency in document
                freq = doc_token_freqs.get(token, 0)
                # BM25 scoring formula
                numerator = self.idf[token] * freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
                score += numerator / denominator if denominator != 0 else 0
        
        return score
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """Search for most relevant documents for the query."""
        if not self.corpus or not query:
            return []
            
        query_tokens = self.tokenize(query)
        scores = []
        
        for i in range(self.n_docs):
            score = self._score_one(query_tokens, i)
            scores.append((i, score))
        
        # Sort by score in descending order
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores[:top_k]

class HydroidRetrievalEngine:
    """
    Advanced hybrid search engine combining dense and sparse retrieval.
    
    Features:
    - Vector search with FAISS
    - Keyword search with BM25
    - Section-based prioritization
    - Context-aware re-ranking
    - Debug logging for transparency
    """
    
    def __init__(self, 
                 vector_index_manager: VectorIndexManager, 
                 chunk_manifest: ChunkManifest,
                 debug_dir: str = "./retrieval_debug"):
        self.vector_index_manager = vector_index_manager
        self.chunk_manifest = chunk_manifest
        self.bm25 = SimpleBM25()
        self.debug_dir = debug_dir
        self.last_query_debug_data_for_ui = None
        
        # Create debug directory if it doesn't exist
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            
        # Initialize the BM25 engine with empty corpus
        # It will be populated with update_corpus
        self.update_corpus([])
        
    def update_corpus(self, chunks: List[DocumentChunk]) -> None:
        """
        Update the search corpus with new chunks.
        
        Args:
            chunks: List of DocumentChunk objects to index
        """
        # Extract text content for BM25
        text_corpus = [c.text for c in chunks if c.text]
        chunk_hashes = [c.metadata.content_hash for c in chunks if c.metadata.content_hash]
        
        print(f"HydroidEngine: Fitting BM25 with {len(text_corpus)} chunks.")
        self.bm25.fit(text_corpus)
        # Store the mapping from corpus index to chunk hash for BM25 results
        self.corpus_idx_to_hash = {i: h for i, h in enumerate(chunk_hashes)}
        self.hash_to_corpus_idx = {h: i for i, h in enumerate(chunk_hashes)}
        print(f"HydroidEngine: BM25 fit complete.")
    
    def search(self, 
              query: str, 
              top_k: int = 5, 
              semantic_weight: float = 0.7,
              keyword_weight: float = 0.3,
              use_fallback: bool = True,
              use_reranking: bool = True,
              debug_log: bool = True) -> List[Dict[str, Any]]:
        """
        Perform hybrid search using semantic and keyword methods.
        
        Args:
            query: Query string
            top_k: Number of results to return
            semantic_weight: Weight for semantic search results
            keyword_weight: Weight for keyword search results
            use_fallback: Whether to use fallback methods if primary methods fail
            use_reranking: Whether to apply context-aware re-ranking
            debug_log: Whether to save debug logs
            
        Returns:
            List of search results with scores and metadata
        """
        # Handle empty query
        if not query or not query.strip():
            return []
            
        try:
            search_start = time.time()
            
            # 1. Semantic search with FAISS
            vector_results = []
            try:
                # The vector_index_manager.search_index returns (faiss_id, distance, entry) tuples
                # We need to convert these to (score, content_hash) tuples for fusion
                raw_vector_results = self.vector_index_manager.search_index(query, top_k=top_k*2)
                
                # Convert the results format: (faiss_id, distance, entry) -> (score, hash)
                for result in raw_vector_results:
                    if isinstance(result, tuple):
                        if len(result) == 3:
                            # Handle (faiss_id, distance, entry) format
                            faiss_id, distance, entry = result
                            # Convert distance to similarity score (lower distance = higher similarity)
                            similarity_score = 1.0 / (1.0 + distance)
                            content_hash = entry.content_hash
                            vector_results.append((similarity_score, content_hash))
                        elif len(result) == 2:
                            # Handle (score, hash) format
                            vector_results.append(result)
                        else:
                            print(f"Warning: Unexpected vector result format with {len(result)} elements: {result}")
                    else:
                        print(f"Warning: Unexpected vector result type: {type(result)}")
            except Exception as e:
                print(f"Error in vector search: {e}")
                if not use_fallback:
                    raise
            
            # 2. Keyword search with BM25
            bm25_results = []
            try:
                bm25_results = self.bm25.search(query, top_k=top_k*2)  # Get more results for fusion
            except Exception as e:
                print(f"Error in BM25 search: {e}")
                if not use_fallback:
                    raise
                    
            # 3. Score normalization and fusion
            combined_results = self._fuse_results(
                query,
                vector_results, 
                bm25_results, 
                semantic_weight, 
                keyword_weight,
                top_k
            )
            
            # 4. Apply context-aware re-ranking if enabled
            if use_reranking and len(combined_results) > 1:
                combined_results = self._apply_reranking(query, combined_results)
            
            # 5. Format results with chunk objects
            search_results = self._format_search_results(combined_results)
            
            # 6. Save debug log
            search_end = time.time()
            if debug_log:
                self._save_debug_log(
                    query, 
                    vector_results, 
                    bm25_results, 
                    combined_results,
                    search_results,
                    search_end - search_start
                )
            
            return search_results
            
        except Exception as e:
            print(f"Error in HydroidRetrievalEngine.search: {e}")
            if debug_log:
                self._save_error_log(query, str(e))
            return []
            
    def _fuse_results(self, 
                     query: str,
                     vector_results: List[Tuple[float, str]], 
                     bm25_results: List[Tuple[int, float]], 
                     semantic_weight: float, 
                     keyword_weight: float,
                     top_k: int) -> List[Dict[str, Any]]:
        """
        Fuse vector and BM25 search results.
        
        Args:
            query: The search query
            vector_results: Results from vector search
            bm25_results: Results from BM25 search
            semantic_weight: Weight for semantic scores
            keyword_weight: Weight for keyword scores
            top_k: Maximum number of results to return
            
        Returns:
            List of combined results with normalized scores
        """
        # Convert to dictionaries for easier processing
        vector_dict = {}
        bm25_dict = {}
        
        # Process vector results - (score, chunk_hash)
        for result in vector_results:
            # Safely handle different return formats
            if isinstance(result, tuple) and len(result) == 2:
                score, chunk_hash = result
                vector_dict[chunk_hash] = score
            else:
                print(f"Warning: Unexpected vector result format: {result}")
            
        # Process BM25 results - (idx, score)
        for result in bm25_results:
            # Safely handle different return formats
            if isinstance(result, tuple) and len(result) == 2:
                idx, score = result
                if idx in self.corpus_idx_to_hash:
                    chunk_hash = self.corpus_idx_to_hash[idx]
                    bm25_dict[chunk_hash] = score
            else:
                print(f"Warning: Unexpected BM25 result format: {result}")
        
        # Combine all unique chunk hashes
        all_chunks = set(vector_dict.keys()) | set(bm25_dict.keys())
        
        # Skip empty results
        if not all_chunks:
            return []
        
        # Get max scores for normalization
        vector_scores = np.array(list(vector_dict.values())) if vector_dict else np.array([0.0])
        bm25_scores = np.array(list(bm25_dict.values())) if bm25_dict else np.array([0.0])
        
        # Check if semantic search produced useful results
        semantic_max = np.max(vector_scores) if len(vector_scores) > 0 else 0.0
        keyword_max = np.max(bm25_scores) if len(bm25_scores) > 0 else 0.0
        
        print(f"Max scores - Semantic: {semantic_max:.4f}, Keyword: {keyword_max:.4f}")
        
        # Dynamically adjust weights if semantic search fails but keyword search works
        original_semantic_weight = semantic_weight
        original_keyword_weight = keyword_weight
        
        # If semantic scores are all very low but we have good keyword matches,
        # adjust the weights to rely more on keyword search
        if semantic_max < 0.1 and keyword_max > 0.5:
            semantic_weight = 0.1
            keyword_weight = 0.9
            print(f"Adjusting weights due to poor semantic results: {original_semantic_weight:.2f}/{original_keyword_weight:.2f} -> {semantic_weight:.2f}/{keyword_weight:.2f}")
        
        # Normalize scores to [0, 1] range
        max_vector_score = np.max(vector_scores) if vector_scores.size > 0 else 1.0
        max_bm25_score = np.max(bm25_scores) if bm25_scores.size > 0 else 1.0
        
        # Avoid division by zero
        if max_vector_score == 0:
            max_vector_score = 1.0
        if max_bm25_score == 0:
            max_bm25_score = 1.0
        
        # Combine results with normalized scores
        combined_results = []
        
        for chunk_hash in all_chunks:
            # Get normalized scores
            semantic_score = vector_dict.get(chunk_hash, 0.0) / max_vector_score
            keyword_score = bm25_dict.get(chunk_hash, 0.0) / max_bm25_score
            
            # Apply weighted combination
            combined_score = (semantic_score * semantic_weight + 
                            keyword_score * keyword_weight)
            
            # Check if this is a high-priority section (summary, conclusion, etc.)
            boosted = False
            if chunk_hash in self.chunk_manifest.chunks:
                chunk = self.chunk_manifest.chunks[chunk_hash]
                if chunk.metadata.priority == "high":
                    combined_score *= 1.2  # Boost by 20%
                    boosted = True
            
            combined_results.append({
                'chunk_hash': chunk_hash,
                'combined_score': combined_score,
                'semantic_score': semantic_score,
                'keyword_score': keyword_score,
                'boosted': boosted
            })
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Return top-k results
        return combined_results[:top_k]
        
    def _apply_reranking(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rerank results to improve diversity and coherence.
        
        Args:
            query: The search query
            results: The combined search results
            
        Returns:
            Reranked results
        """
        if not results:
            return []
            
        # Get chunks for all results
        chunks_by_hash = {}
        for res in results:
            chunk = self.chunk_manifest.get_chunk(res['chunk_hash'])
            if chunk:
                chunks_by_hash[res['chunk_hash']] = chunk
                
        # Don't re-rank if we couldn't find all chunks
        if len(chunks_by_hash) != len(results):
            return results
        
        # Group chunks by document
        docs_to_chunks = defaultdict(list)
        for res in results:
            chunk = chunks_by_hash[res['chunk_hash']]
            docs_to_chunks[chunk.metadata.doc_name].append((res, chunk))
            
        # Sort chunks within each document by position
        for doc, chunks in docs_to_chunks.items():
            chunks.sort(key=lambda x: x[1].metadata.chunk_index)
            
        # Rerank to promote diversity and coherence
        reranked_results = []
        used_docs = set()
        used_chunks = set()
        
        # First pass: take one result from each document to promote diversity
        candidate_selection = sorted(results, key=lambda x: x['combined_score'], reverse=True)
        
        for res in candidate_selection:
            chunk = chunks_by_hash[res['chunk_hash']]
            doc_name = chunk.metadata.doc_name
            
            if doc_name not in used_docs:
                reranked_results.append(res)
                used_docs.add(doc_name)
                used_chunks.add(res['chunk_hash'])
                
                # Break if we have enough results
                if len(reranked_results) >= len(results):
                    break
        
        # Second pass: add remaining results by score
        for res in candidate_selection:
            if res['chunk_hash'] not in used_chunks:
                reranked_results.append(res)
                used_chunks.add(res['chunk_hash'])
                
                # Break if we have enough results
                if len(reranked_results) >= len(results):
                    break
        
        return reranked_results
        
    def _format_search_results(self, combined_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format the combined results with chunk data.
        
        Args:
            combined_results: The combined search results
            
        Returns:
            Formatted results with chunk data
        """
        if not combined_results:
            return []
            
        formatted_results = []
        
        for res in combined_results:
            chunk_hash = res['chunk_hash']
            chunk = self.chunk_manifest.get_chunk(chunk_hash)
            
            if not chunk:
                continue
                
            formatted_results.append({
                'chunk': chunk,
                'combined_score': res['combined_score'],
                'semantic_score': res['semantic_score'],
                'keyword_score': res['keyword_score'],
                'boosted': res['boosted'],
                'visualization_data': {
                    'score_breakdown': {
                        'semantic': res['semantic_score'],
                        'keyword': res['keyword_score'],
                        'combined': res['combined_score']
                    }
                }
            })
            
        return formatted_results
        
    def _save_debug_log(self, 
                       query: str,
                       vector_results: List[Tuple[float, str]],
                       bm25_results: List[Tuple[int, float]],
                       combined_results: List[Dict[str, Any]],
                       search_results: List[Dict[str, Any]],
                       execution_time: float) -> None:
        """
        Save debug information for the search.
        
        Args:
            query: The search query
            vector_results: Results from vector search
            bm25_results: Results from BM25 search
            combined_results: Results from fusion algorithm
            search_results: Formatted search results
            execution_time: Search execution time
        """
        if not self.debug_dir:
            return
            
        # Generate a unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        query_slug = "_".join(query.lower().split()[:3])  # First 3 words
        safe_slug = "".join([c if c.isalnum() else "_" for c in query_slug])
        
        debug_filename = os.path.join(self.debug_dir, f"query_{safe_slug}_{timestamp}.json")
        
        # Prepare debug info
        debug_info = {
            'timestamp': timestamp,
            'query': query,
            'execution_time': execution_time,
            'parameters': {
                'top_k': len(search_results),
                'semantic_weight': combined_results[0]['semantic_score'] / combined_results[0]['combined_score'] 
                                   if combined_results and combined_results[0]['combined_score'] > 0 else 0.7,
                'keyword_weight': combined_results[0]['keyword_score'] / combined_results[0]['combined_score']
                                  if combined_results and combined_results[0]['combined_score'] > 0 else 0.3
            },
            'vector_results': [{'score': float(score), 'hash': h} for score, h in vector_results[:5]],
            'bm25_results': [{'idx': int(idx), 'score': float(score)} for idx, score in bm25_results[:5]],
            'combined_results': [
                {
                    'hash': result['chunk_hash'],
                    'combined_score': float(result['combined_score']),
                    'semantic_score': float(result['semantic_score']),
                    'keyword_score': float(result['keyword_score']),
                    'boosted': result.get('boosted', False)
                } 
                for result in combined_results[:5]
            ],
            'final_results': [
                {
                    'chunk_hash': result['chunk'].metadata.content_hash,
                    'doc_name': result['chunk'].metadata.doc_name,
                    'section': result['chunk'].metadata.section_title or "No section",
                    'combined_score': float(result['combined_score']),
                    'semantic_score': float(result['semantic_score']),
                    'keyword_score': float(result['keyword_score']),
                    'boosted': result.get('boosted', False),
                    'chunk_type': result['chunk'].metadata.chunk_type,
                    'snippet': result['chunk'].text[:200] + "..." if len(result['chunk'].text) > 200 else result['chunk'].text
                }
                for result in search_results[:5]
            ]
        }
        
        # Save debug info
        with open(debug_filename, 'w') as f:
            json.dump(debug_info, f, indent=2)
            
        # Keep most recent debug data for UI display
        self.last_query_debug_data_for_ui = debug_info
            
    def _save_error_log(self, query: str, error_message: str) -> None:
        """
        Save error information for debugging.
        
        Args:
            query: The search query
            error_message: The error message
        """
        if not self.debug_dir:
            return
            
        # Generate a safe filename from the query
        safe_query = "".join(c if c.isalnum() else "_" for c in query[:15])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"error_{safe_query}_{timestamp}.json"
        filepath = os.path.join(self.debug_dir, filename)
        
        # Prepare error data
        error_data = {
            'query': query,
            'timestamp': timestamp,
            'error': error_message
        }
        
        # Write to file
        try:
            with open(filepath, 'w') as f:
                json.dump(error_data, f, indent=2)
            print(f"HydroidEngine: Error log saved to {filepath}")
        except Exception as e:
            print(f"Error saving error log: {e}")

# Example Usage (Conceptual - would be in RAGApp or similar)
if __name__ == '__main__':
    # This is just for conceptual illustration, actual setup will be in RAGApp
    print("Conceptual HydroidRetrievalEngine example run.")
    
    # --- Mock necessary components --- 
    class MockVIM:
        def search_index(self, query, top_k):
            print(f"MockVIM: Searching for '{query}' (top_k={top_k})")
            # (faiss_id, distance, IndexEntry(content_hash=hash, ...))
            # Let's return some dummy data where content_hash matters
            return [
                (0, 0.1, type('IndexEntry', (object,), {'content_hash': 'hash1', 'doc_name': 'docA', 'chunk_index':0})),
                (1, 0.5, type('IndexEntry', (object,), {'content_hash': 'hash2', 'doc_name': 'docA', 'chunk_index':1})),
                (2, 0.2, type('IndexEntry', (object,), {'content_hash': 'hash3', 'doc_name': 'docB', 'chunk_index':0})),
            ]

    class MockChunkManifest:
        def __init__(self):
            self.mock_chunks = {
                'hash1': DocumentChunk(text="Vector highly relevant text about apples and AI.", metadata=ChunkMetadata(content_hash='hash1', doc_name='docA')),
                'hash2': DocumentChunk(text="Some other text about bananas.", metadata=ChunkMetadata(content_hash='hash2', doc_name='docA')),
                'hash3': DocumentChunk(text="BM25 relevant text about apples and oranges.", metadata=ChunkMetadata(content_hash='hash3', doc_name='docB')),
                'hash4': DocumentChunk(text="Purely BM25 text with apples.", metadata=ChunkMetadata(content_hash='hash4', doc_name='docC'))
            }
        def get_chunk(self, content_hash: str) -> Optional[DocumentChunk]:
            return self.mock_chunks.get(content_hash)

    # --- Setup ---    
    mock_vim = MockVIM()
    mock_manifest = MockChunkManifest()
    all_test_chunks = list(mock_manifest.mock_chunks.values()) # Chunks for BM25

    engine = HydroidRetrievalEngine(vector_index_manager=mock_vim, chunk_manifest=mock_manifest, debug_dir="./engine_debug_logs")
    engine.update_corpus(all_test_chunks)

    # --- Test Search --- 
    test_query = "apples AI"
    results = engine.search(test_query, top_k=2, debug_log=True)

    print(f"\nSearch Results for '{test_query}':")
    if results:
        for result in results:
            print(f"  Score: {result['combined_score']:.4f} | Doc: {result['doc_name']} | Hash: {result['chunk'].metadata.content_hash}")
            print(f"    Text: {result['chunk'].text[:100]}...")
    else:
        print("  No results found.") 