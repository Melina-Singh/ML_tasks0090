"""
Main document retrieval system that orchestrates all components.
"""

import numpy as np
from typing import List, Dict, Optional, Union
from backend.text_processor import TextProcessor
from backend.embedding_models import TFIDFEmbedder, BaseEmbedder
from backend.distance_calculator import DistanceCalculator

class DocumentRetriever:
    """
    Main class for document embedding and retrieval operations.
    
    This class orchestrates text processing, embedding generation, and 
    similarity search across document chunks.
    """
    
    def __init__(self, 
                 embedder: Optional[BaseEmbedder] = None, 
                 chunk_size: int = 500, 
                 overlap: int = 50):
        """
        Initialize the document retriever.
        
        Args:
            embedder: Embedding model to use (default: TFIDFEmbedder)
            chunk_size: Maximum words per chunk
            overlap: Overlapping words between chunks
        """
        self.embedder = embedder or TFIDFEmbedder()
        self.processor = TextProcessor()
        self.calculator = DistanceCalculator()
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        # Document storage
        self.document_chunks = []
        self.embeddings = None
        self.original_document = None
        self.is_loaded = False
    
    def load_document(self, document_text: str) -> int:
        """
        Process and embed a document for retrieval.
        
        Args:
            document_text: Raw document content
            
        Returns:
            int: Number of chunks created
            
        Raises:
            ValueError: If document is empty or invalid
        """
        if not document_text or not document_text.strip():
            raise ValueError("Document text cannot be empty")
        
        # Store original document
        self.original_document = document_text
        
        # Process and chunk document
        processed_text = self.processor.preprocess(document_text)
        self.document_chunks = self.processor.chunk_text(
            processed_text, self.chunk_size, self.overlap
        )
        
        if not self.document_chunks:
            raise ValueError("No valid chunks created from document")
        
        # Create embeddings
        self.embeddings = self.embedder.fit_transform(self.document_chunks)
        self.is_loaded = True
        
        return len(self.document_chunks)
    
    def _get_query_embedding(self, query: str):
        """
        Get embedding for a search query.
        
        Args:
            query: Search query string
            
        Returns:
            Query embedding vector
        """
        if not self.is_loaded:
            raise ValueError("No document loaded. Call load_document() first.")
        
        processed_query = self.processor.preprocess(query)
        return self.embedder.transform([processed_query])
    
    def search(self, 
               query: str, 
               top_k: int = 3, 
               distance_metric: str = 'cosine',
               return_scores: bool = True) -> List[Dict]:
        """
        Search for most relevant document chunks.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            distance_metric: Distance metric to use
            return_scores: Whether to include similarity scores
            
        Returns:
            List of dictionaries containing search results
            
        Raises:
            ValueError: If no document loaded or invalid parameters
        """
        if not self.is_loaded:
            raise ValueError("No document loaded. Call load_document() first.")
        
        if top_k <= 0 or top_k > len(self.document_chunks):
            top_k = min(max(1, top_k), len(self.document_chunks))
        
        # Get query embedding
        query_embedding = self._get_query_embedding(query)
        
        # Calculate distances
        if distance_metric == 'cosine':
            distances, similarities = self.calculator.cosine_distance(
                query_embedding, self.embeddings
            )
            top_indices = np.argsort(distances)[:top_k]
            scores = similarities[top_indices] if return_scores else distances[top_indices]
            score_type = 'similarity' if return_scores else 'distance'
        else:
            distance_result = self.calculator.calculate_distance(
                distance_metric, query_embedding, self.embeddings
            )
            distances = distance_result if isinstance(distance_result, np.ndarray) else distance_result[0]
            top_indices = np.argsort(distances)[:top_k]
            scores = distances[top_indices]
            score_type = 'distance'
        
        return self._format_results(top_indices, scores, distance_metric, score_type)
    
    def _format_results(self, indices: np.ndarray, scores: np.ndarray, 
                       metric: str, score_type: str) -> List[Dict]:
        """
        Format search results into structured output.
        
        Args:
            indices: Chunk indices
            scores: Similarity/distance scores
            metric: Distance metric used
            score_type: Type of score (similarity or distance)
            
        Returns:
            Formatted results list
        """
        results = []
        for i, idx in enumerate(indices):
            results.append({
                'chunk_id': int(idx),
                'content': self.document_chunks[idx],
                'score': float(scores[i]),
                'score_type': score_type,
                'metric': metric,
                'chunk_length': len(self.document_chunks[idx].split())
            })
        return results
    
    def get_chunk_by_id(self, chunk_id: int) -> Optional[str]:
        """
        Get chunk content by ID.
        
        Args:
            chunk_id: Chunk identifier
            
        Returns:
            Chunk content or None if invalid ID
        """
        if not self.is_loaded or chunk_id < 0 or chunk_id >= len(self.document_chunks):
            return None
        return self.document_chunks[chunk_id]
    
    def get_document_stats(self) -> Dict:
        """
        Get statistics about the loaded document.
        
        Returns:
            Document statistics dictionary
        """
        if not self.is_loaded:
            return {"error": "No document loaded"}
        
        chunk_lengths = [len(chunk.split()) for chunk in self.document_chunks]
        original_stats = self.processor.get_text_stats(self.original_document)
        
        return {
            'original_document': original_stats,
            'chunks': {
                'total_chunks': len(self.document_chunks),
                'avg_chunk_length': np.mean(chunk_lengths),
                'min_chunk_length': min(chunk_lengths),
                'max_chunk_length': max(chunk_lengths),
                'chunk_size_setting': self.chunk_size,
                'overlap_setting': self.overlap
            },
            'embeddings': {
                'embedding_dimension': self.embeddings.shape[1],
                'embedding_type': type(self.embedder).__name__,
                'vocabulary_size': getattr(self.embedder, 'get_vocab_size', lambda: 'N/A')()
            }
        }
    
    def compare_metrics(self, query: str, top_k: int = 3) -> Dict[str, List[Dict]]:
        """
        Compare results across different distance metrics.
        
        Args:
            query: Search query
            top_k: Number of results per metric
            
        Returns:
            Dictionary with results for each metric
        """
        if not self.is_loaded:
            raise ValueError("No document loaded")
        
        metrics = self.calculator.get_available_metrics()
        comparison = {}
        
        for metric in metrics:
            try:
                results = self.search(query, top_k=top_k, distance_metric=metric)
                comparison[metric] = results
            except Exception as e:
                comparison[metric] = {"error": str(e)}
        
        return comparison
    
    def reset(self):
        """Reset the retriever state."""
        self.document_chunks = []
        self.embeddings = None
        self.original_document = None
        self.is_loaded = False