"""
Distance calculation utilities for comparing embeddings.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.spatial.distance import cityblock

from typing import Tuple, Union

class DistanceCalculator:
    """Calculates various distance metrics between query and document embeddings."""
    
    @staticmethod
    def cosine_distance(query_embedding, doc_embeddings) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate cosine distance (1 - cosine similarity).
        
        Args:
            query_embedding: Query vector
            doc_embeddings: Document embedding matrix
            
        Returns:
            tuple: (distances, similarities) arrays
        """
        similarities = cosine_similarity(query_embedding, doc_embeddings).flatten()
        distances = 1 - similarities
        return distances, similarities
    
    @staticmethod
    def euclidean_distance(query_embedding, doc_embeddings) -> np.ndarray:
        """
        Calculate Euclidean (L2) distance.
        
        Args:
            query_embedding: Query vector
            doc_embeddings: Document embedding matrix
            
        Returns:
            np.ndarray: Distance array
        """
        return euclidean_distances(query_embedding, doc_embeddings).flatten()
    
    @staticmethod
    def cityblock_distance(query_embedding, doc_embeddings) -> np.ndarray:
        """
        Calculate cityblock (L1) distance.
        
        Args:
            query_embedding: Query vector (sparse matrix)
            doc_embeddings: Document embedding matrix (sparse matrix)
            
        Returns:
            np.ndarray: Distance array
        """
        query_dense = query_embedding.toarray()[0]
        embeddings_dense = doc_embeddings.toarray()
        
        distances = [cityblock(query_dense, chunk_embedding) 
                    for chunk_embedding in embeddings_dense]
        return np.array(distances)
    
    @staticmethod
    def jaccard_distance(query_embedding, doc_embeddings, threshold=0.01) -> np.ndarray:
        """
        Calculate Jaccard distance for binary/sparse vectors.
        
        Args:
            query_embedding: Query vector
            doc_embeddings: Document embedding matrix
            threshold: Threshold for binarization
            
        Returns:
            np.ndarray: Distance array
        """
        # Binarize embeddings
        query_binary = (query_embedding.toarray() > threshold).astype(int)[0]
        doc_binary = (doc_embeddings.toarray() > threshold).astype(int)
        
        distances = []
        for doc_vec in doc_binary:
            intersection = np.sum(query_binary & doc_vec)
            union = np.sum(query_binary | doc_vec)
            jaccard_sim = intersection / union if union > 0 else 0
            distances.append(1 - jaccard_sim)
        
        return np.array(distances)
    
    @classmethod
    def get_available_metrics(cls):
        """Get list of available distance metrics."""
        return ['cosine', 'euclidean', 'cityblock', 'jaccard']
    
    @classmethod
    def calculate_distance(cls, metric: str, query_embedding, doc_embeddings) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Calculate distance using specified metric.
        
        Args:
            metric: Distance metric name
            query_embedding: Query vector
            doc_embeddings: Document embedding matrix
            
        Returns:
            Distance array or (distance, similarity) tuple for cosine
        """
        if metric == 'cosine':
            return cls.cosine_distance(query_embedding, doc_embeddings)
        elif metric == 'euclidean':
            return cls.euclidean_distance(query_embedding, doc_embeddings)
        elif metric == 'cityblock':
            return cls.cityblock_distance(query_embedding, doc_embeddings)
        elif metric == 'jaccard':
            return cls.jaccard_distance(query_embedding, doc_embeddings)
        else:
            raise ValueError(f"Unsupported metric: {metric}. Available: {cls.get_available_metrics()}")
    
    @staticmethod
    def normalize_scores(scores: np.ndarray, method='min_max') -> np.ndarray:
        """
        Normalize scores to [0, 1] range.
        
        Args:
            scores: Raw scores
            method: Normalization method ('min_max' or 'z_score')
            
        Returns:
            np.ndarray: Normalized scores
        """
        if method == 'min_max':
            min_score, max_score = scores.min(), scores.max()
            if max_score == min_score:
                return np.ones_like(scores)
            return (scores - min_score) / (max_score - min_score)
        elif method == 'z_score':
            mean_score, std_score = scores.mean(), scores.std()
            if std_score == 0:
                return np.zeros_like(scores)
            return (scores - mean_score) / std_score
        else:
            raise ValueError("Method must be 'min_max' or 'z_score'")