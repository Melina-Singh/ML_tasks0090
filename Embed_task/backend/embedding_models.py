"""
Embedding model implementations with abstract base class for extensibility.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from abc import ABC, abstractmethod

class BaseEmbedder(ABC):
    """Abstract base class for all embedding models."""
    
    @abstractmethod
    def fit(self, texts):
        """
        Fit the embedder on a collection of texts.
        
        Args:
            texts (list): List of text documents
            
        Returns:
            self: Returns the fitted embedder
        """
        pass
    
    @abstractmethod
    def transform(self, texts):
        """
        Transform texts to embeddings.
        
        Args:
            texts (list): List of text documents
            
        Returns:
            array-like: Embedding vectors
        """
        pass
    
    @abstractmethod
    def fit_transform(self, texts):
        """
        Fit embedder and transform texts in one step.
        
        Args:
            texts (list): List of text documents
            
        Returns:
            array-like: Embedding vectors
        """
        pass

class TFIDFEmbedder(BaseEmbedder):
    """TF-IDF based text embedding model."""
    
    def __init__(self, max_features=1000, ngram_range=(1, 2), min_df=1):
        """
        Initialize TF-IDF embedder.
        
        Args:
            max_features (int): Maximum number of features
            ngram_range (tuple): Range of n-grams to consider
            min_df (int): Minimum document frequency
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=ngram_range,
            min_df=min_df
        )
        self.is_fitted = False
    
    def fit(self, texts):
        """Fit the TF-IDF vectorizer."""
        self.vectorizer.fit(texts)
        self.is_fitted = True
        return self
    
    def transform(self, texts):
        """Transform texts to TF-IDF vectors."""
        if not self.is_fitted:
            raise ValueError("Embedder must be fitted before transform")
        return self.vectorizer.transform(texts)
    
    def fit_transform(self, texts):
        """Fit and transform in one step."""
        embeddings = self.vectorizer.fit_transform(texts)
        self.is_fitted = True
        return embeddings
    
    def get_feature_names(self):
        """Get feature names from the fitted vectorizer."""
        if not self.is_fitted:
            raise ValueError("Embedder must be fitted first")
        return self.vectorizer.get_feature_names_out()
    
    def get_vocab_size(self):
        """Get vocabulary size."""
        if not self.is_fitted:
            return 0
        return len(self.vectorizer.vocabulary_)

class CountEmbedder(BaseEmbedder):
    """Simple count-based embedding model (alternative to TF-IDF)."""
    
    def __init__(self, max_features=1000):
        """Initialize count embedder."""
        from sklearn.feature_extraction.text import CountVectorizer
        self.vectorizer = CountVectorizer(
            max_features=max_features,
            stop_words='english'
        )
        self.is_fitted = False
    
    def fit(self, texts):
        """Fit the count vectorizer."""
        self.vectorizer.fit(texts)
        self.is_fitted = True
        return self
    
    def transform(self, texts):
        """Transform texts to count vectors."""
        if not self.is_fitted:
            raise ValueError("Embedder must be fitted before transform")
        return self.vectorizer.transform(texts)
    
    def fit_transform(self, texts):
        """Fit and transform in one step."""
        embeddings = self.vectorizer.fit_transform(texts)
        self.is_fitted = True
        return embeddings