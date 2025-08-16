"""
Text processing utilities for document preprocessing and chunking.
"""

import re

class TextProcessor:
    """Handles text preprocessing and document chunking operations."""
    
    @staticmethod
    def preprocess(text):
        """
        Clean and normalize text by removing special characters and converting to lowercase.
        
        Args:
            text (str): Raw input text
            
        Returns:
            str: Cleaned and normalized text
        """
        # Remove non-alphabetic characters except spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        # Remove extra whitespaces
        return ' '.join(text.split())
    
    @staticmethod
    def chunk_text(text, chunk_size=500, overlap=50):
        """
        Split text into overlapping chunks to preserve context at boundaries.
        
        Args:
            text (str): Input text to chunk
            chunk_size (int): Maximum words per chunk
            overlap (int): Number of overlapping words between chunks
            
        Returns:
            list: List of text chunks
        """
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if len(chunk.strip()) > 0:
                chunks.append(chunk)
        
        return chunks
    
    @staticmethod
    def get_text_stats(text):
        """
        Get basic statistics about the text.
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Text statistics
        """
        words = text.split()
        return {
            'characters': len(text),
            'words': len(words),
            'sentences': len([s for s in text.split('.') if s.strip()]),
            'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0
        }