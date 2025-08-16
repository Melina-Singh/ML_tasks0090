"""
Document loading utilities for various input sources including PDF support.
"""

import os
from typing import Dict, Optional, Tuple

class DocumentLoader:
    """Handles loading documents from various sources including PDF files."""
    
    @staticmethod
    def load_from_file(file_path: str, encoding: str = 'utf-8') -> str:
        """Load document content from a file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                return file.read()
        except UnicodeDecodeError as e:
            raise UnicodeDecodeError(f"Failed to decode file with {encoding} encoding: {e}")
    
    @staticmethod
    def load_from_pdf(file_path: str) -> str:
        """
        Extract text from PDF file using available PDF library.
        Tries multiple libraries in order of preference.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        # Try pdfplumber first (most accurate)
        try:
            import pdfplumber
            return DocumentLoader._extract_with_pdfplumber(file_path)
        except ImportError:
            pass
        
        # Try pymupdf (fastest)
        try:
            import fitz
            return DocumentLoader._extract_with_pymupdf(file_path)
        except ImportError:
            pass
        
        # Try PyPDF2 (most common)
        try:
            import PyPDF2
            return DocumentLoader._extract_with_pypdf2(file_path)
        except ImportError:
            pass
        
        raise ImportError(
            "No PDF processing library found. Install one of: "
            "'pip install pdfplumber' (recommended), "
            "'pip install pymupdf', or "
            "'pip install PyPDF2'"
        )
    
    @staticmethod
    def _extract_with_pdfplumber(file_path: str) -> str:
        """Extract text using pdfplumber (most accurate)."""
        import pdfplumber
        
        text = ""
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    except Exception as e:
                        print(f"Warning: Failed to extract text from page {page_num + 1}: {e}")
                        continue
        except Exception as e:
            raise Exception(f"Failed to process PDF with pdfplumber: {str(e)}")
        
        if not text.strip():
            raise ValueError("No text content found in PDF file")
        
        return text.strip()
    
    @staticmethod
    def _extract_with_pymupdf(file_path: str) -> str:
        """Extract text using pymupdf (fastest)."""
        import fitz
        
        text = ""
        try:
            pdf_document = fitz.open(file_path)
            for page_num in range(pdf_document.page_count):
                try:
                    page = pdf_document[page_num]
                    page_text = page.get_text()
                    if page_text:
                        text += page_text + "\n"
                except Exception as e:
                    print(f"Warning: Failed to extract text from page {page_num + 1}: {e}")
                    continue
            pdf_document.close()
        except Exception as e:
            raise Exception(f"Failed to process PDF with pymupdf: {str(e)}")
        
        if not text.strip():
            raise ValueError("No text content found in PDF file")
        
        return text.strip()
    
    @staticmethod
    def _extract_with_pypdf2(file_path: str) -> str:
        """Extract text using PyPDF2 (basic)."""
        import PyPDF2
        
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    except Exception as e:
                        print(f"Warning: Failed to extract text from page {page_num + 1}: {e}")
                        continue
        except Exception as e:
            raise Exception(f"Failed to process PDF with PyPDF2: {str(e)}")
        
        if not text.strip():
            raise ValueError("No text content found in PDF file")
        
        return text.strip()
    
    @staticmethod
    def load_from_text(text: str) -> str:
        """Load document from text string."""
        return text.strip() if text else ""
    
    @staticmethod
    def get_file_type(filename: str) -> str:
        """Get file type from filename extension."""
        if not filename:
            return 'unknown'
        
        extension = os.path.splitext(filename.lower())[1]
        file_types = {
            '.pdf': 'pdf',
            '.txt': 'text',
            '.md': 'markdown',
            '.markdown': 'markdown'
        }
        return file_types.get(extension, 'unknown')
    
    @staticmethod
    def load_from_file_auto(file_path: str, encoding: str = 'utf-8') -> Tuple[str, str]:
        """
        Automatically detect file type and extract content.
        
        Args:
            file_path: Path to the file
            encoding: Text encoding for non-PDF files
            
        Returns:
            Tuple of (content, file_type)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        filename = os.path.basename(file_path)
        file_type = DocumentLoader.get_file_type(filename)
        
        if file_type == 'pdf':
            content = DocumentLoader.load_from_pdf(file_path)
        elif file_type in ['text', 'markdown']:
            content = DocumentLoader.load_from_file(file_path, encoding)
        else:
            # Try to read as text file anyway
            try:
                content = DocumentLoader.load_from_file(file_path, encoding)
                file_type = 'text'
            except Exception:
                raise ValueError(f"Unsupported file type: {file_type}. Supported types: PDF, TXT, MD")
        
        return content, file_type
    
    @staticmethod
    def get_sample_documents() -> Dict[str, str]:
        """Get predefined sample documents for testing."""
        return {
            'ml_basics': """Machine learning is a subset of artificial intelligence that focuses on algorithms 
that can learn from data without being explicitly programmed. Deep learning uses 
neural networks with multiple layers to model and understand complex patterns in data.
Natural language processing enables computers to understand, interpret, and generate 
human language in a valuable way.

Supervised learning uses labeled training data to learn a mapping function from 
inputs to outputs. Common supervised learning tasks include classification and 
regression. Unsupervised learning finds hidden patterns in data without labeled 
examples. Clustering and dimensionality reduction are typical unsupervised tasks.

Reinforcement learning involves training agents to make decisions in an environment
by receiving rewards or penalties. This approach has been successful in game playing,
robotics, and autonomous systems. Feature engineering is crucial in traditional
machine learning, involving the selection and transformation of input variables.

Cross-validation helps assess model performance and prevent overfitting. Common
techniques include k-fold cross-validation and train-validation-test splits.
Hyperparameter tuning optimizes model configuration for better performance.""",
            
            'data_science': """Data science is an interdisciplinary field that combines statistics, programming, 
and domain expertise to extract meaningful insights from structured and unstructured data.
The data science process typically involves data collection, cleaning, exploration, 
modeling, and interpretation of results.

Data preprocessing is crucial and often takes 80% of a data scientist's time. This 
includes handling missing values, outlier detection, data normalization, and feature 
scaling. Statistical modeling provides frameworks for making predictions and inferences.

Exploratory data analysis (EDA) helps understand data patterns, distributions, and
relationships. Visualization tools like matplotlib, seaborn, and plotly are essential
for communicating findings. Database knowledge is important for data extraction using
SQL and NoSQL systems.

Big data technologies like Hadoop, Spark, and cloud platforms enable processing
large datasets. Machine learning integration automates pattern recognition and
predictive modeling. Business intelligence and domain knowledge ensure insights
are actionable and valuable.""",
            
            'ai_ethics': """Artificial intelligence ethics addresses the moral implications and societal impact 
of AI systems. As AI becomes more prevalent in decision-making processes, ensuring 
ethical development and deployment is crucial for maintaining public trust.

Algorithmic bias occurs when AI systems perpetuate or amplify existing societal 
inequalities. Transparency and explainability are essential for understanding how AI systems 
make decisions, especially in high-stakes applications like healthcare and finance.

Privacy protection involves safeguarding personal data used in AI training and
inference. Data governance frameworks ensure responsible collection, storage, and
usage of information. Accountability mechanisms determine responsibility when AI
systems cause harm or make errors.

Fairness considerations ensure AI systems treat all groups equitably. This includes
demographic parity, equalized odds, and individual fairness metrics. Robustness
testing evaluates AI performance under adversarial conditions and edge cases.

Human-AI collaboration focuses on augmenting rather than replacing human capabilities.
This includes designing interfaces that preserve human agency and decision-making
authority in critical situations."""
        }
    
    @staticmethod
    def get_document_info(content: str) -> Dict[str, any]:
        """Get basic information about a document."""
        if not content:
            return {
                'total_characters': 0,
                'total_words': 0,
                'total_lines': 0,
                'non_empty_lines': 0,
                'average_words_per_line': 0,
                'estimated_reading_time_minutes': 0,
            }
        
        lines = content.split('\n')
        words = content.split()
        non_empty_lines = [line for line in lines if line.strip()]
        
        return {
            'total_characters': len(content),
            'total_words': len(words),
            'total_lines': len(lines),
            'non_empty_lines': len(non_empty_lines),
            'average_words_per_line': len(words) / len(non_empty_lines) if non_empty_lines else 0,
            'estimated_reading_time_minutes': round(len(words) / 200, 1),  # 200 words per minute average
        }
    
    @staticmethod
    def validate_document(content: str, min_words: int = 10) -> bool:
        """Validate if document content is suitable for processing."""
        if not content or not content.strip():
            return False
        
        word_count = len(content.split())
        return word_count >= min_words
    
    @staticmethod
    def get_supported_formats() -> Dict[str, Dict[str, any]]:
        """Get information about supported file formats and their capabilities."""
        formats = {
            'pdf': {
                'extension': '.pdf',
                'description': 'Portable Document Format',
                'requires_library': True,
                'supported_libraries': ['pdfplumber', 'pymupdf', 'PyPDF2'],
                'recommended_library': 'pdfplumber'
            },
            'text': {
                'extension': '.txt',
                'description': 'Plain Text File',
                'requires_library': False,
                'supported_libraries': ['built-in'],
                'recommended_library': 'built-in'
            },
            'markdown': {
                'extension': '.md',
                'description': 'Markdown Document',
                'requires_library': False,
                'supported_libraries': ['built-in'],
                'recommended_library': 'built-in'
            }
        }
        
        # Check which PDF libraries are available
        pdf_libraries = []
        try:
            import pdfplumber
            pdf_libraries.append('pdfplumber')
        except ImportError:
            pass
        
        try:
            import fitz
            pdf_libraries.append('pymupdf')
        except ImportError:
            pass
        
        try:
            import PyPDF2
            pdf_libraries.append('PyPDF2')
        except ImportError:
            pass
        
        formats['pdf']['available_libraries'] = pdf_libraries
        formats['pdf']['is_available'] = len(pdf_libraries) > 0
        
        return formats