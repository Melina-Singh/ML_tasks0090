"""
Flask backend server to connect the web frontend with Python modules.
"""

from flask import Flask, request, jsonify, render_template_string, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import json
import os
import tempfile

# Import our modules
from backend.document_loader import DocumentLoader
from backend.document_retriever import DocumentRetriever
from backend.embedding_models import TFIDFEmbedder, CountEmbedder

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Configuration for file uploads
UPLOAD_FOLDER = tempfile.gettempdir()
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'md'}

app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Global retriever instance
retriever = None

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Serve the web interface"""
    # You can serve the HTML file here or use render_template
    return send_from_directory('frontend', 'index.html')

@app.route('/api/upload-file', methods=['POST'])
def upload_file():
    """Handle file upload and text extraction"""
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file type
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Only PDF, TXT, and MD files are allowed'}), 400
        
        # Check file size (additional check beyond Flask's MAX_CONTENT_LENGTH)
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)  # Reset file pointer
        
        if file_size > MAX_CONTENT_LENGTH:
            return jsonify({'error': f'File too large. Maximum size is {MAX_CONTENT_LENGTH // (1024*1024)}MB'}), 400
        
        # Secure filename and save temporarily
        filename = secure_filename(file.filename)
        if not filename:
            return jsonify({'error': 'Invalid filename'}), 400
            
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        
        try:
            file.save(file_path)
            
            # Extract text based on file type
            content, file_type = DocumentLoader.load_from_file_auto(file_path)
            
            # Validate extracted content
            if not DocumentLoader.validate_document(content, min_words=10):
                return jsonify({'error': 'Document contains insufficient text content (minimum 10 words required)'}), 400
            
            # Get document info
            doc_info = DocumentLoader.get_document_info(content)
            
            # Clean up temporary file
            if os.path.exists(file_path):
                os.remove(file_path)
            
            return jsonify({
                'success': True,
                'text': content,
                'filename': filename,
                'file_type': file_type,
                'word_count': doc_info['total_words'],
                'character_count': doc_info['total_characters'],
                'file_size': file_size,
                'reading_time': doc_info['estimated_reading_time_minutes'],
                'message': f'File "{filename}" processed successfully'
            })
            
        except Exception as e:
            # Clean up temporary file on error
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({'error': f'File processing failed: {str(e)}'}), 500
            
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/api/load-document', methods=['POST'])
def load_document():
    """Load and process document"""
    global retriever
    
    try:
        data = request.json
        document_text = data.get('document_text', '')
        embedding_model = data.get('embedding_model', 'tfidf')
        chunk_size = data.get('chunk_size', 500)
        overlap = data.get('overlap', 50)
        
        if not document_text.strip():
            return jsonify({'error': 'Document text is required'}), 400
        
        # Choose embedding model
        if embedding_model == 'count':
            embedder = CountEmbedder()
        else:
            embedder = TFIDFEmbedder()
        
        # Initialize retriever
        retriever = DocumentRetriever(
            embedder=embedder,
            chunk_size=chunk_size,
            overlap=overlap
        )
        
        # Load document
        num_chunks = retriever.load_document(document_text)
        
        # Get document stats
        stats = retriever.get_document_stats()
        
        return jsonify({
            'success': True,
            'num_chunks': num_chunks,
            'stats': stats,
            'message': f'Document processed successfully with {num_chunks} chunks'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/search', methods=['POST'])
def search_document():
    """Search document with specified parameters"""
    global retriever
    
    if retriever is None or not retriever.is_loaded:
        return jsonify({'error': 'No document loaded'}), 400
    
    try:
        data = request.json
        query = data.get('query', '')
        distance_metric = data.get('distance_metric', 'cosine')
        top_k = data.get('top_k', 3)
        
        if not query.strip():
            return jsonify({'error': 'Query is required'}), 400
        
        # Perform search
        results = retriever.search(
            query=query,
            top_k=top_k,
            distance_metric=distance_metric
        )
        
        return jsonify({
            'success': True,
            'results': results,
            'query': query,
            'metric': distance_metric
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/compare-metrics', methods=['POST'])
def compare_metrics():
    """Compare all distance metrics for a query"""
    global retriever
    
    if retriever is None or not retriever.is_loaded:
        return jsonify({'error': 'No document loaded'}), 400
    
    try:
        data = request.json
        query = data.get('query', '')
        top_k = data.get('top_k', 1)
        
        if not query.strip():
            return jsonify({'error': 'Query is required'}), 400
        
        # Compare metrics
        comparison = retriever.compare_metrics(query, top_k=top_k)
        
        return jsonify({
            'success': True,
            'comparison': comparison,
            'query': query
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/sample-documents', methods=['GET'])
def get_sample_documents():
    """Get available sample documents"""
    try:
        samples = DocumentLoader.get_sample_documents()
        return jsonify({
            'success': True,
            'documents': samples
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint with PDF support detection"""
    pdf_support = False
    pdf_library = None
    
    # Check for PDF processing libraries
    try:
        import pdfplumber
        pdf_support = True
        pdf_library = 'pdfplumber'
    except ImportError:
        try:
            import PyPDF2
            pdf_support = True
            pdf_library = 'PyPDF2'
        except ImportError:
            try:
                import fitz  # pymupdf
                pdf_support = True
                pdf_library = 'pymupdf'
            except ImportError:
                pass
    
    return jsonify({
        'status': 'healthy',
        'service': 'Document Retrieval API',
        'document_loaded': retriever is not None and retriever.is_loaded,
        'pdf_support': pdf_support,
        'pdf_library': pdf_library,
        'supported_formats': ['PDF', 'TXT', 'MD'] if pdf_support else ['TXT', 'MD'],
        'max_file_size_mb': MAX_CONTENT_LENGTH // (1024 * 1024),
        'upload_folder': UPLOAD_FOLDER
    })

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({
        'error': f'File too large. Maximum size is {MAX_CONTENT_LENGTH // (1024*1024)}MB'
    }), 413

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors"""
    return jsonify({
        'error': 'Internal server error occurred'
    }), 500

if __name__ == '__main__':
    print("🚀 Starting Document Retrieval API Server...")
    print("📡 Web interface available at: http://localhost:5000")
    print("🔧 API endpoints:")
    print("   POST /api/upload-file")
    print("   POST /api/load-document")
    print("   POST /api/search")
    print("   POST /api/compare-metrics")
    print("   GET  /api/sample-documents")
    print("   GET  /health")
    
    # Check PDF support on startup
    try:
        import pdfplumber
        print("✅ PDF support available (pdfplumber)")
    except ImportError:
        try:
            import PyPDF2
            print("⚠️  PDF support available (PyPDF2 - less accurate)")
        except ImportError:
            try:
                import fitz
                print("✅ PDF support available (pymupdf)")
            except ImportError:
                print("❌ PDF support not available - install: pip install pdfplumber")
    
    app.run(debug=True, port=5000)