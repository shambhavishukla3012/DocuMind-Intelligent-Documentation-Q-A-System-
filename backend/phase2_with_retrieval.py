"""
Phase 2: RAG approach
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import ollama
import os
import numpy as np

app = Flask(__name__)
# CORS(app)
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Global variables
conversation_history = []
embedding_model = None
document_chunks = []
chunk_embeddings = []


def load_embedding_model():
    """
    Load the sentence transformer model for creating embeddings.
    'all-MiniLM-L6-v2'  
    """
    print("Loading embedding model (this may take a moment)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Embedding model loaded")
    return model


def load_documents():
    """
    Load all documentation files from data/documentation/
    
    Returns a list of documents with their content and metadata.
    """
    docs_path = os.path.join('..', 'data', 'documentation')
    documents = []
    
    print(f"\nLoading documents from: {docs_path}")
    
    if not os.path.exists(docs_path):
        print(f"Directory not found: {docs_path}")
        return documents
    
    for filename in os.listdir(docs_path):
        if filename.endswith('.txt') or filename.endswith('.md'):
            filepath = os.path.join(docs_path, filename)
            print(f"Loading: {filename}")
            
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                documents.append({
                    'filename': filename,
                    'content': content,
                    'path': filepath
                })
    
    print(f"Loaded {len(documents)} documents")
    return documents


def chunk_text(text, chunk_size=500, overlap=50):
    """
    Split text into smaller chunks with overlap.

    Args:
        text: The text to chunk
        chunk_size: Number of words per chunk
        overlap: Number of words to overlap between chunks
    
    Returns:
        List of text chunks
    """
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)
    
    return chunks


def create_embeddings(chunks, model):
    """
    Convert text chunks into vector embeddings.
    Args:
        chunks: List of text chunks
        model: Sentence transformer model
    
    Returns:
        Numpy array of embeddings
    """
    print(f"Creating embeddings for {len(chunks)} chunks...")
    embeddings = model.encode(chunks, show_progress_bar=True)
    print("Embeddings created!")
    return embeddings


def cosine_similarity(vec1, vec2):
    """
    Calculate similarity between two vectors.
    
    Result is between 0 and 1:
    - 1.0 = identical
    - 0.0 = completely different
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)


def search_documents(query, top_k=10):
    """
    Search for relevant document chunks.    
    Args:
        query: User's question
        top_k: How many chunks to return
    
    Returns:
        List of (chunk_text, similarity_score) tuples
    """
    if not document_chunks:
        return []
    
    # Convert query to embedding
    query_embedding = embedding_model.encode([query])[0]
    
    # Calculate similarity with all chunks
    similarities = []
    for i, chunk_embedding in enumerate(chunk_embeddings):
        similarity = cosine_similarity(query_embedding, chunk_embedding)
        similarities.append((document_chunks[i], similarity))
    
    # Sort by similarity (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Return top K
    return similarities[:top_k]


def simple_rerank(query, chunks):
    """
    Re-rank chunks for better relevance.

    Args:
        query: User's question
        chunks: List of (chunk, similarity_score) tuples
    
    Returns:
        Re-ranked list of chunks
    """
    query_words = set(query.lower().split())
    
    reranked = []
    for chunk_text, sim_score in chunks:
        chunk_words = set(chunk_text.lower().split())
        
        # Count matching words
        word_overlap = len(query_words.intersection(chunk_words))
        
        # Combine similarity score with word overlap
        combined_score = sim_score + (word_overlap * 0.01)
        
        reranked.append((chunk_text, combined_score))
    
    # Sort by new score
    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked


def build_prompt_with_context(query, context_chunks, top_n=3):
    """
    Build a prompt that includes retrieved context.

    Args:
        query: User's question
        context_chunks: Retrieved chunks
        top_n: How many chunks to include
    
    Returns:
        Formatted prompt string
    """
    # top N chunks
    top_chunks = context_chunks[:top_n]
    
    # Build context section
    context = "\n\n".join([
        f"[Source {i+1}]: {chunk[0][:500]}..."  # Limit chunk length
        for i, chunk in enumerate(top_chunks)
    ])
    
    # Build full prompt
    prompt = f"""You are a helpful assistant. Answer the user's question based on the provided documentation.

        DOCUMENTATION:
        {context}

        USER QUESTION: {query}

        Please provide a clear, accurate answer based on the documentation above. If the documentation doesn't contain the answer, say so."""

    return prompt


# Initialize on startup
print("=" * 50)
print("Phase 2: Chatbot with Retrieval & Re-ranking")
print("=" * 50)

print("\nInitializing...")

# Load embedding model
embedding_model = load_embedding_model()

# Load documents
documents = load_documents()

# Create chunks from all documents
print("\nChunking documents...")
all_chunks = []
for doc in documents:
    chunks = chunk_text(doc['content'], chunk_size=500, overlap=50)
    print(f"  {doc['filename']}: {len(chunks)} chunks")
    all_chunks.extend(chunks)

document_chunks = all_chunks
print(f"Total chunks: {len(document_chunks)}")

# Create embeddings for all chunks
if document_chunks:
    chunk_embeddings = create_embeddings(document_chunks, embedding_model)
    print(f"Created {len(chunk_embeddings)} embeddings")
else:
    print("No documents loaded - chatbot will work without retrieval")

print("\n" + "=" * 50)
print("Ready! Backend running with RAG enabled!")
print("=" * 50)


@app.route('/health', methods=['GET'])
def health_check():
    """Check if service is running"""
    return jsonify({
        'status': 'healthy',
        'message': 'Phase 2 backend running',
        'documents_loaded': len(document_chunks),
        'rag_enabled': len(document_chunks) > 0
    }), 200


@app.route('/chat', methods=['POST'])
def chat():
    """
    Handle chat with RAG.
    
    Process:
    1. Get user question
    2. Search documents for relevant chunks
    3. Re-rank results
    4. Build prompt with context
    5. Send to LLM
    6. Return response
    """
    global conversation_history
    
    try:
        data = request.json
        user_message = data.get('message', '')
        reset = data.get('reset', False)
        
        if reset:
            conversation_history = []
            return jsonify({
                'response': 'Conversation reset!',
                'error': None,
                'sources_used': 0
            }), 200
        
        if not user_message:
            return jsonify({
                'response': None,
                'error': 'No message provided'
            }), 400
        
        # If documents, use RAG
        if document_chunks:
            print(f"\nSearching for: {user_message}")
            
            # 1. Search for relevant chunks
            search_results = search_documents(user_message, top_k=10)
            print(f"Found {len(search_results)} similar chunks")
            
            # 2. Re-rank for relevance
            reranked = simple_rerank(user_message, search_results)
            print(f"Re-ranked results")
            
            # 3. Build prompt with context
            prompt_with_context = build_prompt_with_context(
                user_message, 
                reranked, 
                top_n=3
            )
            
            # 4. Send to LLM
            response = ollama.chat(
                model='llama3.2:3b',
                messages=[{
                    'role': 'user',
                    'content': prompt_with_context
                }]
            )
            
            assistant_message = response['message']['content']
            
            return jsonify({
                'response': assistant_message,
                'error': None,
                'sources_used': 3,  # used top 3 chunks
                'rag_enabled': True
            }), 200
        
        else:
            # No documents - fall back to regular chat
            conversation_history.append({
                'role': 'user',
                'content': user_message
            })
            
            response = ollama.chat(
                model='llama3.2:3b',
                messages=conversation_history
            )
            
            assistant_message = response['message']['content']
            
            conversation_history.append({
                'role': 'assistant',
                'content': assistant_message
            })
            
            return jsonify({
                'response': assistant_message,
                'error': None,
                'sources_used': 0,
                'rag_enabled': False
            }), 200
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            'response': None,
            'error': f'Error: {str(e)}'
        }), 500


@app.route('/clear', methods=['POST'])
def clear_history():
    """Clear conversation history"""
    global conversation_history
    conversation_history = []
    return jsonify({'message': 'History cleared'}), 200


if __name__ == '__main__':
    app.run(debug=True, port=5001)
