"""
Phase 1: Basic Chatbot Backend
================================
Simple Flask API that connects to Ollama LLM.

Endpoints:
- POST /chat - Send message, get response
- GET /health - Check if service is running
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import ollama

app = Flask(__name__)
CORS(app)  # Allow frontend to connect

# Store conversation history in memory
conversation_history = []


@app.route('/health', methods=['GET'])
def health_check():
    """Check if service is running and Ollama is accessible"""
    try:
        # Try to connect to Ollama
        ollama.list()
        return jsonify({
            'status': 'healthy',
            'message': 'Backend is running and Ollama is accessible'
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'message': f'Error: {str(e)}'
        }), 500


@app.route('/chat', methods=['POST'])
def chat():
    """
    Handle chat messages
    
    Request body:
    {
        "message": "User's message",
        "reset": false  (optional - clear history)
    }
    
    Response:
    {
        "response": "LLM's response",
        "error": null
    }
    """
    global conversation_history
    
    try:
        data = request.json
        user_message = data.get('message', '')
        reset = data.get('reset', False)
        
        # Reset conversation if requested
        if reset:
            conversation_history = []
            return jsonify({
                'response': 'Conversation reset!',
                'error': None
            }), 200
        
        if not user_message:
            return jsonify({
                'response': None,
                'error': 'No message provided'
            }), 400
        
        # Add user message to history
        conversation_history.append({
            'role': 'user',
            'content': user_message
        })
        
        # Get response from Ollama
        response = ollama.chat(
            model='llama3.2:3b',
            messages=conversation_history
        )
        
        assistant_message = response['message']['content']
        
        # Add assistant response to history
        conversation_history.append({
            'role': 'assistant',
            'content': assistant_message
        })
        
        return jsonify({
            'response': assistant_message,
            'error': None
        }), 200
        
    except Exception as e:
        return jsonify({
            'response': None,
            'error': f'Error: {str(e)}'
        }), 500


@app.route('/clear', methods=['POST'])
def clear_history():
    """Clear conversation history"""
    global conversation_history
    conversation_history = []
    return jsonify({
        'message': 'History cleared',
        'error': None
    }), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)