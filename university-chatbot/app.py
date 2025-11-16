# Flask API Backend for University Chatbot
# This version works without a database file.

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import logging
from datetime import datetime
import os

# Import the main chatbot module
from chatbot_core import UniversityChatbot
from dataset import LOCATIONS_DATA # Import data for API endpoints

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize chatbot
logger.info("Initializing University Chatbot (in-memory mode)...")
chatbot = UniversityChatbot()
logger.info("Chatbot initialized successfully!")


@app.route('/chat', methods=['POST'])
def chat():
    """Main chat endpoint"""
    try:
        data = request.json
        if not data or 'message' not in data:
            return jsonify({'error': 'No message provided', 'status': 'error'}), 400
        
        user_message = data['message']
        logger.info(f"Received query: {user_message}")
        
        bot_response_data = chatbot.get_response(user_message)
        
        api_response = {
            'status': 'success',
            'response': bot_response_data['response'],
            'timestamp': datetime.now().isoformat(),
            'data': bot_response_data.get('data')
        }
        return jsonify(api_response)
    
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'error': 'Failed to process request', 'message': str(e)}), 500

@app.route('/static/<path:path>')
def send_static(path):
    """Serves static files, specifically for PDFs."""
    return send_from_directory('static', path)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to confirm the server is running."""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.route('/locations', methods=['GET'])
def get_locations():
    """Endpoint to get a list of all known locations from memory."""
    try:
        location_list = [{'name': loc[0], 'building': loc[1], 'floor': loc[2], 'hours': loc[3]} for loc in LOCATIONS_DATA]
        return jsonify({'status': 'success', 'locations': location_list})
    except Exception as e:
        logger.error(f"Error fetching locations: {str(e)}")
        return jsonify({'status': 'error', 'error': 'Failed to fetch locations'}), 500

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors for unknown routes."""
    return jsonify({'status': 'error', 'error': 'Endpoint not found'}), 404


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=True)