#!/usr/bin/env bash
# ==============================================================================
# FAKEBUSTER — Automated Production Startup Script
# AI-Powered Fake News Detection System ("Bust the Fake. Find the Facts.")
# ==============================================================================

set -e

echo "🛡️  ================================================================"
echo "🛡️  Starting FAKEBUSTER: AI-Powered Fake News Detection System"
echo "🛡️  ================================================================"

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but was not found on PATH."
    exit 1
fi

# Set up virtual environment if not present
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment (.venv)..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "📥 Checking and installing project requirements..."
pip install -q -r requirements.txt

# Download required NLTK corpora
echo "🧠 Verifying NLTK English stopwords and WordNet corpora..."
python3 -c "import nltk; nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True)"

# Launch Flask Web Server
echo "🚀 Launching FAKEBUSTER Flask Microservice on http://127.0.0.1:5000"
export FLASK_APP=app.py
export FLASK_ENV=development
python3 app.py
