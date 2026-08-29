@echo off
REM ==============================================================================
REM FAKEBUSTER — Windows Automated Production Startup Script
REM AI-Powered Fake News Detection System ("Bust the Fake. Find the Facts.")
REM ==============================================================================

echo ----------------------------------------------------------------------
echo  FAKEBUSTER: AI-Powered Fake News Detection System
echo  "Bust the Fake. Find the Facts."
echo ----------------------------------------------------------------------

python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not found on PATH. Please install Python 3.8+
    pause
    exit /b 1
)

if not exist "venv" (
    echo [INFO] Creating Python virtual environment...
    python -m venv venv
)

echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

echo [INFO] Installing required dependencies...
pip install -q -r requirements.txt

echo [INFO] Verifying NLTK corpora (stopwords, wordnet)...
python -c "import nltk; nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True)"

echo [INFO] Starting FAKEBUSTER server on http://localhost:5000...
python app.py
pause
