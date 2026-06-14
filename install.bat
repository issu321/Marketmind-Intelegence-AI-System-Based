@echo off
cls
echo ========================================
echo   MarketMind Installation Script
echo   Competitive Intelligence Platform
echo ========================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH.
    echo Please install Python 3.11 or higher from https://python.org
    pause
    exit /b 1
)

echo.
echo [1/5] Creating virtual environment...
if not exist "venv" (
    python -m venv venv
)

echo.
echo [2/5] Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo [3/5] Upgrading pip...
python -m pip install --upgrade pip

echo.
echo [4/5] Installing dependencies...
pip install -r requirements.txt

echo.
echo [5/5] Downloading NLTK data...
python -c "
import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('omw-1.4', quiet=True)
print('NLTK data downloaded successfully.')
"

echo.
echo ========================================
echo   Installation Complete!
echo ========================================
echo.
echo To start MarketMind:
echo   venv\Scripts\activate.bat
echo   python app.py
echo.
echo Then open: http://localhost:5000
echo.
echo Default admin credentials:
echo   Username: admin
echo   Password: admin123
echo.
pause
