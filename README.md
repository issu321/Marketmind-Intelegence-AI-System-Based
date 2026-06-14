---
title: MarketMind
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 5000
app_file: app.py
pinned: false
---

# MarketMind - Competitive Intelligence and Trend Prediction Platform

An enterprise-grade, AI-powered platform for market analytics, demand forecasting, competitive intelligence, and strategic business insights.

## Features

### Machine Learning Forecasting Engine
- **Multi-Model Support**: Random Forest, XGBoost, LightGBM, CatBoost, Gradient Boosting
- **Auto-Model Selection**: Automatically picks the best model using cross-validation
- **Forecast Windows**: 30, 90, 180, 365 days
- **Confidence Intervals**: Statistical bounds on predictions
- **Backtesting**: Historical validation of model performance

### Competitor Intelligence Center
- Competitor scoring (Overall, Threat, Growth, Innovation, Market Position)
- SWOT analysis generation
- Market concentration analysis (HHI)
- Strategic landscape insights

### Consumer Intelligence Engine
- Sentiment analysis (Positive, Negative, Neutral)
- Brand health scoring
- Emotion breakdown (Joy, Anger, Trust, Fear, etc.)
- Topic extraction from feedback

### Business Opportunity Detection
- Automated opportunity identification
- Revenue potential estimation
- Market readiness scoring
- Risk level assessment

### Scenario Simulation Lab
- Price change simulation
- Marketing spend impact
- Demand shift modeling
- Competitor move analysis
- New product launch simulation

### Executive Command Center
- Business health dashboard
- Strategic recommendations
- Unified KPI overview

### Advanced Visualization
- Interactive Plotly charts
- Correlation heatmaps
- Forecast charts with confidence bands
- Comparison visualizations

### Report Center
- PDF, HTML, CSV, JSON reports
- Executive summaries
- Key findings and recommendations

### Database Admin Center
- User management
- Dataset monitoring
- Activity logs
- Login history
- Database health metrics
- CSV export for all tables

## Technology Stack

- **Backend**: Python 3.11+, Flask, SQLAlchemy
- **Machine Learning**: scikit-learn, XGBoost, LightGBM, CatBoost
- **Data Processing**: pandas, NumPy, SciPy
- **Visualization**: Plotly, Chart.js
- **NLP**: NLTK, TextBlob
- **Database**: SQLite (SQLAlchemy ORM)

## Installation

### Quick Start (Linux/Mac)
```bash
chmod +x install.sh
./install.sh
source venv/bin/activate
python app.py
```

### Windows
```cmd
install.bat
venv\Scripts\activate.bat
python app.py
```

### Docker
```bash
docker build -t marketmind .
docker run -p 5000:5000 marketmind
```

### Manual Installation
```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon'); nltk.download('wordnet')"
python app.py
```

## Access

After starting the application:
- **URL**: http://localhost:5000
- **Default Admin**: username `admin`, password `admin123`

## Project Structure

```
MarketMind/
  app.py                  # Main Flask application
  predictor.py            # ML forecasting engine
  analyzer.py             # Data analysis & consumer intelligence
  intelligence.py         # Competitive intelligence
  utils.py                # Utilities and helpers
  database/
    models.py             # SQLAlchemy ORM models
  templates/              # HTML templates
  static/                 # CSS, JS, images
  uploads/                # Uploaded datasets
  reports/                # Generated reports
  models/                 # Saved ML models
  requirements.txt        # Python dependencies
  Dockerfile              # Docker configuration
  install.sh              # Linux/Mac installer
  install.bat             # Windows installer
  README.md               # This file
```

## Authentication

- User registration with validation
- Secure password hashing (PBKDF2-SHA256)
- Login/logout with session management
- Password change functionality
- Role-based access (user/admin)
- Activity logging
- Login history tracking

## Design System

**Quantum Glass Intelligence** - A premium glassmorphism design featuring:
- Glass cards with backdrop blur
- Aurora background effects
- Floating particle animations
- Neon border highlights
- Animated KPI cards
- Smooth transitions
- Light/Dark theme support

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Landing page |
| `/login` | POST | Authenticate |
| `/register` | POST | Create account |
| `/dashboard` | GET | Main dashboard |
| `/datasets` | GET/POST | Dataset management |
| `/forecasting` | GET/POST | Run forecasts |
| `/competitors` | GET/POST | Competitor analysis |
| `/consumer` | GET/POST | Consumer insights |
| `/opportunities` | GET/POST | Opportunity detection |
| `/simulation` | GET/POST | Scenario simulation |
| `/executive` | GET | Executive dashboard |
| `/reports` | GET/POST | Report generation |
| `/visualizations` | GET/POST | Chart creation |
| `/search` | GET | Global search |
| `/admin` | GET | Admin panel |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SECRET_KEY` | (built-in) | Flask secret key |
| `DATABASE_URI` | `sqlite:///marketmind.db` | Database connection |
| `PORT` | `5000` | Server port |
| `FLASK_DEBUG` | `False` | Debug mode |

## License

This project is provided for enterprise use.
