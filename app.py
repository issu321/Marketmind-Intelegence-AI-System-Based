"""
MarketMind - Competitive Intelligence and Trend Prediction Platform
Main Flask Application - Deploy Ready for Hugging Face Spaces
"""

import os
import sys
import json
import traceback
import io
import base64
from datetime import datetime, timedelta
from functools import wraps

import pandas as pd
import numpy as np

# Try to import plotly, but provide fallback if not available
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.utils import PlotlyJSONEncoder
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("WARNING: Plotly not available. Charts will be disabled.")

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session, send_file, abort
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

# ==========================================
# UNIVERSAL NUMPY/PANDAS JSON SERIALIZER
# Fixes: "Object of type int32 is not JSON serializable"
# ==========================================

def _convert_numpy_types(obj):
    """Recursively convert numpy/pandas types to native Python JSON-safe types."""
    if isinstance(obj, dict):
        return {k: _convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_types(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(_convert_numpy_types(v) for v in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return _convert_numpy_types(obj.tolist())
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, pd.Timestamp):
        return obj.strftime('%Y-%m-%d %H:%M:%S')
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, bytes):
        return obj.decode('utf-8', errors='ignore')
    elif pd.isna(obj):
        return None
    else:
        return obj


def _safe_jsonify(data):
    """jsonify wrapper that auto-converts all numpy/pandas types."""
    return jsonify(_convert_numpy_types(data))


# ==========================================
# SAFE MODULE IMPORTS WITH FALLBACKS
# ==========================================

# Import MarketMind modules with safe fallbacks
MODULES_AVAILABLE = {}

try:
    from database.models import db, User, Dataset, Forecast, Report, CompetitorAnalysis, ConsumerInsight, Opportunity, ScenarioSimulation, ActivityLog, LoginHistory, AnalyticsCache, SystemSetting
    MODULES_AVAILABLE['models'] = True
except ImportError as e:
    print(f"WARNING: database.models not available: {e}")
    MODULES_AVAILABLE['models'] = False
    # Create minimal stubs
    db = SQLAlchemy()
    class User:
        pass
    class Dataset:
        pass
    class Forecast:
        pass
    class Report:
        pass
    class CompetitorAnalysis:
        pass
    class ConsumerInsight:
        pass
    class Opportunity:
        pass
    class ScenarioSimulation:
        pass
    class ActivityLog:
        pass
    class LoginHistory:
        pass
    class AnalyticsCache:
        pass
    class SystemSetting:
        pass

try:
    from predictor import ForecastingEngine, DemandForecaster, RevenueForecaster
    MODULES_AVAILABLE['predictor'] = True
except ImportError as e:
    print(f"WARNING: predictor not available: {e}")
    MODULES_AVAILABLE['predictor'] = False
    class ForecastingEngine:
        def forecast(self, *args, **kwargs):
            return {'success': False, 'error': 'Forecasting module not available'}


try:
    from analyzer import DataAnalyzer, ConsumerIntelligence, OpportunityDetector, ScenarioSimulator, ExecutiveInsights
    MODULES_AVAILABLE['analyzer'] = True
except ImportError as e:
    print(f"WARNING: analyzer not available: {e}")
    MODULES_AVAILABLE['analyzer'] = False
    class ConsumerIntelligence:
        def analyze_consumer_feedback(self, *args, **kwargs):
            return {'success': False, 'error': 'Consumer intelligence module not available'}
    class OpportunityDetector:
        def detect_opportunities(self, *args, **kwargs):
            return {'success': False, 'error': 'Opportunity detector not available'}
    class ScenarioSimulator:
        def simulate(self, *args, **kwargs):
            return {'success': False, 'error': 'Scenario simulator not available'}
        def get_available_scenarios(self):
            return []
    class ExecutiveInsights:
        def generate_executive_summary(self, *args, **kwargs):
            return {'success': False, 'error': 'Executive insights not available'}

try:
    from intelligence import CompetitorIntelligence, MarketIntelligence, StrategicIntelligence
    MODULES_AVAILABLE['intelligence'] = True
except ImportError as e:
    print(f"WARNING: intelligence not available: {e}")
    MODULES_AVAILABLE['intelligence'] = False
    class CompetitorIntelligence:
        def analyze_competitor(self, *args, **kwargs):
            return {'success': False, 'error': 'Competitor intelligence not available'}
    class MarketIntelligence:
        pass
    class StrategicIntelligence:
        pass

try:
    from utils import *
    MODULES_AVAILABLE['utils'] = True
except ImportError as e:
    print(f"WARNING: utils not available: {e}")
    MODULES_AVAILABLE['utils'] = False
    # Define minimal utility functions
    def format_number(x):
        return str(x)
    def format_currency(x):
        return f"${x}"
    def format_percentage(x):
        return f"{x}%"
    def score_to_color(x):
        return "#3b82f6"
    def score_to_rating(x):
        return "Good"
    def time_ago(x):
        return str(x)
    def validate_username(x):
        return len(x) >= 3 and len(x) <= 50
    def validate_email(x):
        return "@" in x
    def validate_password_strength(x):
        return len(x) >= 8, "Password must be at least 8 characters"
    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv', 'xlsx', 'xls'}
    def validate_file_size(file):
        return True
    def generate_unique_filename(filename):
        import uuid
        ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else 'csv'
        return f"{uuid.uuid4().hex}.{ext}"
    def clean_dataframe(df):
        return df
    def infer_column_types(df):
        return {col: 'unknown' for col in df.columns}
    def detect_data_quality_issues(df):
        return []
    def generate_summary_stats(df):
        return {}
    def clear_analytics_cache(user_id):
        pass
    def get_db_size(db_path):
        return "0 MB"
    def get_directory_size(path):
        return "0 MB"


# Initialize Flask App
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'marketmind-secret-key-2024-change-in-production')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URI', 'sqlite:///marketmind.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB

# Ensure directories exist (critical for containerized environments)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = os.environ.get('UPLOAD_FOLDER', os.path.join(BASE_DIR, 'uploads'))
app.config['REPORTS_FOLDER'] = os.environ.get('REPORTS_FOLDER', os.path.join(BASE_DIR, 'reports'))
app.config['MODELS_FOLDER'] = os.environ.get('MODELS_FOLDER', os.path.join(BASE_DIR, 'models'))

# Create directories if they don't exist
for folder in [app.config['UPLOAD_FOLDER'], app.config['REPORTS_FOLDER'], app.config['MODELS_FOLDER'],
               os.path.join(BASE_DIR, 'database'), os.path.join(BASE_DIR, 'static', 'css'),
               os.path.join(BASE_DIR, 'static', 'js'), os.path.join(BASE_DIR, 'static', 'images'),
               os.path.join(BASE_DIR, 'templates')]:
    os.makedirs(folder, exist_ok=True)

# Initialize Extensions
if MODULES_AVAILABLE['models']:
    db.init_app(app)
else:
    db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'info'

# Initialize Intelligence Engines (with safe fallbacks)
forecasting_engine = ForecastingEngine() if MODULES_AVAILABLE['predictor'] else None
competitor_intel = CompetitorIntelligence() if MODULES_AVAILABLE['intelligence'] else None
market_intel = MarketIntelligence() if MODULES_AVAILABLE['intelligence'] else None
consumer_intel = ConsumerIntelligence() if MODULES_AVAILABLE['analyzer'] else None
opportunity_detector = OpportunityDetector() if MODULES_AVAILABLE['analyzer'] else None
scenario_simulator = ScenarioSimulator() if MODULES_AVAILABLE['analyzer'] else None
executive_insights = ExecutiveInsights() if MODULES_AVAILABLE['analyzer'] else None
strategic_intel = StrategicIntelligence() if MODULES_AVAILABLE['intelligence'] else None

# Context processor for template globals
@app.context_processor
def inject_globals():
    return {
        'now': datetime.now(),
        'format_number': format_number,
        'format_currency': format_currency,
        'format_percentage': format_percentage,
        'score_to_color': score_to_color,
        'score_to_rating': score_to_rating,
        'time_ago': time_ago
    }

# User Loader
@login_manager.user_loader
def load_user(user_id):
    if MODULES_AVAILABLE['models']:
        return User.query.get(int(user_id))
    return None

# Before Request Handler
@app.before_request
def before_request():
    if current_user.is_authenticated and MODULES_AVAILABLE['models']:
        current_user.updated_at = datetime.now()
        db.session.commit()

# Database Initialization
with app.app_context():
    if MODULES_AVAILABLE['models']:
        db.create_all()
        # Create default admin user
        try:
            admin = User.query.filter_by(username='admin').first()
            if not admin:
                admin = User(
                    username='admin',
                    email='admin@marketmind.ai',
                    first_name='System',
                    last_name='Administrator',
                    role='admin',
                    is_active=True,
                    created_at=datetime.now()
                )
                admin.set_password('admin123')
                db.session.add(admin)
                db.session.commit()
        except Exception as e:
            print(f"Warning: Could not create admin user: {e}")


# ==========================================
# AUTHENTICATION ROUTES
# ==========================================

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        first_name = request.form.get('first_name', '').strip()
        last_name = request.form.get('last_name', '').strip()
        company = request.form.get('company', '').strip()

        if not username or not email or not password:
            flash('All required fields must be filled.', 'error')
            return render_template('auth/register.html')

        if not validate_username(username):
            flash('Username must be 3-50 characters, alphanumeric with underscores/hyphens only.', 'error')
            return render_template('auth/register.html')

        if not validate_email(email):
            flash('Please enter a valid email address.', 'error')
            return render_template('auth/register.html')

        is_valid, msg = validate_password_strength(password)
        if not is_valid:
            flash(msg, 'error')
            return render_template('auth/register.html')

        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return render_template('auth/register.html')

        if MODULES_AVAILABLE['models']:
            if User.query.filter_by(username=username).first():
                flash('Username already exists.', 'error')
                return render_template('auth/register.html')
            if User.query.filter_by(email=email).first():
                flash('Email already registered.', 'error')
                return render_template('auth/register.html')

            user = User(
                username=username,
                email=email,
                first_name=first_name,
                last_name=last_name,
                company=company,
                role='user',
                theme_preference=request.form.get('theme', 'dark'),
                created_at=datetime.now()
            )
            user.set_password(password)
            db.session.add(user)
            db.session.commit()

            log = ActivityLog(user_id=user.id, action='register', 
                             details=f'New registration: {username}', 
                             ip_address=request.remote_addr)
            db.session.add(log)
            db.session.commit()

        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('auth/register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        remember = request.form.get('remember', False) == 'on'

        if not username or not password:
            flash('Please enter both username and password.', 'error')
            return render_template('auth/login.html')

        if MODULES_AVAILABLE['models']:
            user = User.query.filter((User.username == username) | (User.email == username)).first()

            if user and user.check_password(password):
                if not user.is_active:
                    flash('Your account has been deactivated. Contact admin.', 'error')
                    return render_template('auth/login.html')

                login_user(user, remember=remember)
                user.last_login = datetime.now()
                user.login_count += 1
                db.session.commit()

                log = ActivityLog(user_id=user.id, action='login', 
                                 details='User logged in', 
                                 ip_address=request.remote_addr)
                db.session.add(log)

                login_hist = LoginHistory(user_id=user.id, 
                                         ip_address=request.remote_addr,
                                         user_agent=request.user_agent.string[:255] if request.user_agent else '',
                                         status='success')
                db.session.add(login_hist)
                db.session.commit()

                next_page = request.args.get('next')
                flash(f'Welcome back, {user.full_name}!', 'success')
                return redirect(next_page) if next_page else redirect(url_for('dashboard'))
            else:
                if user:
                    login_hist = LoginHistory(user_id=user.id, 
                                             ip_address=request.remote_addr,
                                             user_agent=request.user_agent.string[:255] if request.user_agent else '',
                                             status='failed',
                                             failure_reason='Invalid password')
                    db.session.add(login_hist)
                    db.session.commit()

        flash('Invalid username or password.', 'error')

    return render_template('auth/login.html')

@app.route('/logout')
@login_required
def logout():
    if MODULES_AVAILABLE['models']:
        log = ActivityLog(user_id=current_user.id, action='logout', 
                         details='User logged out', 
                         ip_address=request.remote_addr)
        db.session.add(log)
        db.session.commit()

    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        if MODULES_AVAILABLE['models']:
            user = User.query.filter_by(email=email).first()
            if user:
                flash('Password reset instructions sent to your email.', 'success')
            else:
                flash('If the email exists, reset instructions have been sent.', 'info')
        else:
            flash('If the email exists, reset instructions have been sent.', 'info')
        return redirect(url_for('login'))
    return render_template('auth/forgot_password.html')

@app.route('/change-password', methods=['POST'])
@login_required
def change_password():
    current = request.form.get('current_password', '')
    new_pass = request.form.get('new_password', '')
    confirm = request.form.get('confirm_password', '')

    if not current_user.check_password(current):
        flash('Current password is incorrect.', 'error')
        return redirect(url_for('profile'))
    if new_pass != confirm:
        flash('New passwords do not match.', 'error')
        return redirect(url_for('profile'))

    is_valid, msg = validate_password_strength(new_pass)
    if not is_valid:
        flash(msg, 'error')
        return redirect(url_for('profile'))

    current_user.set_password(new_pass)
    db.session.commit()

    if MODULES_AVAILABLE['models']:
        log = ActivityLog(user_id=current_user.id, action='password_change', 
                         details='Password changed', 
                         ip_address=request.remote_addr)
        db.session.add(log)
        db.session.commit()

    flash('Password changed successfully.', 'success')
    return redirect(url_for('profile'))

# ==========================================
# MAIN PAGES
# ==========================================

@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    stats = {
        'users': User.query.count() if MODULES_AVAILABLE['models'] else 0,
        'datasets': Dataset.query.count() if MODULES_AVAILABLE['models'] else 0,
        'forecasts': Forecast.query.count() if MODULES_AVAILABLE['models'] else 0,
        'reports': Report.query.count() if MODULES_AVAILABLE['models'] else 0
    }
    return render_template('landing.html', stats=stats)

@app.route('/dashboard')
@login_required
def dashboard():
    if not MODULES_AVAILABLE['models']:
        return render_template('dashboard.html', datasets=[], forecasts=[], reports=[], stats={}, activity=[])

    user_datasets = Dataset.query.filter_by(user_id=current_user.id).order_by(Dataset.created_at.desc()).limit(5).all()
    user_forecasts = Forecast.query.filter_by(user_id=current_user.id).order_by(Forecast.created_at.desc()).limit(5).all()
    user_reports = Report.query.filter_by(user_id=current_user.id).order_by(Report.created_at.desc()).limit(5).all()

    stats = {
        'datasets': Dataset.query.filter_by(user_id=current_user.id).count(),
        'forecasts': Forecast.query.filter_by(user_id=current_user.id).count(),
        'reports': Report.query.filter_by(user_id=current_user.id).count(),
        'analyses': CompetitorAnalysis.query.filter_by(user_id=current_user.id).count()
    }

    activity = ActivityLog.query.filter_by(user_id=current_user.id).order_by(ActivityLog.created_at.desc()).limit(10).all()

    return render_template('dashboard.html', 
                         datasets=user_datasets, 
                         forecasts=user_forecasts, 
                         reports=user_reports,
                         stats=stats,
                         activity=activity)

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy', 
        'timestamp': datetime.now().isoformat(),
        'modules': {k: v for k, v in MODULES_AVAILABLE.items()}
    })

# ==========================================
# PROFILE MANAGEMENT
# ==========================================

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == 'POST':
        current_user.first_name = request.form.get('first_name', '').strip()
        current_user.last_name = request.form.get('last_name', '').strip()
        current_user.email = request.form.get('email', '').strip()
        current_user.company = request.form.get('company', '').strip()
        current_user.job_title = request.form.get('job_title', '').strip()
        current_user.phone = request.form.get('phone', '').strip()
        current_user.bio = request.form.get('bio', '').strip()
        current_user.theme_preference = request.form.get('theme', 'dark')
        current_user.language = request.form.get('language', 'en')
        current_user.timezone = request.form.get('timezone', 'UTC')
        current_user.notification_enabled = request.form.get('notifications') == 'on'
        db.session.commit()

        if MODULES_AVAILABLE['models']:
            log = ActivityLog(user_id=current_user.id, action='profile_update', 
                             details='Profile updated', 
                             ip_address=request.remote_addr)
            db.session.add(log)
            db.session.commit()

        flash('Profile updated successfully.', 'success')
        return redirect(url_for('profile'))
    return render_template('profile.html')

@app.route('/theme/<theme>')
@login_required
def set_theme(theme):
    if theme in ['light', 'dark']:
        current_user.theme_preference = theme
        db.session.commit()
        session['theme'] = theme
    return redirect(request.referrer or url_for('dashboard'))


# ==========================================
# DATASET & FILE MANAGEMENT
# ==========================================

@app.route('/datasets')
@login_required
def datasets():
    if not MODULES_AVAILABLE['models']:
        flash('Database not available.', 'error')
        return redirect(url_for('dashboard'))

    page = request.args.get('page', 1, type=int)
    per_page = 20

    datasets_query = Dataset.query.filter_by(user_id=current_user.id).order_by(Dataset.created_at.desc())
    total = datasets_query.count()
    datasets = datasets_query.offset((page - 1) * per_page).limit(per_page).all()

    pagination = {
        'page': page,
        'per_page': per_page,
        'total': total,
        'total_pages': (total + per_page - 1) // per_page,
        'has_prev': page > 1,
        'has_next': page < ((total + per_page - 1) // per_page)
    }

    return render_template('datasets/list.html', datasets=datasets, pagination=pagination)

@app.route('/datasets/upload', methods=['GET', 'POST'])
@login_required
def upload_dataset():
    if not MODULES_AVAILABLE['models']:
        flash('Database not available for uploads.', 'error')
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected.', 'error')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No file selected.', 'error')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            try:
                if not validate_file_size(file):
                    flash('File size exceeds 100MB limit.', 'error')
                    return redirect(request.url)

                unique_filename = generate_unique_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                file.save(filepath)

                ext = unique_filename.rsplit('.', 1)[1].lower()
                if ext == 'csv':
                    df = pd.read_csv(filepath)
                elif ext in ['xlsx', 'xls']:
                    df = pd.read_excel(filepath)
                else:
                    flash('Unsupported file format.', 'error')
                    os.remove(filepath)
                    return redirect(request.url)

                df = clean_dataframe(df)
                col_types = infer_column_types(df)

                date_col = None
                value_col = None
                category_col = None

                for col, ctype in col_types.items():
                    if ctype == 'date' and date_col is None:
                        date_col = col
                    elif ctype == 'value' and value_col is None:
                        value_col = col
                    elif ctype == 'category' and category_col is None:
                        category_col = col

                quality_issues = detect_data_quality_issues(df)
                quality_score = max(0, 100 - len([i for i in quality_issues if i['severity'] == 'critical']) * 20 
                                  - len([i for i in quality_issues if i['severity'] == 'warning']) * 10)

                dataset = Dataset(
                    user_id=current_user.id,
                    filename=unique_filename,
                    original_filename=secure_filename(file.filename),
                    file_path=filepath,
                    file_size=os.path.getsize(filepath),
                    file_type=ext,
                    row_count=len(df),
                    column_count=len(df.columns),
                    date_column=date_col,
                    value_column=value_col,
                    category_column=category_col,
                    data_quality_score=quality_score,
                    processing_status='completed',
                    is_processed=True,
                    description=request.form.get('description', ''),
                    tags=request.form.get('tags', ''),
                    created_at=datetime.now()
                )
                dataset.set_columns(df.columns.tolist())
                dataset.set_column_types(col_types)
                dataset.set_quality_issues(quality_issues)

                summary = generate_summary_stats(df)
                dataset.set_summary_stats(summary)

                db.session.add(dataset)
                db.session.commit()

                clear_analytics_cache(current_user.id)

                log = ActivityLog(user_id=current_user.id, action='dataset_upload', 
                                details=f'Uploaded dataset: {file.filename}', 
                                ip_address=request.remote_addr)
                db.session.add(log)
                db.session.commit()

                flash(f'Dataset "{file.filename}" uploaded and processed successfully.', 'success')
                return redirect(url_for('view_dataset', dataset_id=dataset.id))

            except Exception as e:
                flash(f'Error processing file: {str(e)}', 'error')
                if 'filepath' in locals() and os.path.exists(filepath):
                    os.remove(filepath)
                return redirect(request.url)
        else:
            flash('Invalid file type. Only CSV and Excel files are allowed.', 'error')

    return render_template('datasets/upload.html')

@app.route('/datasets/<int:dataset_id>')
@login_required
def view_dataset(dataset_id):
    if not MODULES_AVAILABLE['models']:
        abort(404)

    dataset = Dataset.query.filter_by(id=dataset_id, user_id=current_user.id).first_or_404()

    try:
        ext = dataset.filename.rsplit('.', 1)[1].lower()
        filepath = dataset.file_path

        if ext == 'csv':
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)

        df = clean_dataframe(df)
        sample = df.head(100)
        table_html = sample.to_html(classes='data-table', index=False, escape=False)

        return render_template('datasets/view.html', 
                             dataset=dataset, 
                             table_html=table_html,
                             row_count=len(df),
                             columns=df.columns.tolist())

    except Exception as e:
        flash(f'Error loading dataset: {str(e)}', 'error')
        return redirect(url_for('datasets'))

@app.route('/datasets/<int:dataset_id>/delete', methods=['POST'])
@login_required
def delete_dataset(dataset_id):
    if not MODULES_AVAILABLE['models']:
        abort(404)

    dataset = Dataset.query.filter_by(id=dataset_id, user_id=current_user.id).first_or_404()

    try:
        if os.path.exists(dataset.file_path):
            os.remove(dataset.file_path)

        Forecast.query.filter_by(dataset_id=dataset.id).delete()
        Report.query.filter_by(dataset_id=dataset.id).delete()
        CompetitorAnalysis.query.filter_by(dataset_id=dataset.id).delete()
        ConsumerInsight.query.filter_by(dataset_id=dataset.id).delete()
        Opportunity.query.filter_by(dataset_id=dataset.id).delete()

        db.session.delete(dataset)
        db.session.commit()

        log = ActivityLog(user_id=current_user.id, action='dataset_delete', 
                         details=f'Deleted dataset: {dataset.original_filename}', 
                         ip_address=request.remote_addr)
        db.session.add(log)
        db.session.commit()

        flash('Dataset deleted successfully.', 'success')
    except Exception as e:
        flash(f'Error deleting dataset: {str(e)}', 'error')

    return redirect(url_for('datasets'))

@app.route('/api/datasets/<int:dataset_id>/columns')
@login_required
def get_dataset_columns(dataset_id):
    if not MODULES_AVAILABLE['models']:
        return _safe_jsonify({'columns': [], 'types': {}})

    dataset = Dataset.query.filter_by(id=dataset_id, user_id=current_user.id).first_or_404()
    return _safe_jsonify({'columns': dataset.get_columns(), 'types': dataset.get_column_types()})

# ==========================================
# FORECASTING ENGINE
# ==========================================

@app.route('/forecasting')
@login_required
def forecasting():
    if not MODULES_AVAILABLE['models']:
        return render_template('forecasting/index.html', datasets=[], forecasts=[])

    datasets = Dataset.query.filter_by(user_id=current_user.id, is_processed=True).order_by(Dataset.created_at.desc()).all()
    forecasts = Forecast.query.filter_by(user_id=current_user.id).order_by(Forecast.created_at.desc()).all()
    return render_template('forecasting/index.html', datasets=datasets, forecasts=forecasts)

@app.route('/forecasting/run', methods=['POST'])
@login_required
def run_forecast():
    if not MODULES_AVAILABLE['models'] or not MODULES_AVAILABLE['predictor']:
        return _safe_jsonify({'success': False, 'error': 'Forecasting module not available.'})

    try:
        dataset_id = request.form.get('dataset_id', type=int)
        date_col = request.form.get('date_column')
        value_col = request.form.get('value_column')
        forecast_type = request.form.get('forecast_type', 'demand')
        horizon = request.form.get('horizon', 30, type=int)
        model_type = request.form.get('model_type', 'auto')
        confidence = request.form.get('confidence', 0.95, type=float)
        forecast_name = request.form.get('forecast_name', f'Forecast {datetime.now().strftime("%Y-%m-%d %H:%M")}')

        if not dataset_id or not date_col or not value_col:
            return _safe_jsonify({'success': False, 'error': 'Please select dataset, date column, and value column.'})

        dataset = Dataset.query.filter_by(id=dataset_id, user_id=current_user.id).first_or_404()

        ext = dataset.filename.rsplit('.', 1)[1].lower()
        if ext == 'csv':
            df = pd.read_csv(dataset.file_path)
        else:
            df = pd.read_excel(dataset.file_path)

        df = clean_dataframe(df)

        if date_col not in df.columns or value_col not in df.columns:
            return _safe_jsonify({'success': False, 'error': 'Selected columns not found in dataset.'})

        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col, value_col])

        if len(df) < 10:
            return _safe_jsonify({'success': False, 'error': 'Insufficient data points (minimum 10 required).'})

        engine = ForecastingEngine(model_type=model_type)
        result = engine.forecast(df, date_col, value_col, horizon=horizon, confidence=confidence)

        if not result.get('success'):
            return _safe_jsonify({'success': False, 'error': result.get('error', 'Forecast failed')})

        # CRITICAL FIX: Convert all numpy types BEFORE saving to database
        result = _convert_numpy_types(result)

        forecast = Forecast(
            user_id=current_user.id,
            dataset_id=dataset_id,
            name=forecast_name,
            description=f'{forecast_type} forecast using {result.get("model_name", "auto")}',
            forecast_type=forecast_type,
            model_used=result.get('model_name', 'unknown'),
            forecast_horizon=horizon,
            confidence_level=confidence,
            insights='\n'.join(result.get('insights', [])),
            is_active=True,
            created_at=datetime.now()
        )
        forecast.set_metrics(result.get('metrics', {}))
        forecast.set_forecast_data(result.get('forecast_data', []))
        forecast.set_feature_importance(result.get('feature_importance', {}))

        db.session.add(forecast)
        db.session.commit()

        log = ActivityLog(user_id=current_user.id, action='forecast_run', 
                         details=f'Ran forecast: {forecast_name}', 
                         ip_address=request.remote_addr)
        db.session.add(log)
        db.session.commit()

        result['forecast_id'] = forecast.id
        return _safe_jsonify(result)

    except Exception as e:
        return _safe_jsonify({'success': False, 'error': str(e), 'traceback': traceback.format_exc()})

@app.route('/forecasting/<int:forecast_id>')
@login_required
def view_forecast(forecast_id):
    if not MODULES_AVAILABLE['models']:
        abort(404)
    forecast = Forecast.query.filter_by(id=forecast_id, user_id=current_user.id).first_or_404()
    return render_template('forecasting/view.html', forecast=forecast)

@app.route('/forecasting/<int:forecast_id>/chart')
@login_required
def forecast_chart(forecast_id):
    if not MODULES_AVAILABLE['models'] or not PLOTLY_AVAILABLE:
        return _safe_jsonify({'success': False, 'error': 'Chart generation not available'})

    forecast = Forecast.query.filter_by(id=forecast_id, user_id=current_user.id).first_or_404()

    forecast_data = forecast.get_forecast_data()
    if not forecast_data:
        return _safe_jsonify({'success': False, 'error': 'No forecast data available'})

    dates = [d['date'] for d in forecast_data]
    predictions = [d['prediction'] for d in forecast_data]
    lower = [d['lower_bound'] for d in forecast_data]
    upper = [d['upper_bound'] for d in forecast_data]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates, y=predictions,
        mode='lines+markers',
        name='Forecast',
        line=dict(color='#3b82f6', width=3),
        marker=dict(size=6)
    ))

    fig.add_trace(go.Scatter(
        x=dates + dates[::-1],
        y=upper + lower[::-1],
        fill='toself',
        fillcolor='rgba(59, 130, 246, 0.1)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Confidence Interval'
    ))

    fig.update_layout(
        title=f'{forecast.name} - Forecast Chart',
        xaxis_title='Date',
        yaxis_title='Value',
        template='plotly_dark',
        height=500
    )

    return _safe_jsonify(json.loads(json.dumps(fig, cls=PlotlyJSONEncoder)))



# ==========================================
# COMPETITOR INTELLIGENCE
# ==========================================

@app.route('/competitors')
@login_required
def competitors():
    if not MODULES_AVAILABLE['models']:
        return render_template('competitors/index.html', datasets=[], analyses=[])

    datasets = Dataset.query.filter_by(user_id=current_user.id, is_processed=True).all()
    analyses = CompetitorAnalysis.query.filter_by(user_id=current_user.id).order_by(CompetitorAnalysis.created_at.desc()).all()
    return render_template('competitors/index.html', datasets=datasets, analyses=analyses)

@app.route('/competitors/analyze', methods=['POST'])
@login_required
def analyze_competitors():
    if not MODULES_AVAILABLE['models'] or not MODULES_AVAILABLE['intelligence']:
        return _safe_jsonify({'success': False, 'error': 'Competitor intelligence module not available.'})

    try:
        dataset_id = request.form.get('dataset_id', type=int)
        competitor_col = request.form.get('competitor_column')
        value_col = request.form.get('value_column')
        date_col = request.form.get('date_column', '') or None
        price_col = request.form.get('price_column', '') or None
        review_col = request.form.get('review_column', '') or None

        if not dataset_id or not competitor_col or not value_col:
            return _safe_jsonify({'success': False, 'error': 'Missing required parameters.'})

        dataset = Dataset.query.filter_by(id=dataset_id, user_id=current_user.id).first_or_404()

        ext = dataset.filename.rsplit('.', 1)[1].lower()
        if ext == 'csv':
            df = pd.read_csv(dataset.file_path)
        else:
            df = pd.read_excel(dataset.file_path)

        df = clean_dataframe(df)

        result = competitor_intel.analyze_competitor(df, competitor_col, value_col, date_col, price_col, review_col)

        if not result.get('success'):
            return _safe_jsonify(result)

        for analysis in result.get('analyses', []):
            comp = CompetitorAnalysis(
                user_id=current_user.id,
                dataset_id=dataset_id,
                competitor_name=analysis['competitor_name'],
                analysis_type='comprehensive',
                overall_score=analysis.get('overall_score', 0),
                threat_score=analysis.get('threat_score', 0),
                growth_score=analysis.get('growth_score', 0),
                market_position_score=analysis.get('market_position_score', 0),
                innovation_score=analysis.get('innovation_score', 0),
                pricing_score=analysis.get('pricing_score', 0),
                sentiment_score=analysis.get('sentiment_score', 0),
                insights='\n'.join(result.get('insights', [])),
                created_at=datetime.now()
            )
            comp.set_strengths(analysis.get('strengths', []))
            comp.set_weaknesses(analysis.get('weaknesses', []))
            comp.set_opportunities(analysis.get('opportunities', []))
            comp.set_threats(analysis.get('threats', []))
            comp.set_analysis_data(analysis)
            db.session.add(comp)

        db.session.commit()

        log = ActivityLog(user_id=current_user.id, action='competitor_analysis', 
                         details=f'Analyzed competitors in dataset {dataset.original_filename}', 
                         ip_address=request.remote_addr)
        db.session.add(log)
        db.session.commit()

        return _safe_jsonify(result)

    except Exception as e:
        return _safe_jsonify({'success': False, 'error': str(e)})

@app.route('/competitors/<int:analysis_id>')
@login_required
def view_competitor(analysis_id):
    if not MODULES_AVAILABLE['models']:
        abort(404)
    analysis = CompetitorAnalysis.query.filter_by(id=analysis_id, user_id=current_user.id).first_or_404()
    return render_template('competitors/view.html', analysis=analysis)

# ==========================================
# CONSUMER INTELLIGENCE
# ==========================================

@app.route('/consumer')
@login_required
def consumer_intelligence():
    if not MODULES_AVAILABLE['models']:
        return render_template('consumer/index.html', datasets=[], insights=[])

    datasets = Dataset.query.filter_by(user_id=current_user.id, is_processed=True).all()
    insights = ConsumerInsight.query.filter_by(user_id=current_user.id).order_by(ConsumerInsight.created_at.desc()).all()
    return render_template('consumer/index.html', datasets=datasets, insights=insights)

@app.route('/consumer/analyze', methods=['POST'])
@login_required
def analyze_consumer():
    if not MODULES_AVAILABLE['models'] or not MODULES_AVAILABLE['analyzer']:
        return _safe_jsonify({'success': False, 'error': 'Consumer intelligence module not available.'})

    try:
        dataset_id = request.form.get('dataset_id', type=int)
        text_col = request.form.get('text_column')
        rating_col = request.form.get('rating_column', '') or None

        if not dataset_id or not text_col:
            return _safe_jsonify({'success': False, 'error': 'Missing required parameters.'})

        dataset = Dataset.query.filter_by(id=dataset_id, user_id=current_user.id).first_or_404()

        ext = dataset.filename.rsplit('.', 1)[1].lower()
        if ext == 'csv':
            df = pd.read_csv(dataset.file_path)
        else:
            df = pd.read_excel(dataset.file_path)

        df = clean_dataframe(df)

        result = consumer_intel.analyze_consumer_feedback(df, text_col, rating_col)

        if not result.get('success'):
            return _safe_jsonify(result)

        insight = ConsumerInsight(
            user_id=current_user.id,
            dataset_id=dataset_id,
            insight_type='sentiment_analysis',
            positive_score=result.get('positive_score', 0),
            negative_score=result.get('negative_score', 0),
            neutral_score=result.get('neutral_score', 0),
            brand_health_score=result.get('brand_health_score', 0),
            emotion_score=result.get('emotion_score', 0),
            trust_score=result.get('trust_score', 0),
            insights=result.get('insights', ''),
            recommendations=json.dumps(result.get('recommendations', [])),
            created_at=datetime.now()
        )
        insight.set_sentiment_distribution(result.get('sentiment_distribution', {}))
        insight.set_key_topics(result.get('key_topics', []))
        insight.set_emotion_breakdown(result.get('emotion_breakdown', {}))

        db.session.add(insight)
        db.session.commit()

        result['insight_id'] = insight.id
        return _safe_jsonify(result)

    except Exception as e:
        return _safe_jsonify({'success': False, 'error': str(e)})

# ==========================================
# OPPORTUNITY DETECTION
# ==========================================

@app.route('/opportunities')
@login_required
def opportunities():
    if not MODULES_AVAILABLE['models']:
        return render_template('opportunities/index.html', datasets=[], opportunities=[])

    datasets = Dataset.query.filter_by(user_id=current_user.id, is_processed=True).all()
    opportunities_list = Opportunity.query.filter_by(user_id=current_user.id).order_by(Opportunity.created_at.desc()).all()
    return render_template('opportunities/index.html', datasets=datasets, opportunities=opportunities_list)

@app.route('/opportunities/detect', methods=['POST'])
@login_required
def detect_opportunities():
    if not MODULES_AVAILABLE['models'] or not MODULES_AVAILABLE['analyzer']:
        return _safe_jsonify({'success': False, 'error': 'Opportunity detection module not available.'})

    try:
        dataset_id = request.form.get('dataset_id', type=int)
        date_col = request.form.get('date_column')
        value_col = request.form.get('value_column')
        category_col = request.form.get('category_column', '') or None

        if not dataset_id or not date_col or not value_col:
            return _safe_jsonify({'success': False, 'error': 'Missing required parameters.'})

        dataset = Dataset.query.filter_by(id=dataset_id, user_id=current_user.id).first_or_404()

        ext = dataset.filename.rsplit('.', 1)[1].lower()
        if ext == 'csv':
            df = pd.read_csv(dataset.file_path)
        else:
            df = pd.read_excel(dataset.file_path)

        df = clean_dataframe(df)

        result = opportunity_detector.detect_opportunities(df, date_col, value_col, category_col)

        if not result.get('success'):
            return _safe_jsonify(result)

        for opp in result.get('opportunities', []):
            opportunity = Opportunity(
                user_id=current_user.id,
                dataset_id=dataset_id,
                opportunity_name=opp['opportunity_name'],
                opportunity_type=opp['opportunity_type'],
                opportunity_score=opp.get('opportunity_score', 0),
                revenue_potential=opp.get('revenue_potential', 0),
                market_readiness_score=opp.get('market_readiness_score', 0),
                investment_score=opp.get('investment_score', 0),
                risk_level=opp.get('risk_level', 'medium'),
                description=opp.get('description', ''),
                recommendations=json.dumps(opp.get('recommendations', [])),
                is_active=True,
                created_at=datetime.now()
            )
            db.session.add(opportunity)

        db.session.commit()
        return _safe_jsonify(result)

    except Exception as e:
        return _safe_jsonify({'success': False, 'error': str(e)})

# ==========================================
# SCENARIO SIMULATION
# ==========================================

@app.route('/simulation')
@login_required
def simulation():
    if not MODULES_AVAILABLE['models']:
        return render_template('simulation/index.html', datasets=[], scenarios=[], available_scenarios=[])

    datasets = Dataset.query.filter_by(user_id=current_user.id, is_processed=True).all()
    scenarios = ScenarioSimulation.query.filter_by(user_id=current_user.id).order_by(ScenarioSimulation.created_at.desc()).all()
    available_scenarios = scenario_simulator.get_available_scenarios() if scenario_simulator else []
    return render_template('simulation/index.html', datasets=datasets, scenarios=scenarios, available_scenarios=available_scenarios)

@app.route('/simulation/run', methods=['POST'])
@login_required
def run_simulation():
    if not MODULES_AVAILABLE['models'] or not MODULES_AVAILABLE['analyzer']:
        return _safe_jsonify({'success': False, 'error': 'Scenario simulation module not available.'})

    try:
        dataset_id = request.form.get('dataset_id', type=int)
        value_col = request.form.get('value_column')
        date_col = request.form.get('date_column')
        scenario_type = request.form.get('scenario_type')
        impact_percentage = request.form.get('impact_percentage', 10, type=float)

        if not dataset_id or not value_col or not date_col or not scenario_type:
            return _safe_jsonify({'success': False, 'error': 'Missing required parameters.'})

        dataset = Dataset.query.filter_by(id=dataset_id, user_id=current_user.id).first_or_404()

        ext = dataset.filename.rsplit('.', 1)[1].lower()
        if ext == 'csv':
            df = pd.read_csv(dataset.file_path)
        else:
            df = pd.read_excel(dataset.file_path)

        df = clean_dataframe(df)

        result = scenario_simulator.simulate(df, value_col, date_col, scenario_type, impact_percentage)

        if not result.get('success'):
            return _safe_jsonify(result)

        sim = ScenarioSimulation(
            user_id=current_user.id,
            dataset_id=dataset_id,
            name=f"{result.get('scenario_name', scenario_type)} - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            scenario_type=scenario_type,
            recommendations=json.dumps(result.get('recommendations', [])),
            created_at=datetime.now()
        )
        sim.set_parameters({'impact_percentage': impact_percentage, 'scenario_type': scenario_type})
        sim.set_baseline_metrics(result.get('baseline_metrics', {}))
        sim.set_simulated_metrics(result.get('simulated_metrics', {}))
        sim.set_impact_analysis(result.get('impact_analysis', {}))
        sim.set_comparison_chart(result.get('comparison_data', {}))

        db.session.add(sim)
        db.session.commit()

        result['simulation_id'] = sim.id
        return _safe_jsonify(result)

    except Exception as e:
        return _safe_jsonify({'success': False, 'error': str(e)})


# ==========================================
# EXECUTIVE COMMAND CENTER
# ==========================================

@app.route('/executive')
@login_required
def executive_center():
    try:
        if not MODULES_AVAILABLE['models']:
            return render_template('executive/index.html',
                                 summary={},
                                 kpis={
                                     'datasets': 0,
                                     'forecasts': 0,
                                     'analyses': 0,
                                     'opportunities': 0,
                                     'health_scores': {}
                                 },
                                 latest_forecast=None,
                                 latest_competitor=None,
                                 latest_consumer=None)

        latest_forecast = Forecast.query.filter_by(user_id=current_user.id).order_by(Forecast.created_at.desc()).first()
        latest_competitor = CompetitorAnalysis.query.filter_by(user_id=current_user.id).order_by(CompetitorAnalysis.created_at.desc()).first()
        latest_consumer = ConsumerInsight.query.filter_by(user_id=current_user.id).order_by(ConsumerInsight.created_at.desc()).first()
        latest_opportunity = Opportunity.query.filter_by(user_id=current_user.id).order_by(Opportunity.created_at.desc()).first()

        forecast_data = None
        if latest_forecast:
            forecast_data = {
                'success': True,
                'model_name': latest_forecast.model_used,
                'forecast_data': latest_forecast.get_forecast_data(),
                'metrics': latest_forecast.get_metrics(),
                'summary': {
                    'growth_projection': 0,
                    'prediction_trend': 'stable'
                }
            }
            fd = latest_forecast.get_forecast_data()
            if fd and len(fd) > 1:
                first_pred = fd[0]['prediction']
                last_pred = fd[-1]['prediction']
                if first_pred != 0:
                    forecast_data['summary']['growth_projection'] = round(((last_pred - first_pred) / abs(first_pred)) * 100, 2)
                forecast_data['summary']['prediction_trend'] = 'increasing' if last_pred > first_pred else 'decreasing' if last_pred < first_pred else 'stable'

        competitor_data = None
        if latest_competitor:
            competitor_data = {
                'success': True,
                'overall_score': latest_competitor.overall_score,
                'threat_score': latest_competitor.threat_score,
                'growth_score': latest_competitor.growth_score,
                'market_position_score': latest_competitor.market_position_score,
                'innovation_score': latest_competitor.innovation_score,
                'pricing_score': latest_competitor.pricing_score,
                'sentiment_score': latest_competitor.sentiment_score,
                'insights': latest_competitor.insights or ''
            }

        consumer_data = None
        if latest_consumer:
            consumer_data = {
                'success': True,
                'positive_score': latest_consumer.positive_score,
                'negative_score': latest_consumer.negative_score,
                'neutral_score': latest_consumer.neutral_score,
                'brand_health_score': latest_consumer.brand_health_score,
                'emotion_score': latest_consumer.emotion_score,
                'trust_score': latest_consumer.trust_score
            }

        opportunity_data = None
        if latest_opportunity:
            opportunity_data = {
                'success': True,
                'total_detected': Opportunity.query.filter_by(user_id=current_user.id).count()
            }

        if executive_insights:
            summary = executive_insights.generate_executive_summary(
                forecast_result=forecast_data,
                competitor_analysis=competitor_data,
                consumer_insights=consumer_data,
                opportunities=opportunity_data
            )
        else:
            summary = {'success': False, 'summary': {}}

        kpis = {
            'datasets': Dataset.query.filter_by(user_id=current_user.id).count(),
            'forecasts': Forecast.query.filter_by(user_id=current_user.id).count(),
            'analyses': CompetitorAnalysis.query.filter_by(user_id=current_user.id).count(),
            'opportunities': Opportunity.query.filter_by(user_id=current_user.id).count(),
            'health_scores': summary.get('summary', {}).get('overall_health', {}) if summary.get('success') else {}
        }

        return render_template('executive/index.html', 
                             summary=summary.get('summary', {}) if summary.get('success') else {},
                             kpis=kpis,
                             latest_forecast=latest_forecast,
                             latest_competitor=latest_competitor,
                             latest_consumer=latest_consumer)

    except Exception as e:
        import traceback
        app.logger.error(f"Executive center error: {str(e)}\n{traceback.format_exc()}")
        return render_template('executive/index.html',
                             summary={},
                             kpis={
                                 'datasets': Dataset.query.filter_by(user_id=current_user.id).count() if MODULES_AVAILABLE['models'] else 0,
                                 'forecasts': Forecast.query.filter_by(user_id=current_user.id).count() if MODULES_AVAILABLE['models'] else 0,
                                 'analyses': CompetitorAnalysis.query.filter_by(user_id=current_user.id).count() if MODULES_AVAILABLE['models'] else 0,
                                 'opportunities': Opportunity.query.filter_by(user_id=current_user.id).count() if MODULES_AVAILABLE['models'] else 0,
                                 'health_scores': {}
                             },
                             latest_forecast=None,
                             latest_competitor=None,
                             latest_consumer=None)

# ==========================================
# REPORT CENTER
# ==========================================

@app.route('/reports')
@login_required
def reports():
    if not MODULES_AVAILABLE['models']:
        return render_template('reports/index.html', reports=[])

    reports_list = Report.query.filter_by(user_id=current_user.id).order_by(Report.created_at.desc()).all()
    return render_template('reports/index.html', reports=reports_list)

@app.route('/reports/generate', methods=['POST'])
@login_required
def generate_report():
    if not MODULES_AVAILABLE['models']:
        flash('Report generation not available.', 'error')
        return redirect(url_for('reports'))

    try:
        report_type = request.form.get('report_type', 'summary')
        title = request.form.get('title', f'Report - {datetime.now().strftime("%Y-%m-%d")}')
        format_type = request.form.get('format', 'html')

        datasets = Dataset.query.filter_by(user_id=current_user.id).all()
        forecasts = Forecast.query.filter_by(user_id=current_user.id).all()
        analyses = CompetitorAnalysis.query.filter_by(user_id=current_user.id).all()

        content = {
            'title': title,
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'user': current_user.full_name,
            'summary': {
                'datasets_analyzed': len(datasets),
                'forecasts_generated': len(forecasts),
                'competitors_analyzed': len(analyses)
            }
        }

        findings = []
        if forecasts:
            latest = forecasts[-1]
            fd = latest.get_forecast_data()
            if fd:
                findings.append(f"Latest forecast '{latest.name}' shows {fd[-1]['prediction'] if fd else 'N/A'} projected value.")
        if analyses:
            top_threat = max(analyses, key=lambda x: x.threat_score)
            findings.append(f"Highest competitive threat from {top_threat.competitor_name} with threat score of {top_threat.threat_score:.1f}.")

        report = Report(
            user_id=current_user.id,
            title=title,
            report_type=report_type,
            format=format_type,
            content=json.dumps(content),
            summary=f"Report covering {len(datasets)} datasets, {len(forecasts)} forecasts, and {len(analyses)} competitor analyses.",
            is_favorite=False,
            created_at=datetime.now()
        )
        report.set_key_findings(findings)
        report.set_recommendations([
            "Continue monitoring market trends and competitor activities.",
            "Regularly update forecasts with latest data.",
            "Review opportunities on a weekly basis."
        ])

        db.session.add(report)
        db.session.commit()

        flash('Report generated successfully.', 'success')
        return redirect(url_for('view_report', report_id=report.id))

    except Exception as e:
        flash(f'Error generating report: {str(e)}', 'error')
        return redirect(url_for('reports'))

@app.route('/reports/<int:report_id>')
@login_required
def view_report(report_id):
    if not MODULES_AVAILABLE['models']:
        abort(404)
    report = Report.query.filter_by(id=report_id, user_id=current_user.id).first_or_404()
    return render_template('reports/view.html', report=report)

@app.route('/reports/<int:report_id>/download')
@login_required
def download_report(report_id):
    if not MODULES_AVAILABLE['models']:
        abort(404)

    report = Report.query.filter_by(id=report_id, user_id=current_user.id).first_or_404()

    try:
        content = json.loads(report.content) if report.content else {}

        if report.format == 'json':
            output = io.StringIO()
            json.dump(content, output, indent=2)
            output.seek(0)
            return send_file(io.BytesIO(output.getvalue().encode()), 
                           mimetype='application/json',
                           as_attachment=True, 
                           download_name=f"{report.title.replace(' ', '_')}.json")

        elif report.format == 'csv':
            output = io.StringIO()
            output.write(f"Report: {report.title}\n")
            output.write(f"Generated: {content.get('generated_at', 'N/A')}\n")
            output.write(f"Findings: {', '.join(report.get_key_findings())}\n")
            output.seek(0)
            return send_file(io.BytesIO(output.getvalue().encode()),
                           mimetype='text/csv',
                           as_attachment=True,
                           download_name=f"{report.title.replace(' ', '_')}.csv")

        else:
            html_content = render_template('reports/template.html', report=report, content=content)
            return send_file(io.BytesIO(html_content.encode()),
                           mimetype='text/html',
                           as_attachment=True,
                           download_name=f"{report.title.replace(' ', '_')}.html")

    except Exception as e:
        flash(f'Error downloading report: {str(e)}', 'error')
        return redirect(url_for('view_report', report_id=report_id))


# ==========================================
# VISUALIZATION CENTER - FIXED
# ==========================================

@app.route('/visualizations')
@login_required
def visualizations():
    if not MODULES_AVAILABLE['models']:
        return render_template('visualizations/index.html', datasets=[])

    datasets = Dataset.query.filter_by(user_id=current_user.id, is_processed=True).all()
    return render_template('visualizations/index.html', datasets=datasets)

@app.route('/visualizations/generate', methods=['POST'])
@login_required
def generate_visualization():
    if not PLOTLY_AVAILABLE:
        return _safe_jsonify({'success': False, 'error': 'Plotly not available for chart generation.'})

    try:
        dataset_id = request.form.get('dataset_id', type=int)
        chart_type = request.form.get('chart_type', 'line')
        x_col = request.form.get('x_column')
        y_col = request.form.get('y_column')
        color_col = request.form.get('color_column', '') or None

        if not dataset_id or not x_col or not y_col:
            return _safe_jsonify({'success': False, 'error': 'Missing required parameters.'})

        if not MODULES_AVAILABLE['models']:
            return _safe_jsonify({'success': False, 'error': 'Database not available.'})

        dataset = Dataset.query.filter_by(id=dataset_id, user_id=current_user.id).first_or_404()

        ext = dataset.filename.rsplit('.', 1)[1].lower()
        if ext == 'csv':
            df = pd.read_csv(dataset.file_path)
        else:
            df = pd.read_excel(dataset.file_path)

        df = clean_dataframe(df)

        if x_col not in df.columns:
            return _safe_jsonify({'success': False, 'error': f'X column "{x_col}" not found in dataset. Available: {list(df.columns)}'})
        if y_col not in df.columns:
            return _safe_jsonify({'success': False, 'error': f'Y column "{y_col}" not found in dataset. Available: {list(df.columns)}'})
        if color_col and color_col not in df.columns:
            color_col = None

        if chart_type in ['line', 'bar', 'scatter', 'box', 'histogram']:
            df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
            df = df.dropna(subset=[y_col])

        if chart_type == 'pie':
            df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
            df = df.dropna(subset=[x_col, y_col])

        if len(df) == 0:
            return _safe_jsonify({'success': False, 'error': 'No valid data after cleaning. Check column data types.'})

        try:
            if chart_type == 'line':
                fig = px.line(df, x=x_col, y=y_col, color=color_col, 
                             title=f'{y_col} over {x_col}', markers=True)
            elif chart_type == 'bar':
                fig = px.bar(df, x=x_col, y=y_col, color=color_col, 
                            title=f'{y_col} by {x_col}', text_auto='.2s')
            elif chart_type == 'scatter':
                fig = px.scatter(df, x=x_col, y=y_col, color=color_col, 
                                title=f'{y_col} vs {x_col}', size_max=15)
            elif chart_type == 'histogram':
                fig = px.histogram(df, x=y_col, color=color_col, 
                                  title=f'Distribution of {y_col}', nbins=20)
            elif chart_type == 'box':
                fig = px.box(df, x=x_col, y=y_col, color=color_col, 
                            title=f'{y_col} Distribution by {x_col}', points='outliers')
            elif chart_type == 'pie':
                pie_data = df.groupby(x_col, as_index=False)[y_col].sum()
                pie_data = pie_data.sort_values(y_col, ascending=False).head(20)
                fig = px.pie(pie_data, names=x_col, values=y_col, 
                            title=f'{y_col} by {x_col}', hole=0.3)
            else:
                fig = px.line(df, x=x_col, y=y_col, color=color_col)
        except Exception as chart_err:
            return _safe_jsonify({'success': False, 'error': f'Chart generation error: {str(chart_err)}'})

        fig.update_layout(
            template='plotly_dark',
            height=600,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(30,30,50,0.5)',
            font=dict(color='#e2e8f0'),
            title_font=dict(size=16, color='#e2e8f0'),
            legend=dict(
                bgcolor='rgba(0,0,0,0.3)',
                bordercolor='rgba(255,255,255,0.1)',
                borderwidth=1
            ),
            margin=dict(l=60, r=30, t=80, b=60)
        )

        fig.update_xaxes(
            gridcolor='rgba(255,255,255,0.1)',
            linecolor='rgba(255,255,255,0.2)',
            tickfont=dict(color='#94a3b8')
        )
        fig.update_yaxes(
            gridcolor='rgba(255,255,255,0.1)',
            linecolor='rgba(255,255,255,0.2)',
            tickfont=dict(color='#94a3b8')
        )

        chart_json = json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))

        return _safe_jsonify({
            'success': True,
            'chart': chart_json
        })

    except Exception as e:
        return _safe_jsonify({'success': False, 'error': str(e)})

# ==========================================
# SEARCH CENTER - FIXED
# ==========================================

@app.route('/search')
@login_required
def search():
    query = request.args.get('q', '').strip()
    category = request.args.get('category', 'all')

    valid_categories = ['all', 'datasets', 'forecasts', 'reports', 'analyses']
    if category not in valid_categories:
        category = 'all'

    results = []

    if query and MODULES_AVAILABLE['models']:
        escaped_query = query.replace('%', '\\%').replace('_', '\\_')
        search_pattern = f'%{escaped_query}%'

        if category in ['all', 'datasets']:
            try:
                datasets = Dataset.query.filter(
                    Dataset.user_id == current_user.id
                ).filter(
                    db.or_(
                        Dataset.original_filename.ilike(search_pattern),
                        db.func.coalesce(Dataset.description, '').ilike(search_pattern),
                        db.func.coalesce(Dataset.tags, '').ilike(search_pattern)
                    )
                ).order_by(Dataset.created_at.desc()).all()

                for d in datasets:
                    results.append({
                        'type': 'dataset', 
                        'title': d.original_filename, 
                        'id': d.id, 
                        'created': d.created_at,
                        'subtitle': f"{d.row_count or 0} rows · {d.file_type.upper()}"
                    })
            except Exception as e:
                app.logger.error(f"Search datasets error: {e}")

        if category in ['all', 'forecasts']:
            try:
                forecasts = Forecast.query.filter(
                    Forecast.user_id == current_user.id
                ).filter(
                    Forecast.name.ilike(search_pattern)
                ).order_by(Forecast.created_at.desc()).all()

                for f in forecasts:
                    results.append({
                        'type': 'forecast', 
                        'title': f.name, 
                        'id': f.id, 
                        'created': f.created_at,
                        'subtitle': f"{f.model_used} · {f.forecast_horizon} days"
                    })
            except Exception as e:
                app.logger.error(f"Search forecasts error: {e}")

        if category in ['all', 'reports']:
            try:
                reports_list = Report.query.filter(
                    Report.user_id == current_user.id
                ).filter(
                    db.or_(
                        Report.title.ilike(search_pattern),
                        db.func.coalesce(Report.summary, '').ilike(search_pattern)
                    )
                ).order_by(Report.created_at.desc()).all()

                for r in reports_list:
                    results.append({
                        'type': 'report', 
                        'title': r.title, 
                        'id': r.id, 
                        'created': r.created_at,
                        'subtitle': r.report_type.title()
                    })
            except Exception as e:
                app.logger.error(f"Search reports error: {e}")

        if category in ['all', 'analyses']:
            try:
                analyses = CompetitorAnalysis.query.filter(
                    CompetitorAnalysis.user_id == current_user.id
                ).filter(
                    db.or_(
                        CompetitorAnalysis.competitor_name.ilike(search_pattern),
                        db.func.coalesce(CompetitorAnalysis.insights, '').ilike(search_pattern)
                    )
                ).order_by(CompetitorAnalysis.created_at.desc()).all()

                for a in analyses:
                    results.append({
                        'type': 'analysis', 
                        'title': a.competitor_name, 
                        'id': a.id, 
                        'created': a.created_at,
                        'subtitle': f"Threat Score: {a.threat_score:.1f}"
                    })
            except Exception as e:
                app.logger.error(f"Search analyses error: {e}")

    return render_template('search/index.html', results=results, query=query, category=category)

# ==========================================
# ADMIN CENTER
# ==========================================

@app.route('/admin')
@login_required
def admin_dashboard():
    if not current_user.is_admin:
        abort(403)

    if not MODULES_AVAILABLE['models']:
        return render_template('admin/dashboard.html', stats={}, recent_users=[], recent_activity=[], db_info={})

    stats = {
        'total_users': User.query.count(),
        'active_users': User.query.filter_by(is_active=True).count(),
        'total_datasets': Dataset.query.count(),
        'total_forecasts': Forecast.query.count(),
        'total_reports': Report.query.count(),
        'total_analyses': CompetitorAnalysis.query.count(),
        'total_opportunities': Opportunity.query.count(),
        'recent_logins': LoginHistory.query.filter(
            LoginHistory.created_at >= datetime.now() - timedelta(days=7)
        ).count(),
        'activities_today': ActivityLog.query.filter(
            ActivityLog.created_at >= datetime.now() - timedelta(days=1)
        ).count()
    }

    recent_users = User.query.order_by(User.created_at.desc()).limit(10).all()
    recent_activity = ActivityLog.query.order_by(ActivityLog.created_at.desc()).limit(20).all()

    db_info = {
        'size': get_db_size('marketmind.db'),
        'tables': ['users', 'datasets', 'forecasts', 'reports', 'competitor_analyses',
                  'consumer_insights', 'opportunities', 'scenario_simulations',
                  'activity_logs', 'login_history', 'analytics_cache', 'system_settings']
    }

    return render_template('admin/dashboard.html', stats=stats, recent_users=recent_users,
                         recent_activity=recent_activity, db_info=db_info)

@app.route('/admin/users')
@login_required
def admin_users():
    if not current_user.is_admin:
        abort(403)

    if not MODULES_AVAILABLE['models']:
        return render_template('admin/users.html', users=[], page=1, total=0, total_pages=1)

    page = request.args.get('page', 1, type=int)
    per_page = 25

    users_query = User.query.order_by(User.created_at.desc())
    total = users_query.count()
    users = users_query.offset((page - 1) * per_page).limit(per_page).all()

    return render_template('admin/users.html', users=users, page=page, 
                         total=total, total_pages=(total + per_page - 1) // per_page)

@app.route('/admin/user/<int:user_id>/toggle', methods=['POST'])
@login_required
def admin_toggle_user(user_id):
    if not current_user.is_admin:
        abort(403)

    if not MODULES_AVAILABLE['models']:
        flash('Database not available.', 'error')
        return redirect(url_for('admin_users'))

    user = User.query.get_or_404(user_id)
    if user.id == current_user.id:
        flash('Cannot deactivate yourself.', 'error')
        return redirect(url_for('admin_users'))

    user.is_active = not user.is_active
    db.session.commit()

    flash(f"User {user.username} {'activated' if user.is_active else 'deactivated'}.", 'success')
    return redirect(url_for('admin_users'))

@app.route('/admin/datasets')
@login_required
def admin_datasets():
    if not current_user.is_admin:
        abort(403)

    if not MODULES_AVAILABLE['models']:
        return render_template('admin/datasets.html', datasets=[])

    datasets = Dataset.query.order_by(Dataset.created_at.desc()).limit(100).all()
    return render_template('admin/datasets.html', datasets=datasets)

@app.route('/admin/activity')
@login_required
def admin_activity():
    if not current_user.is_admin:
        abort(403)

    if not MODULES_AVAILABLE['models']:
        return render_template('admin/activity.html', logs=[])

    logs = ActivityLog.query.order_by(ActivityLog.created_at.desc()).limit(200).all()
    return render_template('admin/activity.html', logs=logs)

@app.route('/admin/login-history')
@login_required
def admin_login_history():
    if not current_user.is_admin:
        abort(403)

    if not MODULES_AVAILABLE['models']:
        return render_template('admin/login_history.html', history=[])

    history = LoginHistory.query.order_by(LoginHistory.created_at.desc()).limit(200).all()
    return render_template('admin/login_history.html', history=history)

@app.route('/admin/database')
@login_required
def admin_database():
    if not current_user.is_admin:
        abort(403)

    if not MODULES_AVAILABLE['models']:
        return render_template('admin/database.html', tables={}, storage={})

    tables = {
        'users': {
            'count': User.query.count(),
            'columns': ['id', 'username', 'email', 'role', 'is_active', 'created_at']
        },
        'datasets': {
            'count': Dataset.query.count(),
            'columns': ['id', 'user_id', 'filename', 'row_count', 'processing_status', 'created_at']
        },
        'forecasts': {
            'count': Forecast.query.count(),
            'columns': ['id', 'user_id', 'name', 'model_used', 'forecast_horizon', 'created_at']
        },
        'reports': {
            'count': Report.query.count(),
            'columns': ['id', 'user_id', 'title', 'report_type', 'format', 'created_at']
        },
        'activity_logs': {
            'count': ActivityLog.query.count(),
            'columns': ['id', 'user_id', 'action', 'ip_address', 'created_at']
        },
        'login_history': {
            'count': LoginHistory.query.count(),
            'columns': ['id', 'user_id', 'ip_address', 'status', 'created_at']
        }
    }

    storage = {
        'database': get_db_size('marketmind.db'),
        'uploads': get_directory_size(app.config['UPLOAD_FOLDER']),
        'reports': get_directory_size(app.config['REPORTS_FOLDER']),
        'models': get_directory_size(app.config['MODELS_FOLDER'])
    }

    return render_template('admin/database.html', tables=tables, storage=storage)

@app.route('/admin/export/<table>')
@login_required
def admin_export_table(table):
    if not current_user.is_admin:
        abort(403)

    if not MODULES_AVAILABLE['models']:
        flash('Database not available.', 'error')
        return redirect(url_for('admin_database'))

    try:
        model_map = {
            'users': User,
            'datasets': Dataset,
            'forecasts': Forecast,
            'reports': Report,
            'activity_logs': ActivityLog,
            'login_history': LoginHistory
        }

        model = model_map.get(table)
        if not model:
            flash('Invalid table name.', 'error')
            return redirect(url_for('admin_database'))

        records = model.query.all()
        data = [r.to_dict() for r in records]
        df = pd.DataFrame(data)

        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)

        return send_file(
            io.BytesIO(output.getvalue().encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'{table}_export_{datetime.now().strftime("%Y%m%d")}.csv'
        )

    except Exception as e:
        flash(f'Export error: {str(e)}', 'error')
        return redirect(url_for('admin_database'))


# ==========================================
# DATA API ENDPOINTS
# ==========================================

@app.route('/api/datasets/<int:dataset_id>/preview')
@login_required
def preview_dataset(dataset_id):
    if not MODULES_AVAILABLE['models']:
        return _safe_jsonify({'success': False, 'error': 'Database not available'})

    dataset = Dataset.query.filter_by(id=dataset_id, user_id=current_user.id).first_or_404()

    try:
        ext = dataset.filename.rsplit('.', 1)[1].lower()
        if ext == 'csv':
            df = pd.read_csv(dataset.file_path)
        else:
            df = pd.read_excel(dataset.file_path)

        df = clean_dataframe(df)
        preview = df.head(50).to_dict('records')

        return _safe_jsonify({
            'success': True,
            'columns': df.columns.tolist(),
            'dtypes': {col: str(df[col].dtype) for col in df.columns},
            'preview': preview,
            'total_rows': len(df),
            'total_cols': len(df.columns)
        })

    except Exception as e:
        return _safe_jsonify({'success': False, 'error': str(e)})

@app.route('/api/chart/correlation/<int:dataset_id>')
@login_required
def correlation_chart(dataset_id):
    if not PLOTLY_AVAILABLE or not MODULES_AVAILABLE['models']:
        return _safe_jsonify({'success': False, 'error': 'Chart generation not available'})

    dataset = Dataset.query.filter_by(id=dataset_id, user_id=current_user.id).first_or_404()

    try:
        ext = dataset.filename.rsplit('.', 1)[1].lower()
        if ext == 'csv':
            df = pd.read_csv(dataset.file_path)
        else:
            df = pd.read_excel(dataset.file_path)

        df = clean_dataframe(df)

        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) < 2:
            return _safe_jsonify({'success': False, 'error': 'Need at least 2 numeric columns'})

        corr = numeric_df.corr()

        fig = px.imshow(corr, text_auto='.2f', aspect='auto',
                       color_continuous_scale='RdBu',
                       title='Correlation Heatmap')
        fig.update_layout(
            template='plotly_dark',
            height=600,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(30,30,50,0.5)',
            font=dict(color='#e2e8f0')
        )

        return _safe_jsonify({
            'success': True,
            'chart': json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))
        })

    except Exception as e:
        return _safe_jsonify({'success': False, 'error': str(e)})

# ==========================================
# MAIN APPLICATION ENTRY - HUGGING FACE SPACES READY
# ==========================================

if __name__ == '__main__':
    # Hugging Face Spaces requires binding to 0.0.0.0
    # and using the PORT environment variable (default 7860 for HF, but we use 5000)
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'

    # CRITICAL: Bind to 0.0.0.0 for containerized environments (Docker/HuggingFace)
    # This allows external access to the container
    app.run(host='0.0.0.0', port=port, debug=debug)