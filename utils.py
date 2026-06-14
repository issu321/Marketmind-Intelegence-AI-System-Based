"""
MarketMind - Utilities Module
Provides helper functions, validators, formatters, and data processing utilities.
"""

import os
import re
import json
import hashlib
import secrets
import uuid
import math
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from functools import wraps

import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename


ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls', 'json'}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

DATASET_SCHEMA = {
    'date': ['date', 'datetime', 'timestamp', 'time', 'period', 'day', 'month', 'year'],
    'value': ['value', 'sales', 'revenue', 'amount', 'price', 'cost', 'quantity', 'demand', 
              'units', 'orders', 'customers', 'profit', 'income', 'volume', 'count'],
    'category': ['category', 'segment', 'product', 'region', 'channel', 'type', 'group', 
                 'class', 'department', 'market', 'industry'],
    'competitor': ['competitor', 'competitor_name', 'rival', 'company', 'brand', 'vendor', 'supplier'],
    'sentiment': ['sentiment', 'review', 'rating', 'score', 'feedback', 'comment', 'opinion'],
    'feature': ['feature', 'attribute', 'spec', 'capability', 'function', 'service']
}


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def validate_file_size(file_obj) -> bool:
    """Validate file size is within limits."""
    file_obj.seek(0, os.SEEK_END)
    size = file_obj.tell()
    file_obj.seek(0)
    return size <= MAX_FILE_SIZE


def generate_unique_filename(original_filename: str) -> str:
    """Generate a unique filename with secure name."""
    ext = original_filename.rsplit('.', 1)[1].lower() if '.' in original_filename else 'csv'
    unique_id = secrets.token_hex(8)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{timestamp}_{unique_id}.{ext}"


def generate_id() -> str:
    """Generate a unique ID."""
    return str(uuid.uuid4())


def hash_password(password: str) -> str:
    """Hash a password using Werkzeug-compatible method."""
    from werkzeug.security import generate_password_hash
    return generate_password_hash(password, method='pbkdf2:sha256', salt_length=16)


def format_number(value: Union[int, float], decimals: int = 2) -> str:
    """Format a number with thousand separators."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.{decimals}f}B"
    elif abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.{decimals}f}M"
    elif abs(value) >= 1_000:
        return f"{value / 1_000:.{decimals}f}K"
    return f"{value:.{decimals}f}"


def format_currency(value: Union[int, float], decimals: int = 2) -> str:
    """Format a value as currency."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "$0.00"
    return f"${value:,.{decimals}f}"


def format_percentage(value: Union[int, float], decimals: int = 2) -> str:
    """Format a value as percentage."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "0.00%"
    return f"{value:.{decimals}f}%"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers."""
    try:
        if denominator == 0 or denominator is None:
            return default
        return numerator / denominator
    except (TypeError, ZeroDivisionError):
        return default


def calculate_growth_rate(current: float, previous: float) -> float:
    """Calculate growth rate between two values."""
    if previous is None or previous == 0:
        return 0.0
    try:
        return ((current - previous) / abs(previous)) * 100
    except (TypeError, ZeroDivisionError):
        return 0.0


def calculate_cagr(begin_value: float, end_value: float, periods: int) -> float:
    """Calculate Compound Annual Growth Rate."""
    if begin_value <= 0 or periods <= 0:
        return 0.0
    try:
        return ((end_value / begin_value) ** (1 / periods) - 1) * 100
    except (ValueError, ZeroDivisionError, OverflowError):
        return 0.0


def calculate_moving_average(data: List[float], window: int = 7) -> List[float]:
    """Calculate moving average."""
    if not data or window <= 0:
        return []
    result = []
    for i in range(len(data)):
        if i < window - 1:
            result.append(sum(data[:i+1]) / (i+1))
        else:
            result.append(sum(data[i-window+1:i+1]) / window)
    return result


def calculate_volatility(data: List[float]) -> float:
    """Calculate standard deviation (volatility)."""
    if not data or len(data) < 2:
        return 0.0
    try:
        return float(np.std(data, ddof=1))
    except Exception:
        return 0.0


def detect_outliers(data: List[float], threshold: float = 3.0) -> List[int]:
    """Detect outliers using Z-score method."""
    if not data or len(data) < 3:
        return []
    try:
        arr = np.array(data)
        mean = np.mean(arr)
        std = np.std(arr)
        if std == 0:
            return []
        z_scores = np.abs((arr - mean) / std)
        return [i for i, z in enumerate(z_scores) if z > threshold]
    except Exception:
        return []


def calculate_trend_direction(data: List[float]) -> str:
    """Calculate trend direction from data points."""
    if not data or len(data) < 2:
        return "insufficient_data"
    try:
        x = np.arange(len(data))
        y = np.array(data)
        slope = np.polyfit(x, y, 1)[0]
        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        return "stable"
    except Exception:
        return "unknown"


def calculate_seasonality_strength(data: List[float], period: int = 12) -> float:
    """Calculate seasonality strength in time series."""
    if not data or len(data) < period * 2:
        return 0.0
    try:
        arr = np.array(data)
        detrended = arr - calculate_moving_average(list(arr), period)
        var_residual = np.var(detrended)
        var_original = np.var(arr)
        if var_original == 0:
            return 0.0
        strength = 1 - (var_residual / var_original)
        return max(0.0, min(1.0, float(strength)))
    except Exception:
        return 0.0


def score_to_rating(score: float, max_score: float = 100) -> str:
    """Convert a score to a rating label."""
    ratio = (score / max_score) * 100 if max_score > 0 else 0
    if ratio >= 90:
        return "Excellent"
    elif ratio >= 75:
        return "Good"
    elif ratio >= 60:
        return "Average"
    elif ratio >= 40:
        return "Below Average"
    return "Poor"


def score_to_color(score: float, max_score: float = 100) -> str:
    """Convert a score to a color."""
    ratio = (score / max_score) * 100 if max_score > 0 else 0
    if ratio >= 90:
        return "#10b981"
    elif ratio >= 75:
        return "#22c55e"
    elif ratio >= 60:
        return "#f59e0b"
    elif ratio >= 40:
        return "#f97316"
    return "#ef4444"


def generate_color_palette(n: int) -> List[str]:
    """Generate a color palette for charts."""
    base_colors = [
        '#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6',
        '#ec4899', '#06b6d4', '#84cc16', '#f97316', '#6366f1',
        '#14b8a6', '#a855f7', '#e11d48', '#0ea5e9', '#d946ef'
    ]
    if n <= len(base_colors):
        return base_colors[:n]
    colors = base_colors.copy()
    for i in range(len(base_colors), n):
        hue = (i * 137.5) % 360
        colors.append(f"hsl({hue}, 70%, 50%)")
    return colors


def sanitize_string(value: str, max_length: int = 255) -> str:
    """Sanitize a string input."""
    if not value:
        return ""
    value = str(value).strip()
    value = re.sub(r'<[^>]+>', '', value)
    value = re.sub(r'[^\w\s\-\_\.\@]', '', value)
    return value[:max_length]


def validate_email(email: str) -> bool:
    """Validate email format."""
    if not email:
        return False
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def validate_username(username: str) -> bool:
    """Validate username format."""
    if not username or len(username) < 3 or len(username) > 50:
        return False
    return bool(re.match(r'^[a-zA-Z0-9_\-]+$', username))


def validate_password_strength(password: str) -> Tuple[bool, str]:
    """Validate password strength."""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"
    if not re.search(r'\d', password):
        return False, "Password must contain at least one digit"
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return False, "Password must contain at least one special character"
    return True, "Password is strong"


def parse_date_column(series: pd.Series) -> Optional[pd.Series]:
    """Attempt to parse a pandas series as dates."""
    try:
        return pd.to_datetime(series, errors='coerce')
    except Exception:
        return None


def infer_column_types(df: pd.DataFrame) -> Dict[str, str]:
    """Infer column types from a DataFrame."""
    types = {}
    for col in df.columns:
        col_lower = col.lower().strip()
        detected = False
        for ctype, keywords in DATASET_SCHEMA.items():
            if any(kw in col_lower for kw in keywords):
                types[col] = ctype
                detected = True
                break
        if not detected:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                types[col] = 'date'
            elif pd.api.types.is_numeric_dtype(df[col]):
                types[col] = 'value'
            elif df[col].dtype == 'object':
                unique_ratio = df[col].nunique() / len(df) if len(df) > 0 else 0
                if unique_ratio < 0.3 and df[col].nunique() < 50:
                    types[col] = 'category'
                else:
                    types[col] = 'text'
            else:
                types[col] = 'unknown'
    return types


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare a DataFrame for analysis."""
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna('Unknown')
    df = df.drop_duplicates()
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    return df


def generate_summary_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate summary statistics for a DataFrame."""
    stats = {
        'rows': len(df),
        'columns': len(df.columns),
        'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB",
        'column_stats': {}
    }
    for col in df.columns:
        col_stats = {'dtype': str(df[col].dtype), 'null_count': int(df[col].isnull().sum())}
        if pd.api.types.is_numeric_dtype(df[col]):
            col_stats.update({
                'mean': float(df[col].mean()) if not df[col].empty else 0,
                'median': float(df[col].median()) if not df[col].empty else 0,
                'std': float(df[col].std()) if not df[col].empty else 0,
                'min': float(df[col].min()) if not df[col].empty else 0,
                'max': float(df[col].max()) if not df[col].empty else 0
            })
        else:
            col_stats['unique_values'] = int(df[col].nunique())
        stats['column_stats'][col] = col_stats
    return stats


def create_date_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Create date-based features from a datetime column."""
    df = df.copy()
    if date_col not in df.columns:
        return df
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['quarter'] = df[date_col].dt.quarter
    df['day_of_week'] = df[date_col].dt.dayofweek
    df['day_of_year'] = df[date_col].dt.dayofyear
    df['week_of_year'] = df[date_col].dt.isocalendar().week.astype(int)
    df['is_month_start'] = df[date_col].dt.is_month_start.astype(int)
    df['is_month_end'] = df[date_col].dt.is_month_end.astype(int)
    df['is_quarter_start'] = df[date_col].dt.is_quarter_start.astype(int)
    df['days_from_start'] = (df[date_col] - df[date_col].min()).dt.days
    return df


def time_series_to_supervised(data: List[float], n_lags: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Convert time series to supervised learning format."""
    if not data or len(data) <= n_lags:
        return np.array([]), np.array([])
    X, y = [], []
    for i in range(n_lags, len(data)):
        X.append(data[i-n_lags:i])
        y.append(data[i])
    return np.array(X), np.array(y)


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to maximum length."""
    if not text or len(text) <= max_length:
        return text or ""
    return text[:max_length].rsplit(' ', 1)[0] + "..."


def time_ago(dt: datetime) -> str:
    """Convert datetime to human-readable 'time ago' string."""
    if not dt:
        return "Unknown"
    now = datetime.now()
    diff = now - dt
    seconds = int(diff.total_seconds())
    if seconds < 60:
        return "Just now"
    elif seconds < 3600:
        return f"{seconds // 60} minute{'s' if seconds // 60 != 1 else ''} ago"
    elif seconds < 86400:
        return f"{seconds // 3600} hour{'s' if seconds // 3600 != 1 else ''} ago"
    elif seconds < 604800:
        return f"{seconds // 86400} day{'s' if seconds // 86400 != 1 else ''} ago"
    elif seconds < 2592000:
        return f"{seconds // 604800} week{'s' if seconds // 604800 != 1 else ''} ago"
    return f"{seconds // 2592000} month{'s' if seconds // 2592000 != 1 else ''} ago"


def paginate_data(data: List[Any], page: int, per_page: int = 20) -> Dict[str, Any]:
    """Paginate a list of data."""
    total = len(data)
    total_pages = max(1, (total + per_page - 1) // per_page)
    page = max(1, min(page, total_pages))
    start = (page - 1) * per_page
    end = start + per_page
    return {
        'items': data[start:end],
        'page': page,
        'per_page': per_page,
        'total': total,
        'total_pages': total_pages,
        'has_prev': page > 1,
        'has_next': page < total_pages,
        'prev_page': page - 1 if page > 1 else None,
        'next_page': page + 1 if page < total_pages else None
    }


def dataframe_to_json(df: pd.DataFrame, orient: str = 'records') -> str:
    """Convert DataFrame to JSON string."""
    try:
        return df.to_json(orient=orient, date_format='iso')
    except Exception:
        return "[]"


def get_db_size(db_path: str) -> str:
    """Get database file size."""
    try:
        size = os.path.getsize(db_path)
        if size < 1024:
            return f"{size} B"
        elif size < 1024 * 1024:
            return f"{size / 1024:.2f} KB"
        elif size < 1024 * 1024 * 1024:
            return f"{size / (1024 * 1024):.2f} MB"
        return f"{size / (1024 * 1024 * 1024):.2f} GB"
    except Exception:
        return "Unknown"


def get_directory_size(dir_path: str) -> str:
    """Get total size of a directory."""
    try:
        total = 0
        for dirpath, dirnames, filenames in os.walk(dir_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total += os.path.getsize(fp)
        if total < 1024:
            return f"{total} B"
        elif total < 1024 * 1024:
            return f"{total / 1024:.2f} KB"
        elif total < 1024 * 1024 * 1024:
            return f"{total / (1024 * 1024):.2f} MB"
        return f"{total / (1024 * 1024 * 1024):.2f} GB"
    except Exception:
        return "Unknown"


def calculate_health_score(metrics: Dict[str, float]) -> float:
    """Calculate an overall health score from metrics."""
    if not metrics:
        return 0.0
    weights = {
        'growth': 0.25,
        'stability': 0.20,
        'profitability': 0.25,
        'efficiency': 0.15,
        'momentum': 0.15
    }
    score = 0.0
    total_weight = 0.0
    for key, weight in weights.items():
        if key in metrics and metrics[key] is not None:
            val = max(0.0, min(100.0, float(metrics[key])))
            score += val * weight
            total_weight += weight
    if total_weight == 0:
        avg = sum(v for v in metrics.values() if v is not None) / len([v for v in metrics.values() if v is not None])
        return max(0.0, min(100.0, avg))
    return max(0.0, min(100.0, score / total_weight))


def create_progress_bar_html(percentage: float, color: str = None) -> str:
    """Create an HTML progress bar."""
    if color is None:
        color = score_to_color(percentage)
    return f'<div class="progress-bar-container"><div class="progress-bar-fill" style="width:{percentage}%;background:{color}"></div></div>'


def generate_insight_text(metric_name: str, value: float, trend: str, context: str = "") -> str:
    """Generate human-readable insight text."""
    rating = score_to_rating(value)
    direction = "upward" if trend == "increasing" else "downward" if trend == "decreasing" else "stable"
    texts = [
        f"The {metric_name} is currently rated as {rating.lower()} at {value:.1f}.",
        f"The trend shows a {direction} trajectory.",
    ]
    if trend == "increasing" and value > 70:
        texts.append(f"This strong performance in {metric_name} indicates positive momentum.")
    elif trend == "decreasing" and value < 50:
        texts.append(f"Attention is needed as {metric_name} shows declining performance.")
    elif trend == "stable" and value > 60:
        texts.append(f"The consistent {metric_name} performance provides a solid foundation.")
    if context:
        texts.append(context)
    return " ".join(texts)


def merge_dicts(base: Dict, override: Dict) -> Dict:
    """Deep merge two dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def safe_json_loads(data: str, default: Any = None) -> Any:
    """Safely load JSON string."""
    try:
        return json.loads(data)
    except (json.JSONDecodeError, TypeError):
        return default if default is not None else {}


def chunk_list(data: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split a list into chunks."""
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]


def calculate_confidence_interval(data: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate confidence interval for data."""
    if not data or len(data) < 2:
        return 0.0, 0.0
    try:
        arr = np.array(data)
        mean = np.mean(arr)
        std = np.std(arr, ddof=1)
        n = len(arr)
        from scipy import stats
        t_val = stats.t.ppf((1 + confidence) / 2, n - 1)
        margin = t_val * (std / np.sqrt(n))
        return mean - margin, mean + margin
    except Exception:
        return mean, mean


def detect_data_quality_issues(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Detect data quality issues in a DataFrame."""
    issues = []
    for col in df.columns:
        null_count = df[col].isnull().sum()
        null_pct = (null_count / len(df)) * 100 if len(df) > 0 else 0
        if null_pct > 50:
            issues.append({'column': col, 'issue': 'high_missing', 'severity': 'critical', 
                          'details': f'{null_pct:.1f}% missing values'})
        elif null_pct > 20:
            issues.append({'column': col, 'issue': 'moderate_missing', 'severity': 'warning',
                          'details': f'{null_pct:.1f}% missing values'})
        elif null_pct > 0:
            issues.append({'column': col, 'issue': 'low_missing', 'severity': 'info',
                          'details': f'{null_pct:.1f}% missing values'})
        if pd.api.types.is_numeric_dtype(df[col]):
            outliers = detect_outliers(df[col].dropna().tolist())
            if len(outliers) > len(df) * 0.05:
                issues.append({'column': col, 'issue': 'outliers', 'severity': 'warning',
                              'details': f'{len(outliers)} outliers detected'})
            if df[col].nunique() <= 1:
                issues.append({'column': col, 'issue': 'constant', 'severity': 'info',
                              'details': 'Constant value column'})
    if len(df) == 0:
        issues.append({'column': 'dataset', 'issue': 'empty', 'severity': 'critical',
                      'details': 'Dataset is empty'})
    elif len(df.columns) == 0:
        issues.append({'column': 'dataset', 'issue': 'no_columns', 'severity': 'critical',
                      'details': 'No columns in dataset'})
    return issues


def require_login(f):
    """Decorator to require login for a route."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        from flask_login import current_user
        if not current_user.is_authenticated:
            from flask import redirect, url_for, request
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function


def require_admin(f):
    """Decorator to require admin role for a route."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        from flask_login import current_user
        if not current_user.is_authenticated or not getattr(current_user, 'is_admin', False):
            from flask import abort
            abort(403)
        return f(*args, **kwargs)
    return decorated_function


def log_activity(user_id: int, action: str, details: str = "", ip_address: str = None):
    """Log user activity."""
    try:
        from app import db
        from database.models import ActivityLog
        log = ActivityLog(
            user_id=user_id,
            action=action,
            details=details,
            ip_address=ip_address,
            created_at=datetime.now()
        )
        db.session.add(log)
        db.session.commit()
    except Exception:
        pass


def clear_analytics_cache(user_id: int):
    """Clear analytics cache for a user."""
    try:
        from app import db
        from database.models import AnalyticsCache
        AnalyticsCache.query.filter_by(user_id=user_id).delete()
        db.session.commit()
    except Exception:
        pass
