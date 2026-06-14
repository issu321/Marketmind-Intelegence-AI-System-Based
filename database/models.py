"""
MarketMind - Database Models
SQLAlchemy ORM models for all database entities.
"""

import json
import uuid
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()


class User(UserMixin, db.Model):
    """User model with authentication and profile management."""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    first_name = db.Column(db.String(50), nullable=True)
    last_name = db.Column(db.String(50), nullable=True)
    role = db.Column(db.String(20), default='user')
    is_active = db.Column(db.Boolean, default=True)
    is_verified = db.Column(db.Boolean, default=False)
    avatar_url = db.Column(db.String(255), nullable=True)
    bio = db.Column(db.Text, nullable=True)
    company = db.Column(db.String(100), nullable=True)
    job_title = db.Column(db.String(100), nullable=True)
    phone = db.Column(db.String(20), nullable=True)
    theme_preference = db.Column(db.String(10), default='dark')
    language = db.Column(db.String(10), default='en')
    timezone = db.Column(db.String(50), default='UTC')
    date_format = db.Column(db.String(20), default='YYYY-MM-DD')
    notification_enabled = db.Column(db.Boolean, default=True)
    two_factor_enabled = db.Column(db.Boolean, default=False)
    api_key = db.Column(db.String(64), unique=True, nullable=True)
    last_login = db.Column(db.DateTime, nullable=True)
    login_count = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)
    
    datasets = db.relationship('Dataset', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    reports = db.relationship('Report', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    forecasts = db.relationship('Forecast', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    activity_logs = db.relationship('ActivityLog', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    login_history = db.relationship('LoginHistory', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password, method='pbkdf2:sha256', salt_length=16)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    @property
    def is_admin(self):
        return self.role in ['admin', 'superadmin']
    
    @property
    def is_superadmin(self):
        return self.role == 'superadmin'
    
    @property
    def full_name(self):
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        return self.username
    
    def generate_api_key(self):
        self.api_key = uuid.uuid4().hex
        return self.api_key
    
    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'full_name': self.full_name,
            'role': self.role,
            'is_active': self.is_active,
            'is_verified': self.is_verified,
            'company': self.company,
            'job_title': self.job_title,
            'theme_preference': self.theme_preference,
            'language': self.language,
            'timezone': self.timezone,
            'notification_enabled': self.notification_enabled,
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'login_count': self.login_count,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
    
    def __repr__(self):
        return f'<User {self.username}>'


class Dataset(db.Model):
    """Dataset model for uploaded data files."""
    __tablename__ = 'datasets'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(500), nullable=False)
    file_size = db.Column(db.Integer, default=0)
    file_type = db.Column(db.String(20), nullable=False)
    row_count = db.Column(db.Integer, default=0)
    column_count = db.Column(db.Integer, default=0)
    columns = db.Column(db.Text, nullable=True)
    column_types = db.Column(db.Text, nullable=True)
    date_column = db.Column(db.String(100), nullable=True)
    value_column = db.Column(db.String(100), nullable=True)
    category_column = db.Column(db.String(100), nullable=True)
    summary_stats = db.Column(db.Text, nullable=True)
    data_quality_score = db.Column(db.Float, default=0.0)
    quality_issues = db.Column(db.Text, nullable=True)
    processing_status = db.Column(db.String(20), default='pending')
    is_processed = db.Column(db.Boolean, default=False)
    description = db.Column(db.Text, nullable=True)
    tags = db.Column(db.String(255), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)
    
    def set_columns(self, columns_list):
        self.columns = json.dumps(columns_list)
    
    def get_columns(self):
        return json.loads(self.columns) if self.columns else []
    
    def set_column_types(self, types_dict):
        self.column_types = json.dumps(types_dict)
    
    def get_column_types(self):
        return json.loads(self.column_types) if self.column_types else {}
    
    def set_summary_stats(self, stats_dict):
        self.summary_stats = json.dumps(stats_dict)
    
    def get_summary_stats(self):
        return json.loads(self.summary_stats) if self.summary_stats else {}
    
    def set_quality_issues(self, issues_list):
        self.quality_issues = json.dumps(issues_list)
    
    def get_quality_issues(self):
        return json.loads(self.quality_issues) if self.quality_issues else []
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'filename': self.filename,
            'original_filename': self.original_filename,
            'file_size': self.file_size,
            'file_type': self.file_type,
            'row_count': self.row_count,
            'column_count': self.column_count,
            'columns': self.get_columns(),
            'column_types': self.get_column_types(),
            'date_column': self.date_column,
            'value_column': self.value_column,
            'category_column': self.category_column,
            'data_quality_score': self.data_quality_score,
            'quality_issues': self.get_quality_issues(),
            'processing_status': self.processing_status,
            'is_processed': self.is_processed,
            'description': self.description,
            'tags': self.tags,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
    
    def __repr__(self):
        return f'<Dataset {self.original_filename}>'


class Forecast(db.Model):
    """Forecast model for ML predictions."""
    __tablename__ = 'forecasts'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey('datasets.id'), nullable=True)
    name = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text, nullable=True)
    forecast_type = db.Column(db.String(50), nullable=False)
    model_used = db.Column(db.String(50), nullable=False)
    forecast_horizon = db.Column(db.Integer, default=30)
    confidence_level = db.Column(db.Float, default=0.95)
    metrics = db.Column(db.Text, nullable=True)
    forecast_data = db.Column(db.Text, nullable=True)
    feature_importance = db.Column(db.Text, nullable=True)
    model_performance = db.Column(db.Text, nullable=True)
    backtest_results = db.Column(db.Text, nullable=True)
    insights = db.Column(db.Text, nullable=True)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.now)
    
    def set_metrics(self, metrics_dict):
        self.metrics = json.dumps(metrics_dict)
    
    def get_metrics(self):
        return json.loads(self.metrics) if self.metrics else {}
    
    def set_forecast_data(self, data):
        self.forecast_data = json.dumps(data)
    
    def get_forecast_data(self):
        return json.loads(self.forecast_data) if self.forecast_data else []
    
    def set_feature_importance(self, data):
        self.feature_importance = json.dumps(data)
    
    def get_feature_importance(self):
        return json.loads(self.feature_importance) if self.feature_importance else {}
    
    def set_model_performance(self, data):
        self.model_performance = json.dumps(data)
    
    def get_model_performance(self):
        return json.loads(self.model_performance) if self.model_performance else {}
    
    def set_backtest_results(self, data):
        self.backtest_results = json.dumps(data)
    
    def get_backtest_results(self):
        return json.loads(self.backtest_results) if self.backtest_results else []
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'dataset_id': self.dataset_id,
            'name': self.name,
            'description': self.description,
            'forecast_type': self.forecast_type,
            'model_used': self.model_used,
            'forecast_horizon': self.forecast_horizon,
            'confidence_level': self.confidence_level,
            'metrics': self.get_metrics(),
            'forecast_data': self.get_forecast_data(),
            'feature_importance': self.get_feature_importance(),
            'model_performance': self.get_model_performance(),
            'insights': self.insights,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    def __repr__(self):
        return f'<Forecast {self.name}>'


class Report(db.Model):
    """Report model for generated reports."""
    __tablename__ = 'reports'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey('datasets.id'), nullable=True)
    forecast_id = db.Column(db.Integer, db.ForeignKey('forecasts.id'), nullable=True)
    title = db.Column(db.String(255), nullable=False)
    report_type = db.Column(db.String(50), nullable=False)
    format = db.Column(db.String(20), default='html')
    file_path = db.Column(db.String(500), nullable=True)
    content = db.Column(db.Text, nullable=True)
    summary = db.Column(db.Text, nullable=True)
    key_findings = db.Column(db.Text, nullable=True)
    recommendations = db.Column(db.Text, nullable=True)
    charts_data = db.Column(db.Text, nullable=True)
    kpis = db.Column(db.Text, nullable=True)
    is_favorite = db.Column(db.Boolean, default=False)
    download_count = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.now)
    
    def set_key_findings(self, findings):
        self.key_findings = json.dumps(findings)
    
    def get_key_findings(self):
        return json.loads(self.key_findings) if self.key_findings else []
    
    def set_recommendations(self, recs):
        self.recommendations = json.dumps(recs)
    
    def get_recommendations(self):
        return json.loads(self.recommendations) if self.recommendations else []
    
    def set_charts_data(self, data):
        self.charts_data = json.dumps(data)
    
    def get_charts_data(self):
        return json.loads(self.charts_data) if self.charts_data else {}
    
    def set_kpis(self, data):
        self.kpis = json.dumps(data)
    
    def get_kpis(self):
        return json.loads(self.kpis) if self.kpis else {}
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'dataset_id': self.dataset_id,
            'forecast_id': self.forecast_id,
            'title': self.title,
            'report_type': self.report_type,
            'format': self.format,
            'summary': self.summary,
            'key_findings': self.get_key_findings(),
            'recommendations': self.get_recommendations(),
            'is_favorite': self.is_favorite,
            'download_count': self.download_count,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    def __repr__(self):
        return f'<Report {self.title}>'


class CompetitorAnalysis(db.Model):
    """Competitor analysis model."""
    __tablename__ = 'competitor_analyses'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey('datasets.id'), nullable=True)
    competitor_name = db.Column(db.String(255), nullable=False)
    analysis_type = db.Column(db.String(50), nullable=False)
    overall_score = db.Column(db.Float, default=0.0)
    threat_score = db.Column(db.Float, default=0.0)
    growth_score = db.Column(db.Float, default=0.0)
    market_position_score = db.Column(db.Float, default=0.0)
    innovation_score = db.Column(db.Float, default=0.0)
    pricing_score = db.Column(db.Float, default=0.0)
    sentiment_score = db.Column(db.Float, default=0.0)
    strengths = db.Column(db.Text, nullable=True)
    weaknesses = db.Column(db.Text, nullable=True)
    opportunities = db.Column(db.Text, nullable=True)
    threats = db.Column(db.Text, nullable=True)
    analysis_data = db.Column(db.Text, nullable=True)
    insights = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.now)
    
    def set_strengths(self, data):
        self.strengths = json.dumps(data)
    
    def get_strengths(self):
        return json.loads(self.strengths) if self.strengths else []
    
    def set_weaknesses(self, data):
        self.weaknesses = json.dumps(data)
    
    def get_weaknesses(self):
        return json.loads(self.weaknesses) if self.weaknesses else []
    
    def set_opportunities(self, data):
        self.opportunities = json.dumps(data)
    
    def get_opportunities(self):
        return json.loads(self.opportunities) if self.opportunities else []
    
    def set_threats(self, data):
        self.threats = json.dumps(data)
    
    def get_threats(self):
        return json.loads(self.threats) if self.threats else []
    
    def set_analysis_data(self, data):
        self.analysis_data = json.dumps(data)
    
    def get_analysis_data(self):
        return json.loads(self.analysis_data) if self.analysis_data else {}
    
    def to_dict(self):
        return {
            'id': self.id,
            'competitor_name': self.competitor_name,
            'analysis_type': self.analysis_type,
            'overall_score': self.overall_score,
            'threat_score': self.threat_score,
            'growth_score': self.growth_score,
            'market_position_score': self.market_position_score,
            'innovation_score': self.innovation_score,
            'pricing_score': self.pricing_score,
            'sentiment_score': self.sentiment_score,
            'strengths': self.get_strengths(),
            'weaknesses': self.get_weaknesses(),
            'opportunities': self.get_opportunities(),
            'threats': self.get_threats(),
            'insights': self.insights,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    def __repr__(self):
        return f'<CompetitorAnalysis {self.competitor_name}>'


class ConsumerInsight(db.Model):
    """Consumer insight model."""
    __tablename__ = 'consumer_insights'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey('datasets.id'), nullable=True)
    insight_type = db.Column(db.String(50), nullable=False)
    positive_score = db.Column(db.Float, default=0.0)
    negative_score = db.Column(db.Float, default=0.0)
    neutral_score = db.Column(db.Float, default=0.0)
    brand_health_score = db.Column(db.Float, default=0.0)
    emotion_score = db.Column(db.Float, default=0.0)
    trust_score = db.Column(db.Float, default=0.0)
    sentiment_distribution = db.Column(db.Text, nullable=True)
    key_topics = db.Column(db.Text, nullable=True)
    emotion_breakdown = db.Column(db.Text, nullable=True)
    insights = db.Column(db.Text, nullable=True)
    recommendations = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.now)
    
    def set_sentiment_distribution(self, data):
        self.sentiment_distribution = json.dumps(data)
    
    def get_sentiment_distribution(self):
        return json.loads(self.sentiment_distribution) if self.sentiment_distribution else {}
    
    def set_key_topics(self, data):
        self.key_topics = json.dumps(data)
    
    def get_key_topics(self):
        return json.loads(self.key_topics) if self.key_topics else []
    
    def set_emotion_breakdown(self, data):
        self.emotion_breakdown = json.dumps(data)
    
    def get_emotion_breakdown(self):
        return json.loads(self.emotion_breakdown) if self.emotion_breakdown else {}
    
    def to_dict(self):
        return {
            'id': self.id,
            'insight_type': self.insight_type,
            'positive_score': self.positive_score,
            'negative_score': self.negative_score,
            'neutral_score': self.neutral_score,
            'brand_health_score': self.brand_health_score,
            'emotion_score': self.emotion_score,
            'trust_score': self.trust_score,
            'sentiment_distribution': self.get_sentiment_distribution(),
            'key_topics': self.get_key_topics(),
            'emotion_breakdown': self.get_emotion_breakdown(),
            'insights': self.insights,
            'recommendations': self.recommendations,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    def __repr__(self):
        return f'<ConsumerInsight {self.insight_type}>'


class Opportunity(db.Model):
    """Business opportunity model."""
    __tablename__ = 'opportunities'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey('datasets.id'), nullable=True)
    opportunity_name = db.Column(db.String(255), nullable=False)
    opportunity_type = db.Column(db.String(50), nullable=False)
    opportunity_score = db.Column(db.Float, default=0.0)
    revenue_potential = db.Column(db.Float, default=0.0)
    market_readiness_score = db.Column(db.Float, default=0.0)
    investment_score = db.Column(db.Float, default=0.0)
    risk_level = db.Column(db.String(20), default='medium')
    time_to_market = db.Column(db.String(50), nullable=True)
    description = db.Column(db.Text, nullable=True)
    supporting_data = db.Column(db.Text, nullable=True)
    recommendations = db.Column(db.Text, nullable=True)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.now)
    
    def set_supporting_data(self, data):
        self.supporting_data = json.dumps(data)
    
    def get_supporting_data(self):
        return json.loads(self.supporting_data) if self.supporting_data else {}
    
    def to_dict(self):
        return {
            'id': self.id,
            'opportunity_name': self.opportunity_name,
            'opportunity_type': self.opportunity_type,
            'opportunity_score': self.opportunity_score,
            'revenue_potential': self.revenue_potential,
            'market_readiness_score': self.market_readiness_score,
            'investment_score': self.investment_score,
            'risk_level': self.risk_level,
            'time_to_market': self.time_to_market,
            'description': self.description,
            'supporting_data': self.get_supporting_data(),
            'recommendations': self.recommendations,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    def __repr__(self):
        return f'<Opportunity {self.opportunity_name}>'


class ScenarioSimulation(db.Model):
    """Scenario simulation model."""
    __tablename__ = 'scenario_simulations'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey('datasets.id'), nullable=True)
    name = db.Column(db.String(255), nullable=False)
    scenario_type = db.Column(db.String(50), nullable=False)
    parameters = db.Column(db.Text, nullable=True)
    baseline_metrics = db.Column(db.Text, nullable=True)
    simulated_metrics = db.Column(db.Text, nullable=True)
    impact_analysis = db.Column(db.Text, nullable=True)
    comparison_chart = db.Column(db.Text, nullable=True)
    recommendations = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.now)
    
    def set_parameters(self, data):
        self.parameters = json.dumps(data)
    
    def get_parameters(self):
        return json.loads(self.parameters) if self.parameters else {}
    
    def set_baseline_metrics(self, data):
        self.baseline_metrics = json.dumps(data)
    
    def get_baseline_metrics(self):
        return json.loads(self.baseline_metrics) if self.baseline_metrics else {}
    
    def set_simulated_metrics(self, data):
        self.simulated_metrics = json.dumps(data)
    
    def get_simulated_metrics(self):
        return json.loads(self.simulated_metrics) if self.simulated_metrics else {}
    
    def set_impact_analysis(self, data):
        self.impact_analysis = json.dumps(data)
    
    def get_impact_analysis(self):
        return json.loads(self.impact_analysis) if self.impact_analysis else {}
    
    def set_comparison_chart(self, data):
        self.comparison_chart = json.dumps(data)
    
    def get_comparison_chart(self):
        return json.loads(self.comparison_chart) if self.comparison_chart else {}
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'scenario_type': self.scenario_type,
            'parameters': self.get_parameters(),
            'baseline_metrics': self.get_baseline_metrics(),
            'simulated_metrics': self.get_simulated_metrics(),
            'impact_analysis': self.get_impact_analysis(),
            'recommendations': self.recommendations,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    def __repr__(self):
        return f'<ScenarioSimulation {self.name}>'


class ActivityLog(db.Model):
    """User activity log model."""
    __tablename__ = 'activity_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    action = db.Column(db.String(100), nullable=False)
    details = db.Column(db.Text, nullable=True)
    ip_address = db.Column(db.String(45), nullable=True)
    user_agent = db.Column(db.String(255), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.now)
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'username': self.user.username if self.user else None,
            'action': self.action,
            'details': self.details,
            'ip_address': self.ip_address,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    def __repr__(self):
        return f'<ActivityLog {self.action}>'


class LoginHistory(db.Model):
    """Login history model."""
    __tablename__ = 'login_history'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    ip_address = db.Column(db.String(45), nullable=True)
    user_agent = db.Column(db.String(255), nullable=True)
    status = db.Column(db.String(20), default='success')
    failure_reason = db.Column(db.String(100), nullable=True)
    location = db.Column(db.String(100), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.now)
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'username': self.user.username if self.user else None,
            'ip_address': self.ip_address,
            'status': self.status,
            'failure_reason': self.failure_reason,
            'location': self.location,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    def __repr__(self):
        return f'<LoginHistory {self.status}>'


class AnalyticsCache(db.Model):
    """Analytics cache model for temporary results."""
    __tablename__ = 'analytics_cache'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey('datasets.id'), nullable=True)
    cache_type = db.Column(db.String(50), nullable=False)
    cache_key = db.Column(db.String(255), nullable=False)
    cache_data = db.Column(db.Text, nullable=True)
    expires_at = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.now)
    
    def set_cache_data(self, data):
        self.cache_data = json.dumps(data)
    
    def get_cache_data(self):
        return json.loads(self.cache_data) if self.cache_data else {}
    
    def is_expired(self):
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at
    
    def __repr__(self):
        return f'<AnalyticsCache {self.cache_type}>'


class SystemSetting(db.Model):
    """System settings model."""
    __tablename__ = 'system_settings'
    
    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String(100), unique=True, nullable=False)
    value = db.Column(db.Text, nullable=True)
    description = db.Column(db.String(255), nullable=True)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)
    
    def __repr__(self):
        return f'<SystemSetting {self.key}>'
