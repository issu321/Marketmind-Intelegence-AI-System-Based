"""
MarketMind - Data Analysis & Consumer Intelligence Engine
Provides data analysis, consumer sentiment analysis, opportunity detection,
and business intelligence analytics.
"""

import re
import json
import warnings
import traceback
from collections import Counter
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
from textblob import TextBlob
from scipy import stats
from scipy.signal import find_peaks

warnings.filterwarnings('ignore')

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False


class DataAnalyzer:
    """Enterprise data analysis engine."""
    
    def __init__(self):
        self.analysis_results = {}
    
    def analyze_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive dataset analysis."""
        try:
            df = df.copy()
            
            # Basic info
            analysis = {
                'basic_info': {
                    'row_count': len(df),
                    'column_count': len(df.columns),
                    'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB",
                    'dtypes': {col: str(df[col].dtype) for col in df.columns}
                },
                'columns': []
            }
            
            # Column analysis
            for col in df.columns:
                col_info = {
                    'name': col,
                    'dtype': str(df[col].dtype),
                    'null_count': int(df[col].isnull().sum()),
                    'null_percentage': round(df[col].isnull().sum() / len(df) * 100, 2) if len(df) > 0 else 0,
                    'unique_count': int(df[col].nunique()),
                }
                
                if pd.api.types.is_numeric_dtype(df[col]):
                    numeric_data = df[col].dropna()
                    if len(numeric_data) > 0:
                        col_info.update({
                            'min': float(numeric_data.min()),
                            'max': float(numeric_data.max()),
                            'mean': float(numeric_data.mean()),
                            'median': float(numeric_data.median()),
                            'std': float(numeric_data.std()),
                            'skewness': float(numeric_data.skew()),
                            'kurtosis': float(numeric_data.kurtosis()) if len(numeric_data) > 3 else 0,
                            'quartiles': {
                                'q1': float(numeric_data.quantile(0.25)),
                                'q3': float(numeric_data.quantile(0.75)),
                                'iqr': float(numeric_data.quantile(0.75) - numeric_data.quantile(0.25))
                            }
                        })
                else:
                    col_info['top_values'] = df[col].value_counts().head(5).to_dict()
                
                analysis['columns'].append(col_info)
            
            # Correlation matrix for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                corr_matrix = df[numeric_cols].corr()
                analysis['correlations'] = {
                    'matrix': corr_matrix.round(4).to_dict(),
                    'strongest_positive': self._get_strongest_correlations(corr_matrix, 'positive'),
                    'strongest_negative': self._get_strongest_correlations(corr_matrix, 'negative')
                }
            
            # Distribution analysis
            if len(numeric_cols) > 0:
                analysis['distributions'] = {}
                for col in numeric_cols[:5]:
                    data = df[col].dropna()
                    if len(data) > 3:
                        analysis['distributions'][col] = {
                            'histogram': self._calculate_histogram(data),
                            'normality_test': self._normality_test(data),
                            'outliers': self._detect_outliers(data)
                        }
            
            # Time analysis if date column exists
            date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
            if not date_cols:
                for col in df.columns:
                    try:
                        pd.to_datetime(df[col].iloc[:5])
                        date_cols.append(col)
                        break
                    except Exception:
                        continue
            
            if date_cols:
                analysis['time_analysis'] = self._analyze_time_series(df, date_cols[0], numeric_cols)
            
            # Data quality score
            analysis['data_quality'] = self._calculate_quality_score(df)
            
            return {'success': True, 'analysis': analysis}
        
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _get_strongest_correlations(self, corr_matrix: pd.DataFrame, direction: str) -> List[Dict]:
        """Get strongest correlations from matrix."""
        pairs = []
        cols = corr_matrix.columns
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                val = corr_matrix.iloc[i, j]
                if direction == 'positive' and val > 0.5:
                    pairs.append({'col1': cols[i], 'col2': cols[j], 'correlation': round(float(val), 4)})
                elif direction == 'negative' and val < -0.3:
                    pairs.append({'col1': cols[i], 'col2': cols[j], 'correlation': round(float(val), 4)})
        return sorted(pairs, key=lambda x: abs(x['correlation']), reverse=True)[:10]
    
    def _calculate_histogram(self, data: pd.Series, bins: int = 20) -> Dict:
        """Calculate histogram data."""
        hist, edges = np.histogram(data.dropna(), bins=bins)
        return {
            'bins': bins,
            'frequencies': hist.tolist(),
            'edges': [round(float(e), 4) for e in edges.tolist()]
        }
    
    def _normality_test(self, data: pd.Series) -> Dict:
        """Perform normality test."""
        try:
            clean_data = data.dropna()
            if len(clean_data) < 8:
                return {'test': 'insufficient_data', 'is_normal': None}
            
            stat, p_value = stats.shapiro(clean_data[:min(5000, len(clean_data))])
            return {
                'test': 'shapiro-wilk',
                'statistic': round(float(stat), 6),
                'p_value': round(float(p_value), 6),
                'is_normal': p_value > 0.05
            }
        except Exception:
            return {'test': 'failed', 'is_normal': None}
    
    def _detect_outliers(self, data: pd.Series, method: str = 'iqr') -> List[Dict]:
        """Detect outliers in data."""
        clean_data = data.dropna()
        if len(clean_data) < 10:
            return []
        
        outliers = []
        if method == 'iqr':
            q1 = clean_data.quantile(0.25)
            q3 = clean_data.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_mask = (clean_data < lower) | (clean_data > upper)
            outlier_values = clean_data[outlier_mask].tolist()
            outliers = [{'value': round(float(v), 4), 'index': int(i)} for i, v in zip(clean_data[outlier_mask].index[:20], outlier_values[:20])]
        
        return outliers
    
    def _analyze_time_series(self, df: pd.DataFrame, date_col: str, numeric_cols: List[str]) -> Dict:
        """Analyze time series patterns."""
        try:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.dropna(subset=[date_col]).sort_values(date_col)
            
            time_analysis = {
                'date_range': {
                    'start': df[date_col].min().strftime('%Y-%m-%d') if pd.notna(df[date_col].min()) else None,
                    'end': df[date_col].max().strftime('%Y-%m-%d') if pd.notna(df[date_col].max()) else None,
                    'duration_days': int((df[date_col].max() - df[date_col].min()).days) if pd.notna(df[date_col].max()) and pd.notna(df[date_col].min()) else 0
                },
                'metrics': {}
            }
            
            for col in numeric_cols[:3]:
                data = df[[date_col, col]].dropna()
                if len(data) < 5:
                    continue
                
                values = data[col].values
                
                # Trend
                x = np.arange(len(values))
                if len(x) > 1:
                    slope, intercept, r_value, _, _ = stats.linregress(x, values)
                    trend = 'increasing' if slope > 0.01 else 'decreasing' if slope < -0.01 else 'stable'
                else:
                    slope, r_value, trend = 0, 0, 'stable'
                
                # Volatility
                volatility = float(np.std(values))
                
                # Growth rate
                if len(values) > 1 and values[0] != 0:
                    total_growth = ((values[-1] - values[0]) / abs(values[0])) * 100
                else:
                    total_growth = 0
                
                # Moving averages
                ma_7 = pd.Series(values).rolling(7, min_periods=1).mean().tolist()
                ma_30 = pd.Series(values).rolling(30, min_periods=1).mean().tolist()
                
                time_analysis['metrics'][col] = {
                    'trend': trend,
                    'slope': round(float(slope), 6),
                    'r_squared': round(float(r_value ** 2), 4),
                    'volatility': round(volatility, 4),
                    'total_growth': round(float(total_growth), 2),
                    'moving_avg_7': [round(float(v), 4) for v in ma_7[-10:]],
                    'moving_avg_30': [round(float(v), 4) for v in ma_30[-10:]]
                }
            
            return time_analysis
        except Exception:
            return {}
    
    def _calculate_quality_score(self, df: pd.DataFrame) -> Dict:
        """Calculate data quality score."""
        total_cells = df.shape[0] * df.shape[1]
        null_cells = df.isnull().sum().sum()
        completeness = max(0, (1 - null_cells / total_cells) * 100) if total_cells > 0 else 0
        
        duplicate_ratio = (df.duplicated().sum() / len(df) * 100) if len(df) > 0 else 0
        
        return {
            'score': round(float(completeness), 2),
            'completeness': round(float(completeness), 2),
            'uniqueness': round(float(100 - duplicate_ratio), 2),
            'null_percentage': round(float(null_cells / total_cells * 100), 2) if total_cells > 0 else 0,
            'duplicate_percentage': round(float(duplicate_ratio), 2),
            'status': 'good' if completeness > 90 else 'fair' if completeness > 70 else 'poor'
        }
    
    def calculate_kpis(self, df: pd.DataFrame, value_col: str, date_col: str = None) -> Dict[str, Any]:
        """Calculate key performance indicators."""
        try:
            values = df[value_col].dropna()
            if len(values) == 0:
                return {'success': False, 'error': 'No valid data'}
            
            kpis = {
                'total': round(float(values.sum()), 2),
                'average': round(float(values.mean()), 2),
                'median': round(float(values.median()), 2),
                'minimum': round(float(values.min()), 2),
                'maximum': round(float(values.max()), 2),
                'std_dev': round(float(values.std()), 2),
                'growth_rate': 0.0,
                'trend': 'stable'
            }
            
            if date_col and date_col in df.columns:
                df_sorted = df.sort_values(date_col)
                vals = df_sorted[value_col].dropna()
                if len(vals) > 1:
                    # Calculate period-over-period growth
                    if len(vals) >= 2:
                        recent = vals.iloc[-min(len(vals)//4, 30):].mean()
                        previous = vals.iloc[:min(len(vals)//4, 30)].mean()
                        if previous != 0:
                            kpis['growth_rate'] = round(float(((recent - previous) / abs(previous)) * 100), 2)
                    
                    # Trend direction
                    x = np.arange(len(vals))
                    slope, _, _, _, _ = stats.linregress(x, vals.values)
                    kpis['trend'] = 'increasing' if slope > 0.01 else 'decreasing' if slope < -0.01 else 'stable'
            
            # Percentiles
            kpis['percentiles'] = {
                'p10': round(float(values.quantile(0.10)), 2),
                'p25': round(float(values.quantile(0.25)), 2),
                'p75': round(float(values.quantile(0.75)), 2),
                'p90': round(float(values.quantile(0.90)), 2)
            }
            
            return {'success': True, 'kpis': kpis}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def generate_correlation_heatmap(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate correlation heatmap data."""
        try:
            numeric_df = df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) < 2:
                return {'success': False, 'error': 'Need at least 2 numeric columns'}
            
            corr = numeric_df.corr().round(4)
            
            return {
                'success': True,
                'columns': corr.columns.tolist(),
                'matrix': corr.values.tolist(),
                'shape': corr.shape
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}


class ConsumerIntelligence:
    """Consumer sentiment and intelligence engine."""
    
    def __init__(self):
        self.vader = None
        if NLTK_AVAILABLE:
            try:
                self.vader = SentimentIntensityAnalyzer()
            except Exception:
                pass
    
    def analyze_sentiment(self, texts: List[str], source_type: str = 'reviews') -> Dict[str, Any]:
        """Analyze sentiment of consumer texts."""
        try:
            if not texts:
                return {'success': False, 'error': 'No texts provided'}
            
            results = {
                'positive': 0,
                'negative': 0,
                'neutral': 0,
                'compound_scores': [],
                'details': []
            }
            
            positive_texts = []
            negative_texts = []
            
            for text in texts[:min(len(texts), 5000)]:  # Limit for performance
                text = str(text)[:1000]  # Truncate long texts
                
                if self.vader:
                    scores = self.vader.polarity_scores(text)
                    compound = scores['compound']
                    pos = scores['pos']
                    neg = scores['neg']
                    neu = scores['neu']
                else:
                    blob = TextBlob(text)
                    polarity = blob.sentiment.polarity
                    compound = polarity
                    pos = max(0, polarity)
                    neg = max(0, -polarity)
                    neu = 1 - pos - neg
                
                results['compound_scores'].append(compound)
                
                if compound > 0.05:
                    results['positive'] += 1
                    if len(positive_texts) < 10:
                        positive_texts.append(text[:200])
                elif compound < -0.05:
                    results['negative'] += 1
                    if len(negative_texts) < 10:
                        negative_texts.append(text[:200])
                else:
                    results['neutral'] += 1
            
            total = len(texts[:min(len(texts), 5000)])
            
            # Calculate scores
            pos_pct = (results['positive'] / total * 100) if total > 0 else 0
            neg_pct = (results['negative'] / total * 100) if total > 0 else 0
            neu_pct = (results['neutral'] / total * 100) if total > 0 else 0
            
            compound_scores = results['compound_scores']
            avg_compound = np.mean(compound_scores) if compound_scores else 0
            
            # Brand health: weighted positive ratio
            brand_health = pos_pct * 0.7 + neu_pct * 0.2 + (100 - neg_pct) * 0.1
            
            # Emotion score: intensity of sentiment
            emotion_score = min(100, (np.std(compound_scores) * 200 + abs(avg_compound) * 50)) if compound_scores else 50
            
            # Trust score: consistency of positive sentiment
            trust_score = max(0, 100 - np.std(compound_scores) * 100) if compound_scores and len(compound_scores) > 1 else 50
            
            # Extract key topics/themes
            key_topics = self._extract_topics(texts)
            
            # Emotion breakdown
            emotion_breakdown = {
                'joy': max(0, min(100, pos_pct * 1.2)),
                'anger': max(0, min(100, neg_pct * 0.8)),
                'sadness': max(0, min(100, neg_pct * 0.5)),
                'fear': max(0, min(100, (100 - trust_score) * 0.6)),
                'trust': max(0, min(100, trust_score)),
                'anticipation': max(0, min(100, 50 + avg_compound * 50))
            }
            
            return {
                'success': True,
                'sentiment_distribution': {
                    'positive': round(float(pos_pct), 2),
                    'negative': round(float(neg_pct), 2),
                    'neutral': round(float(neu_pct), 2)
                },
                'positive_score': round(float(pos_pct), 2),
                'negative_score': round(float(neg_pct), 2),
                'neutral_score': round(float(neu_pct), 2),
                'brand_health_score': round(float(brand_health), 2),
                'emotion_score': round(float(emotion_score), 2),
                'trust_score': round(float(trust_score), 2),
                'avg_sentiment': round(float(avg_compound), 4),
                'sentiment_volatility': round(float(np.std(compound_scores)), 4) if compound_scores else 0,
                'key_topics': key_topics,
                'emotion_breakdown': emotion_breakdown,
                'sample_positive': positive_texts,
                'sample_negative': negative_texts,
                'total_analyzed': total
            }
        
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _extract_topics(self, texts: List[str], top_n: int = 10) -> List[Dict]:
        """Extract key topics from texts."""
        try:
            if not NLTK_AVAILABLE:
                return []
            
            try:
                stop_words = set(stopwords.words('english'))
            except:
                stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                             'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                             'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                             'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
                             'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
                             'through', 'during', 'before', 'after', 'above', 'below',
                             'between', 'out', 'off', 'over', 'under', 'again', 'further',
                             'then', 'once', 'here', 'there', 'when', 'where', 'why',
                             'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some',
                             'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
                             'than', 'too', 'very', 'just', 'and', 'but', 'if', 'or',
                             'because', 'until', 'while', 'this', 'that', 'these', 'those',
                             'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'him',
                             'his', 'she', 'her', 'it', 'its', 'they', 'them', 'their'}
            
            all_words = []
            for text in texts[:min(len(texts), 1000)]:
                words = re.findall(r'\b[a-zA-Z]{3,}\b', str(text).lower())
                all_words.extend([w for w in words if w not in stop_words])
            
            word_freq = Counter(all_words)
            top_words = word_freq.most_common(top_n)
            
            return [{'topic': word, 'frequency': freq, 'relevance': round(freq / len(texts) * 100, 2)} 
                    for word, freq in top_words]
        
        except Exception:
            return []
    
    def analyze_consumer_feedback(self, df: pd.DataFrame, text_col: str,
                                  rating_col: str = None) -> Dict[str, Any]:
        """Analyze consumer feedback data."""
        try:
            texts = df[text_col].dropna().astype(str).tolist()
            
            sentiment_result = self.analyze_sentiment(texts)
            
            if rating_col and rating_col in df.columns:
                ratings = df[rating_col].dropna()
                sentiment_result['rating_analysis'] = {
                    'avg_rating': round(float(ratings.mean()), 2),
                    'median_rating': round(float(ratings.median()), 2),
                    'rating_distribution': ratings.value_counts().sort_index().to_dict(),
                    'rating_std': round(float(ratings.std()), 2)
                }
            
            # Generate recommendations
            recommendations = self._generate_consumer_recommendations(sentiment_result)
            sentiment_result['recommendations'] = recommendations
            
            return sentiment_result
        
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _generate_consumer_recommendations(self, analysis: Dict) -> List[str]:
        """Generate recommendations based on consumer analysis."""
        recommendations = []
        
        neg_score = analysis.get('negative_score', 0)
        pos_score = analysis.get('positive_score', 0)
        trust_score = analysis.get('trust_score', 0)
        brand_health = analysis.get('brand_health_score', 0)
        
        if neg_score > 30:
            recommendations.append("Urgent: Address negative feedback trends. Implement immediate customer satisfaction initiatives.")
        
        if trust_score < 60:
            recommendations.append("Focus on building trust through transparency and consistent quality delivery.")
        
        if brand_health < 70:
            recommendations.append("Brand health needs attention. Consider reputation management and brand strengthening campaigns.")
        
        if pos_score > 70:
            recommendations.append("Leverage positive sentiment in marketing campaigns and customer testimonials.")
        else:
            recommendations.append("Develop strategies to convert neutral customers into brand advocates.")
        
        emotion = analysis.get('emotion_breakdown', {})
        if emotion.get('anger', 0) > 40:
            recommendations.append("High anger detected. Investigate root causes and implement conflict resolution processes.")
        
        if emotion.get('fear', 0) > 40:
            recommendations.append("Consumer fear detected. Improve communication and provide reassurance about product/service reliability.")
        
        recommendations.append("Monitor key topics regularly and respond proactively to emerging consumer concerns.")
        
        return recommendations


class OpportunityDetector:
    """Business opportunity detection engine."""
    
    def __init__(self):
        pass
    
    def detect_opportunities(self, df: pd.DataFrame, date_col: str, value_col: str,
                            category_col: str = None) -> Dict[str, Any]:
        """Detect business opportunities from data."""
        try:
            opportunities = []
            
            df = df.copy()
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.dropna(subset=[date_col, value_col])
            
            if len(df) < 10:
                return {'success': False, 'error': 'Insufficient data for opportunity detection'}
            
            if category_col and category_col in df.columns:
                # Analyze by category
                for category in df[category_col].unique():
                    cat_data = df[df[category_col] == category].sort_values(date_col)
                    if len(cat_data) >= 5:
                        opp = self._analyze_opportunity(cat_data, date_col, value_col, str(category), 'category')
                        if opp:
                            opportunities.append(opp)
            else:
                # Overall analysis
                df_sorted = df.sort_values(date_col)
                opp = self._analyze_opportunity(df_sorted, date_col, value_col, 'Overall Market', 'market')
                if opp:
                    opportunities.append(opp)
                
                # Detect growth pockets (periods of accelerated growth)
                growth_opps = self._detect_growth_pockets(df_sorted, date_col, value_col)
                opportunities.extend(growth_opps)
            
            # Sort by opportunity score
            opportunities.sort(key=lambda x: x.get('opportunity_score', 0), reverse=True)
            
            return {
                'success': True,
                'opportunities': opportunities[:20],
                'total_detected': len(opportunities),
                'summary': self._generate_opportunity_summary(opportunities)
            }
        
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _analyze_opportunity(self, df: pd.DataFrame, date_col: str, value_col: str,
                            name: str, opp_type: str) -> Optional[Dict]:
        """Analyze a single opportunity."""
        try:
            values = df[value_col].values.astype(float)
            if len(values) < 5 or np.all(values == 0):
                return None
            
            # Calculate metrics
            x = np.arange(len(values))
            slope, intercept, r_value, _, _ = stats.linregress(x, values)
            
            recent_values = values[-min(7, len(values)):]
            older_values = values[:min(7, len(values))]
            
            recent_avg = np.mean(recent_values)
            older_avg = np.mean(older_values)
            
            # Growth rate
            growth_rate = ((recent_avg - older_avg) / abs(older_avg) * 100) if older_avg != 0 else 0
            
            # Volatility
            volatility = float(np.std(values))
            
            # Trend strength
            trend_strength = min(100, abs(r_value) * 100)
            
            # Opportunity scoring
            opportunity_score = min(100, max(0, 
                (growth_rate * 2 if growth_rate > 0 else 0) + 
                trend_strength * 0.3 + 
                (50 if recent_avg > older_avg else 0)
            ))
            
            revenue_potential = recent_avg * len(values) * 0.1
            
            # Market readiness (based on consistency)
            consistency = max(0, 100 - (volatility / max(np.mean(values), 0.001)) * 100)
            market_readiness = min(100, consistency * 0.7 + trend_strength * 0.3)
            
            # Investment score
            investment_score = min(100, opportunity_score * 0.4 + market_readiness * 0.3 + trend_strength * 0.3)
            
            # Risk level
            if volatility / max(np.mean(values), 0.001) > 0.5:
                risk_level = 'high'
            elif volatility / max(np.mean(values), 0.001) > 0.2:
                risk_level = 'medium'
            else:
                risk_level = 'low'
            
            return {
                'opportunity_name': name,
                'opportunity_type': opp_type,
                'opportunity_score': round(float(opportunity_score), 2),
                'revenue_potential': round(float(revenue_potential), 2),
                'market_readiness_score': round(float(market_readiness), 2),
                'investment_score': round(float(investment_score), 2),
                'risk_level': risk_level,
                'growth_rate': round(float(growth_rate), 2),
                'trend_direction': 'increasing' if slope > 0.01 else 'decreasing' if slope < -0.01 else 'stable',
                'trend_strength': round(float(trend_strength), 2),
                'volatility': round(float(volatility), 4),
                'avg_value': round(float(np.mean(values)), 2),
                'recent_avg': round(float(recent_avg), 2),
                'data_points': len(values),
                'description': self._generate_opportunity_description(name, growth_rate, slope, risk_level),
                'recommendations': self._generate_opportunity_recommendations(growth_rate, risk_level, opportunity_score)
            }
        
        except Exception:
            return None
    
    def _detect_growth_pockets(self, df: pd.DataFrame, date_col: str, value_col: str) -> List[Dict]:
        """Detect periods of accelerated growth."""
        opportunities = []
        try:
            values = df[value_col].values.astype(float)
            
            if len(values) < 14:
                return opportunities
            
            # Calculate rolling growth rates
            window = 7
            growth_rates = []
            for i in range(window, len(values)):
                if values[i - window] != 0:
                    rate = ((values[i] - values[i - window]) / abs(values[i - window])) * 100
                    growth_rates.append((i, rate))
            
            # Find significant growth periods
            if growth_rates:
                rates = [r for _, r in growth_rates]
                mean_rate = np.mean(rates)
                std_rate = np.std(rates)
                
                for idx, rate in growth_rates:
                    if rate > mean_rate + 1.5 * std_rate and rate > 10:
                        opp = {
                            'opportunity_name': f"Growth Period (Day {idx})",
                            'opportunity_type': 'growth_pocket',
                            'opportunity_score': min(100, float(rate)),
                            'revenue_potential': round(float(values[idx] * 10), 2),
                            'market_readiness_score': 75.0,
                            'investment_score': min(100, float(rate) * 0.8),
                            'risk_level': 'medium',
                            'growth_rate': round(float(rate), 2),
                            'trend_direction': 'increasing',
                            'trend_strength': 85.0,
                            'volatility': round(float(np.std(values[max(0, idx-7):idx+1])), 4),
                            'avg_value': round(float(np.mean(values[max(0, idx-7):idx+1])), 2),
                            'recent_avg': round(float(values[idx]), 2),
                            'data_points': min(7, idx + 1),
                            'description': f"Detected accelerated growth of {rate:.1f}% at period {idx}.",
                            'recommendations': ["Investigate factors driving this growth spike.", "Consider scaling operations to capitalize on momentum."]
                        }
                        opportunities.append(opp)
            
            return opportunities
        
        except Exception:
            return opportunities
    
    def _generate_opportunity_description(self, name: str, growth_rate: float,
                                         slope: float, risk_level: str) -> str:
        """Generate human-readable opportunity description."""
        if growth_rate > 20:
            return f"{name} shows exceptional growth potential with a {growth_rate:.1f}% increase. This represents a significant market opportunity with {risk_level} risk."
        elif growth_rate > 10:
            return f"{name} demonstrates strong growth momentum at {growth_rate:.1f}%. Market entry is recommended with appropriate risk management."
        elif growth_rate > 0:
            return f"{name} indicates steady positive growth of {growth_rate:.1f}%. A promising opportunity with manageable risk profile."
        elif growth_rate > -10:
            return f"{name} shows slight decline ({growth_rate:.1f}%) but may present a contrarian opportunity at lower valuations."
        else:
            return f"{name} is experiencing significant decline ({growth_rate:.1f}%). Caution advised unless specific turnaround factors are identified."
    
    def _generate_opportunity_recommendations(self, growth_rate: float, risk_level: str,
                                               score: float) -> List[str]:
        """Generate recommendations for an opportunity."""
        recs = []
        
        if score > 80:
            recs.append("High-priority opportunity: Allocate resources for immediate market entry.")
        elif score > 60:
            recs.append("Promising opportunity: Conduct detailed feasibility study and prepare entry strategy.")
        elif score > 40:
            recs.append("Moderate opportunity: Monitor market conditions and prepare contingency plans.")
        else:
            recs.append("Lower priority: Continue monitoring for improvement in market conditions.")
        
        if growth_rate > 15:
            recs.append("Strong growth trend: Consider scaling operations to capture market share.")
        
        if risk_level == 'high':
            recs.append("High risk detected: Implement robust risk mitigation strategies before investment.")
        elif risk_level == 'low':
            recs.append("Favorable risk profile: Opportunity suitable for conservative investment approaches.")
        
        recs.append("Establish KPIs to monitor performance post-entry and adjust strategy as needed.")
        
        return recs
    
    def _generate_opportunity_summary(self, opportunities: List[Dict]) -> Dict:
        """Generate summary of all opportunities."""
        if not opportunities:
            return {'status': 'no_opportunities', 'message': 'No significant opportunities detected'}
        
        scores = [o.get('opportunity_score', 0) for o in opportunities]
        
        return {
            'status': 'opportunities_found',
            'total_count': len(opportunities),
            'avg_opportunity_score': round(float(np.mean(scores)), 2),
            'max_opportunity_score': round(float(max(scores)), 2),
            'high_opportunities': len([s for s in scores if s > 70]),
            'medium_opportunities': len([s for s in scores if 40 <= s <= 70]),
            'low_opportunities': len([s for s in scores if s < 40]),
            'top_opportunity': opportunities[0]['opportunity_name'] if opportunities else None
        }


class ScenarioSimulator:
    """Business scenario simulation engine."""
    
    def __init__(self):
        self.scenarios = {
            'price_increase': {'description': 'Price Increase', 'default_value': 10},
            'price_decrease': {'description': 'Price Decrease', 'default_value': 10},
            'marketing_increase': {'description': 'Marketing Spend Increase', 'default_value': 20},
            'marketing_decrease': {'description': 'Marketing Spend Decrease', 'default_value': 20},
            'demand_increase': {'description': 'Demand Surge', 'default_value': 15},
            'demand_decrease': {'description': 'Demand Decline', 'default_value': 15},
            'competitor_price_cut': {'description': 'Competitor Price Cut', 'default_value': 10},
            'new_product_launch': {'description': 'New Product Launch', 'default_value': 25}
        }
    
    def simulate(self, df: pd.DataFrame, value_col: str, date_col: str,
                 scenario_type: str, impact_percentage: float,
                 elasticity_assumptions: Dict = None) -> Dict[str, Any]:
        """Run a business scenario simulation."""
        try:
            df = df.copy()
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.dropna(subset=[date_col, value_col])
            
            if len(df) < 5:
                return {'success': False, 'error': 'Insufficient data for simulation'}
            
            values = df[value_col].values.astype(float)
            dates = df[date_col].tolist()
            
            # Calculate baseline metrics
            baseline_total = float(np.sum(values))
            baseline_avg = float(np.mean(values))
            baseline_trend = self._calculate_trend(values)
            
            # Apply scenario impact
            simulated_values = self._apply_scenario(values, scenario_type, impact_percentage, elasticity_assumptions)
            
            # Calculate simulated metrics
            simulated_total = float(np.sum(simulated_values))
            simulated_avg = float(np.mean(simulated_values))
            simulated_trend = self._calculate_trend(simulated_values)
            
            # Impact analysis
            revenue_impact = simulated_total - baseline_total
            revenue_impact_pct = (revenue_impact / abs(baseline_total) * 100) if baseline_total != 0 else 0
            
            impact_analysis = {
                'revenue_impact': round(float(revenue_impact), 2),
                'revenue_impact_percentage': round(float(revenue_impact_pct), 2),
                'avg_change': round(float(simulated_avg - baseline_avg), 2),
                'trend_change': round(float(simulated_trend - baseline_trend), 4),
                'direction': 'positive' if revenue_impact > 0 else 'negative' if revenue_impact < 0 else 'neutral',
                'severity': 'high' if abs(revenue_impact_pct) > 20 else 'medium' if abs(revenue_impact_pct) > 10 else 'low'
            }
            
            # Comparison data for charts
            comparison = {
                'dates': [d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d) for d in dates],
                'baseline': [round(float(v), 4) for v in values],
                'simulated': [round(float(v), 4) for v in simulated_values],
                'difference': [round(float(s - b), 4) for s, b in zip(simulated_values, values)]
            }
            
            # Recommendations
            recommendations = self._generate_simulation_recommendations(scenario_type, impact_analysis)
            
            return {
                'success': True,
                'scenario_type': scenario_type,
                'scenario_name': self.scenarios.get(scenario_type, {}).get('description', scenario_type),
                'impact_percentage': impact_percentage,
                'baseline_metrics': {
                    'total': round(baseline_total, 2),
                    'average': round(baseline_avg, 2),
                    'trend': round(baseline_trend, 4)
                },
                'simulated_metrics': {
                    'total': round(simulated_total, 2),
                    'average': round(simulated_avg, 2),
                    'trend': round(simulated_trend, 4)
                },
                'impact_analysis': impact_analysis,
                'comparison_data': comparison,
                'recommendations': recommendations
            }
        
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _apply_scenario(self, values: np.ndarray, scenario_type: str,
                       impact_percentage: float, elasticity: Dict = None) -> np.ndarray:
        """Apply scenario impact to values."""
        simulated = values.copy().astype(float)
        impact = impact_percentage / 100
        
        if scenario_type == 'price_increase':
            # Revenue = Price * Volume; assume volume drops with price increase
            price_elasticity = elasticity.get('price_elasticity', -1.5) if elasticity else -1.5
            volume_change = price_elasticity * impact
            simulated = simulated * (1 + impact) * (1 + volume_change)
        
        elif scenario_type == 'price_decrease':
            price_elasticity = elasticity.get('price_elasticity', -1.5) if elasticity else -1.5
            volume_change = -price_elasticity * impact
            simulated = simulated * (1 - impact) * (1 + volume_change)
        
        elif scenario_type == 'marketing_increase':
            # Marketing ROI assumption: 5:1 return
            roi = elasticity.get('marketing_roi', 5.0) if elasticity else 5.0
            simulated = simulated * (1 + impact * roi)
        
        elif scenario_type == 'marketing_decrease':
            roi = elasticity.get('marketing_roi', 5.0) if elasticity else 5.0
            simulated = simulated * (1 - impact * roi * 0.5)
        
        elif scenario_type == 'demand_increase':
            simulated = simulated * (1 + impact)
        
        elif scenario_type == 'demand_decrease':
            simulated = simulated * (1 - impact)
        
        elif scenario_type == 'competitor_price_cut':
            # Assume market share loss
            share_loss = impact * 0.5
            simulated = simulated * (1 - share_loss)
        
        elif scenario_type == 'new_product_launch':
            # Initial investment followed by growth
            cannibalization = elasticity.get('cannibalization', 0.2) if elasticity else 0.2
            new_product_lift = impact
            simulated = simulated * (1 - cannibalization) * (1 + new_product_lift)
        
        else:
            simulated = simulated * (1 + impact)
        
        return np.maximum(simulated, 0)  # Ensure non-negative
    
    def _calculate_trend(self, values: np.ndarray) -> float:
        """Calculate trend slope."""
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values))
        slope, _, _, _, _ = stats.linregress(x, values)
        return float(slope)
    
    def _generate_simulation_recommendations(self, scenario_type: str,
                                              impact: Dict) -> List[str]:
        """Generate recommendations based on simulation results."""
        recommendations = []
        direction = impact.get('direction', 'neutral')
        severity = impact.get('severity', 'low')
        
        if direction == 'positive':
            recommendations.append("The scenario projects positive outcomes. Consider proceeding with careful monitoring.")
        elif direction == 'negative':
            recommendations.append("The scenario projects negative impact. Review and adjust strategy before implementation.")
        else:
            recommendations.append("The scenario shows neutral impact. Additional factors may need consideration.")
        
        if severity == 'high':
            recommendations.append("High impact detected: Conduct thorough risk assessment and develop contingency plans.")
        
        if scenario_type in ['price_increase', 'price_decrease']:
            recommendations.append("Monitor competitor pricing responses and customer price sensitivity.")
        elif scenario_type in ['marketing_increase', 'marketing_decrease']:
            recommendations.append("Track marketing ROI metrics and adjust spend allocation based on channel performance.")
        elif scenario_type == 'new_product_launch':
            recommendations.append("Ensure adequate resource allocation for product development, marketing, and support.")
        
        recommendations.append("Run multiple scenarios with varying assumptions to understand sensitivity ranges.")
        recommendations.append("Establish clear success metrics and review points post-implementation.")
        
        return recommendations
    
    def get_available_scenarios(self) -> Dict[str, Dict]:
        """Get list of available scenarios."""
        return self.scenarios


class ExecutiveInsights:
    """Generate executive-level insights and recommendations."""
    
    def __init__(self):
        pass
    
    def generate_executive_summary(self, forecast_result: Dict = None,
                                   competitor_analysis: Dict = None,
                                   consumer_insights: Dict = None,
                                   opportunities: Dict = None,
                                   dataset_stats: Dict = None) -> Dict[str, Any]:
        """Generate comprehensive executive summary."""
        try:
            summary = {
                'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'sections': {},
                'overall_health': {},
                'recommendations': []
            }
            
            # Forecast summary
            if forecast_result and forecast_result.get('success'):
                summary['sections']['forecast'] = {
                    'status': 'available',
                    'model_used': forecast_result.get('model_name', 'N/A'),
                    'growth_projection': forecast_result.get('summary', {}).get('growth_projection', 0),
                    'prediction_trend': forecast_result.get('summary', {}).get('prediction_trend', 'unknown'),
                    'confidence_metrics': forecast_result.get('metrics', {})
                }
            
            # Competitor summary
            if competitor_analysis and competitor_analysis.get('success'):
                summary['sections']['competitive'] = {
                    'status': 'available',
                    'overall_score': competitor_analysis.get('overall_score', 0),
                    'threat_level': competitor_analysis.get('threat_score', 0),
                    'key_insights': competitor_analysis.get('insights', '')
                }
            
            # Consumer summary
            if consumer_insights and consumer_insights.get('success'):
                summary['sections']['consumer'] = {
                    'status': 'available',
                    'sentiment_positive': consumer_insights.get('positive_score', 0),
                    'brand_health': consumer_insights.get('brand_health_score', 0),
                    'trust_score': consumer_insights.get('trust_score', 0)
                }
            
            # Opportunities summary
            if opportunities and opportunities.get('success'):
                summary['sections']['opportunities'] = {
                    'status': 'available',
                    'count': opportunities.get('total_detected', 0),
                    'top_score': opportunities.get('summary', {}).get('max_opportunity_score', 0),
                    'high_priority': opportunities.get('summary', {}).get('high_opportunities', 0)
                }
            
            # Calculate overall health scores
            summary['overall_health'] = self._calculate_overall_health(summary['sections'])
            
            # Generate strategic recommendations
            summary['recommendations'] = self._generate_strategic_recommendations(summary['sections'])
            
            return {'success': True, 'summary': summary}
        
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _calculate_overall_health(self, sections: Dict) -> Dict:
        """Calculate overall health scores."""
        health = {
            'business_health': 50.0,
            'market_health': 50.0,
            'demand_score': 50.0,
            'growth_score': 50.0,
            'opportunity_score': 50.0,
            'risk_score': 50.0
        }
        
        if 'forecast' in sections:
            growth = sections['forecast'].get('growth_projection', 0)
            health['growth_score'] = min(100, max(0, 50 + growth))
            health['demand_score'] = min(100, max(0, 50 + growth * 0.5))
            trend = sections['forecast'].get('prediction_trend', 'stable')
            if trend == 'increasing':
                health['business_health'] += 10
                health['market_health'] += 10
            elif trend == 'decreasing':
                health['business_health'] -= 10
                health['market_health'] -= 10
        
        if 'consumer' in sections:
            brand = sections['consumer'].get('brand_health', 50)
            trust = sections['consumer'].get('trust_score', 50)
            pos = sections['consumer'].get('sentiment_positive', 50)
            health['business_health'] = (health['business_health'] + brand * 0.3 + trust * 0.2) / 1.5
            health['market_health'] = (health['market_health'] + pos * 0.3) / 1.3
        
        if 'competitive' in sections:
            threat = sections['competitive'].get('threat_level', 50)
            health['risk_score'] = min(100, threat)
            health['business_health'] -= (threat - 50) * 0.2
        
        if 'opportunities' in sections:
            opp_count = sections['opportunities'].get('count', 0)
            top_score = sections['opportunities'].get('top_score', 0)
            health['opportunity_score'] = min(100, 30 + opp_count * 5 + top_score * 0.3)
            if opp_count > 0:
                health['business_health'] += 5
        
        # Normalize
        for key in health:
            health[key] = round(float(max(0, min(100, health[key]))), 2)
        
        return health
    
    def _generate_strategic_recommendations(self, sections: Dict) -> List[str]:
        """Generate strategic recommendations."""
        recommendations = []
        
        if 'forecast' in sections:
            trend = sections['forecast'].get('prediction_trend', 'stable')
            growth = sections['forecast'].get('growth_projection', 0)
            if trend == 'increasing' and growth > 10:
                recommendations.append("Capitalize on strong growth trajectory by increasing production capacity and market investment.")
            elif trend == 'decreasing':
                recommendations.append("Implement cost optimization and diversify revenue streams to address declining trends.")
        
        if 'consumer' in sections:
            brand = sections['consumer'].get('brand_health', 50)
            if brand < 60:
                recommendations.append("Launch brand health improvement initiative focusing on customer experience and communication.")
            trust = sections['consumer'].get('trust_score', 50)
            if trust < 60:
                recommendations.append("Develop transparency programs and quality assurance initiatives to rebuild consumer trust.")
        
        if 'competitive' in sections:
            threat = sections['competitive'].get('threat_level', 50)
            if threat > 70:
                recommendations.append("High competitive threat detected: Strengthen competitive positioning through differentiation and innovation.")
        
        if 'opportunities' in sections:
            opp_count = sections['opportunities'].get('count', 0)
            if opp_count > 3:
                recommendations.append(f"Multiple opportunities ({opp_count}) identified: Prioritize based on strategic fit and resource requirements.")
        
        if not recommendations:
            recommendations.append("Continue monitoring market conditions and maintain current strategic direction.")
            recommendations.append("Invest in data collection and analysis capabilities to enhance future decision-making.")
        
        recommendations.append("Schedule quarterly review of all metrics and adjust strategy based on performance data.")
        
        return recommendations