"""
MarketMind - Competitive Intelligence Engine
Provides competitor analysis, market positioning, threat assessment,
and strategic intelligence analytics.
"""

import json
import warnings
import traceback
from collections import Counter
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')


class CompetitorIntelligence:
    """Enterprise competitive intelligence engine."""
    
    def __init__(self):
        self.analysis_cache = {}
    
    def analyze_competitor(self, df: pd.DataFrame, competitor_col: str,
                          value_col: str, date_col: str = None,
                          price_col: str = None, review_col: str = None,
                          feature_cols: List[str] = None) -> Dict[str, Any]:
        """Comprehensive competitor analysis."""
        try:
            df = df.copy()
            competitors = df[competitor_col].unique()
            
            analyses = []
            market_totals = {}
            
            # Calculate market totals for share calculation
            if value_col in df.columns:
                total_market = df[value_col].sum()
            else:
                total_market = len(df)
            
            for competitor in competitors:
                comp_data = df[df[competitor_col] == competitor]
                analysis = self._analyze_single_competitor(
                    comp_data, competitor, value_col, date_col, 
                    price_col, review_col, feature_cols, total_market
                )
                analyses.append(analysis)
            
            # Calculate relative scores
            analyses = self._calculate_relative_scores(analyses)
            
            # Generate competitive landscape insights
            insights = self._generate_landscape_insights(analyses)
            
            return {
                'success': True,
                'competitor_count': len(analyses),
                'analyses': analyses,
                'market_concentration': self._calculate_hhi(analyses),
                'insights': insights,
                'recommendations': self._generate_competitive_recommendations(analyses)
            }
        
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _analyze_single_competitor(self, df: pd.DataFrame, name: str,
                                    value_col: str, date_col: str = None,
                                    price_col: str = None, review_col: str = None,
                                    feature_cols: List[str] = None,
                                    total_market: float = 1) -> Dict:
        """Analyze a single competitor."""
        analysis = {
            'competitor_name': str(name),
            'data_points': len(df)
        }
        
        # Market share
        if value_col in df.columns:
            comp_total = df[value_col].sum()
            analysis['market_share'] = round(float(comp_total / total_market * 100), 2) if total_market > 0 else 0
            analysis['total_value'] = round(float(comp_total), 2)
            
            # Growth trend
            if date_col and date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                df_sorted = df.dropna(subset=[date_col]).sort_values(date_col)
                if len(df_sorted) >= 5:
                    values = df_sorted[value_col].values.astype(float)
                    x = np.arange(len(values))
                    slope, _, r_value, _, _ = stats.linregress(x, values)
                    growth_rate = ((values[-1] - values[0]) / abs(values[0]) * 100) if values[0] != 0 and len(values) > 1 else 0
                    
                    analysis['growth_rate'] = round(float(growth_rate), 2)
                    analysis['trend_direction'] = 'increasing' if slope > 0.01 else 'decreasing' if slope < -0.01 else 'stable'
                    analysis['trend_strength'] = round(float(min(100, abs(r_value) * 100)), 2)
                    analysis['recent_performance'] = round(float(np.mean(values[-min(5, len(values)):])), 2)
                else:
                    analysis['growth_rate'] = 0
                    analysis['trend_direction'] = 'stable'
                    analysis['trend_strength'] = 0
            else:
                analysis['growth_rate'] = 0
                analysis['trend_direction'] = 'stable'
                analysis['trend_strength'] = 0
        
        # Pricing analysis
        if price_col and price_col in df.columns:
            prices = df[price_col].dropna().astype(float)
            if len(prices) > 0:
                analysis['avg_price'] = round(float(prices.mean()), 2)
                analysis['price_range'] = {
                    'min': round(float(prices.min()), 2),
                    'max': round(float(prices.max()), 2)
                }
                analysis['price_volatility'] = round(float(prices.std()), 2)
        
        # Review/sentiment analysis
        if review_col and review_col in df.columns:
            reviews = df[review_col].dropna().astype(str).tolist()
            if reviews:
                from textblob import TextBlob
                sentiments = []
                for review in reviews[:min(len(reviews), 1000)]:
                    blob = TextBlob(str(review))
                    sentiments.append(blob.sentiment.polarity)
                
                avg_sentiment = np.mean(sentiments) if sentiments else 0
                analysis['avg_sentiment'] = round(float(avg_sentiment), 4)
                analysis['sentiment_score'] = round(float((avg_sentiment + 1) / 2 * 100), 2)
                analysis['review_count'] = len(reviews)
        
        # Feature analysis
        if feature_cols:
            features = {}
            for col in feature_cols:
                if col in df.columns:
                    features[col] = {
                        'avg': round(float(df[col].mean()), 2) if pd.api.types.is_numeric_dtype(df[col]) else None,
                        'has_feature': df[col].notna().sum() > 0
                    }
            analysis['features'] = features
        
        return analysis
    
    def _calculate_relative_scores(self, analyses: List[Dict]) -> List[Dict]:
        """Calculate relative competitive scores."""
        if not analyses or len(analyses) < 2:
            for a in analyses:
                a['overall_score'] = 50.0
                a['threat_score'] = 50.0
                a['growth_score'] = 50.0
                a['market_position_score'] = 50.0
                a['innovation_score'] = 50.0
                a['pricing_score'] = 50.0
            return analyses
        
        # Calculate percentiles for each metric
        metrics = {
            'market_share': [a.get('market_share', 0) for a in analyses],
            'growth_rate': [a.get('growth_rate', 0) for a in analyses],
            'trend_strength': [a.get('trend_strength', 0) for a in analyses],
            'sentiment_score': [a.get('sentiment_score', 50) for a in analyses]
        }
        
        for analysis in analyses:
            # Market position score (based on market share)
            ms = analysis.get('market_share', 0)
            max_ms = max(metrics['market_share']) if metrics['market_share'] else 1
            analysis['market_position_score'] = round(float(min(100, (ms / max(max_ms, 0.001)) * 100)), 2)
            
            # Growth score
            gr = analysis.get('growth_rate', 0)
            max_gr = max(max(metrics['growth_rate']), 1)
            analysis['growth_score'] = round(float(min(100, max(0, 50 + (gr / max_gr) * 50))), 2)
            
            # Innovation score (based on trend strength and sentiment)
            ts = analysis.get('trend_strength', 0)
            ss = analysis.get('sentiment_score', 50)
            analysis['innovation_score'] = round(float(min(100, ts * 0.5 + ss * 0.5)), 2)
            
            # Pricing score (competitive pricing = middle range)
            if 'avg_price' in analysis:
                all_prices = [a.get('avg_price', 0) for a in analyses if 'avg_price' in a]
                if all_prices:
                    avg_market_price = np.mean(all_prices)
                    price_ratio = analysis['avg_price'] / avg_market_price if avg_market_price > 0 else 1
                    # Closer to market average is better
                    analysis['pricing_score'] = round(float(max(0, 100 - abs(price_ratio - 1) * 50)), 2)
                else:
                    analysis['pricing_score'] = 50.0
            else:
                analysis['pricing_score'] = 50.0
            
            # Threat score (combination of market share, growth, and trend)
            ms_score = analysis['market_position_score']
            gr_score = analysis['growth_score']
            ts_score = analysis['trend_strength']
            analysis['threat_score'] = round(float(min(100, ms_score * 0.4 + gr_score * 0.3 + ts_score * 0.3)), 2)
            
            # Overall score
            analysis['overall_score'] = round(float(
                analysis['market_position_score'] * 0.25 +
                analysis['growth_score'] * 0.20 +
                analysis['innovation_score'] * 0.20 +
                analysis['pricing_score'] * 0.15 +
                analysis.get('sentiment_score', 50) * 0.10 +
                analysis['threat_score'] * 0.10
            ), 2)
            
            # SWOT
            analysis['strengths'] = self._identify_strengths(analysis)
            analysis['weaknesses'] = self._identify_weaknesses(analysis)
            analysis['opportunities'] = self._identify_opportunities(analysis, metrics)
            analysis['threats'] = self._identify_threats(analysis, metrics)
        
        return analyses
    
    def _identify_strengths(self, analysis: Dict) -> List[str]:
        """Identify competitor strengths."""
        strengths = []
        if analysis.get('market_position_score', 0) > 70:
            strengths.append(f"Strong market position with {analysis.get('market_share', 0):.1f}% market share")
        if analysis.get('growth_score', 0) > 70:
            strengths.append(f"High growth trajectory at {analysis.get('growth_rate', 0):.1f}% growth rate")
        if analysis.get('innovation_score', 0) > 70:
            strengths.append("Strong innovation indicators and positive market momentum")
        if analysis.get('sentiment_score', 50) > 75:
            strengths.append("Positive consumer sentiment and brand perception")
        if analysis.get('pricing_score', 50) > 70:
            strengths.append("Competitive pricing strategy")
        if not strengths:
            strengths.append("Established market presence")
        return strengths
    
    def _identify_weaknesses(self, analysis: Dict) -> List[str]:
        """Identify competitor weaknesses."""
        weaknesses = []
        if analysis.get('market_position_score', 0) < 30:
            weaknesses.append("Limited market share and visibility")
        if analysis.get('growth_score', 0) < 30:
            weaknesses.append(f"Declining growth at {analysis.get('growth_rate', 0):.1f}%")
        if analysis.get('sentiment_score', 50) < 40:
            weaknesses.append("Negative consumer sentiment")
        if analysis.get('data_points', 0) < 10:
            weaknesses.append("Limited data for comprehensive analysis")
        if not weaknesses:
            weaknesses.append("No significant weaknesses identified")
        return weaknesses
    
    def _identify_opportunities(self, analysis: Dict, all_metrics: Dict) -> List[str]:
        """Identify opportunities for competitor."""
        opportunities = []
        if analysis.get('growth_rate', 0) > 15:
            opportunities.append("Capitalize on strong growth momentum for market expansion")
        avg_ms = np.mean(all_metrics.get('market_share', [0])) if all_metrics.get('market_share') else 0
        if analysis.get('market_share', 0) < avg_ms:
            opportunities.append("Potential to gain market share from established players")
        if analysis.get('innovation_score', 0) > 60:
            opportunities.append("Leverage innovation capabilities for new market segments")
        opportunities.append("Explore strategic partnerships and alliances")
        return opportunities
    
    def _identify_threats(self, analysis: Dict, all_metrics: Dict) -> List[str]:
        """Identify threats to competitor."""
        threats = []
        max_ms = max(all_metrics.get('market_share', [0])) if all_metrics.get('market_share') else 0
        if analysis.get('market_share', 0) < max_ms * 0.5 and max_ms > 20:
            threats.append("Dominant competitor poses significant market threat")
        if analysis.get('growth_rate', 0) < 0:
            threats.append("Negative growth trend threatens market position")
        threats.append("Market volatility and changing consumer preferences")
        threats.append("Potential new entrants and disruptive technologies")
        return threats
    
    def _calculate_hhi(self, analyses: List[Dict]) -> Dict:
        """Calculate Herfindahl-Hirschman Index for market concentration."""
        shares = [a.get('market_share', 0) for a in analyses]
        hhi = sum(s ** 2 for s in shares)
        
        if hhi > 2500:
            concentration = 'high'
        elif hhi > 1500:
            concentration = 'moderate'
        else:
            concentration = 'low'
        
        return {
            'hhi': round(float(hhi), 2),
            'concentration_level': concentration,
            'competitor_count': len(analyses),
            'total_market_share': round(float(sum(shares)), 2)
        }
    
    def _generate_landscape_insights(self, analyses: List[Dict]) -> List[str]:
        """Generate competitive landscape insights."""
        insights = []
        
        if not analyses:
            return insights
        
        # Market leader
        leader = max(analyses, key=lambda x: x.get('market_position_score', 0))
        insights.append(f"{leader['competitor_name']} holds the strongest market position with a score of {leader['market_position_score']:.1f}.")
        
        # Fastest grower
        fastest = max(analyses, key=lambda x: x.get('growth_score', 0))
        if fastest.get('growth_rate', 0) > 10:
            insights.append(f"{fastest['competitor_name']} shows the strongest growth momentum at {fastest['growth_rate']:.1f}%.")
        
        # Most threatening
        most_threat = max(analyses, key=lambda x: x.get('threat_score', 0))
        insights.append(f"{most_threat['competitor_name']} presents the highest competitive threat with a score of {most_threat['threat_score']:.1f}.")
        
        # Market dynamics
        growing = sum(1 for a in analyses if a.get('growth_rate', 0) > 5)
        declining = sum(1 for a in analyses if a.get('growth_rate', 0) < -5)
        insights.append(f"Market dynamics: {growing} competitors growing, {declining} declining out of {len(analyses)} total.")
        
        # Sentiment leader
        sentiment_scores = [(a['competitor_name'], a.get('sentiment_score', 50)) for a in analyses if 'sentiment_score' in a]
        if sentiment_scores:
            sentiment_leader = max(sentiment_scores, key=lambda x: x[1])
            insights.append(f"{sentiment_leader[0]} leads in consumer sentiment with a score of {sentiment_leader[1]:.1f}.")
        
        return insights
    
    def _generate_competitive_recommendations(self, analyses: List[Dict]) -> List[str]:
        """Generate strategic recommendations."""
        recommendations = []
        
        hhi = self._calculate_hhi(analyses)
        if hhi['concentration_level'] == 'high':
            recommendations.append("High market concentration detected: Focus on differentiation and niche strategies.")
        elif hhi['concentration_level'] == 'low':
            recommendations.append("Fragmented market presents consolidation and acquisition opportunities.")
        
        # Check for high-threat competitors
        high_threats = [a for a in analyses if a.get('threat_score', 0) > 70]
        if high_threats:
            recommendations.append(f"Monitor {', '.join(t['competitor_name'] for t in high_threats[:3])} closely due to high threat levels.")
        
        recommendations.append("Develop competitive monitoring system to track market changes.")
        recommendations.append("Invest in R&D and innovation to maintain competitive edge.")
        recommendations.append("Strengthen customer relationships to improve retention and sentiment.")
        
        return recommendations
    
    def compare_competitors(self, analyses: List[Dict], 
                           metrics: List[str] = None) -> Dict[str, Any]:
        """Generate detailed competitor comparison."""
        if not analyses:
            return {'success': False, 'error': 'No competitor data available'}
        
        if metrics is None:
            metrics = ['overall_score', 'market_position_score', 'growth_score', 
                      'innovation_score', 'pricing_score', 'threat_score']
        
        comparison = {
            'competitors': [a['competitor_name'] for a in analyses],
            'metrics': {}
        }
        
        for metric in metrics:
            comparison['metrics'][metric] = {
                a['competitor_name']: a.get(metric, 0) for a in analyses
            }
        
        # Rankings
        comparison['rankings'] = {}
        for metric in metrics:
            sorted_competitors = sorted(analyses, key=lambda x: x.get(metric, 0), reverse=True)
            comparison['rankings'][metric] = [
                {'name': a['competitor_name'], 'score': a.get(metric, 0)} 
                for a in sorted_competitors
            ]
        
        # Winner for each metric
        comparison['winners'] = {}
        for metric in metrics:
            winner = max(analyses, key=lambda x: x.get(metric, 0))
            comparison['winners'][metric] = {
                'name': winner['competitor_name'],
                'score': winner.get(metric, 0)
            }
        
        return {'success': True, 'comparison': comparison}
    
    def generate_radar_chart_data(self, analysis: Dict) -> Dict[str, Any]:
        """Generate radar chart data for a competitor."""
        categories = ['Market Position', 'Growth', 'Innovation', 'Pricing', 'Sentiment', 'Threat']
        values = [
            analysis.get('market_position_score', 0),
            analysis.get('growth_score', 0),
            analysis.get('innovation_score', 0),
            analysis.get('pricing_score', 0),
            analysis.get('sentiment_score', 50),
            analysis.get('threat_score', 0)
        ]
        
        return {
            'categories': categories,
            'values': values,
            'competitor': analysis.get('competitor_name', 'Unknown')
        }


class MarketIntelligence:
    """Market trend and intelligence analysis."""
    
    def __init__(self):
        pass
    
    def analyze_market_trends(self, df: pd.DataFrame, date_col: str,
                              value_col: str, category_col: str = None) -> Dict[str, Any]:
        """Analyze overall market trends."""
        try:
            df = df.copy()
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.dropna(subset=[date_col, value_col])
            
            if len(df) < 5:
                return {'success': False, 'error': 'Insufficient data'}
            
            df = df.sort_values(date_col)
            values = df[value_col].values.astype(float)
            
            # Overall trend
            x = np.arange(len(values))
            slope, intercept, r_value, _, _ = stats.linregress(x, values)
            
            trend = 'increasing' if slope > 0.01 else 'decreasing' if slope < -0.01 else 'stable'
            trend_strength = min(100, abs(r_value) * 100)
            
            # Growth metrics
            if len(values) > 1 and values[0] != 0:
                total_growth = ((values[-1] - values[0]) / abs(values[0])) * 100
            else:
                total_growth = 0
            
            # Volatility
            volatility = float(np.std(values))
            
            # Seasonality
            if len(values) >= 14:
                seasonality = self._detect_seasonality(values)
            else:
                seasonality = {'has_seasonality': False, 'strength': 0}
            
            # Moving averages
            ma_7 = pd.Series(values).rolling(7, min_periods=1).mean().tolist()
            ma_30 = pd.Series(values).rolling(30, min_periods=1).mean().tolist()
            
            # Category breakdown
            categories = {}
            if category_col and category_col in df.columns:
                for cat in df[category_col].unique():
                    cat_data = df[df[category_col] == cat][value_col].values.astype(float)
                    if len(cat_data) > 0:
                        categories[str(cat)] = {
                            'total': round(float(np.sum(cat_data)), 2),
                            'avg': round(float(np.mean(cat_data)), 2),
                            'share': round(float(np.sum(cat_data) / np.sum(values) * 100), 2) if np.sum(values) > 0 else 0
                        }
            
            return {
                'success': True,
                'trend': trend,
                'trend_slope': round(float(slope), 6),
                'trend_strength': round(float(trend_strength), 2),
                'total_growth': round(float(total_growth), 2),
                'volatility': round(float(volatility), 4),
                'avg_value': round(float(np.mean(values)), 2),
                'peak_value': round(float(np.max(values)), 2),
                'min_value': round(float(np.min(values)), 2),
                'seasonality': seasonality,
                'moving_averages': {
                    'ma7': [round(float(v), 4) for v in ma_7[-20:]],
                    'ma30': [round(float(v), 4) for v in ma_30[-20:]]
                },
                'categories': categories,
                'data_points': len(values)
            }
        
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _detect_seasonality(self, values: np.ndarray, period: int = 7) -> Dict:
        """Detect seasonality in time series."""
        try:
            if len(values) < period * 2:
                return {'has_seasonality': False, 'strength': 0}
            
            # Decompose using moving average
            trend = pd.Series(values).rolling(period, center=True, min_periods=1).mean().values
            detrended = values - trend
            
            # Calculate seasonality strength
            seasonal_var = np.var(detrended)
            total_var = np.var(values)
            
            if total_var == 0:
                strength = 0
            else:
                strength = max(0, 1 - (seasonal_var / total_var))
            
            return {
                'has_seasonality': strength > 0.3,
                'strength': round(float(strength * 100), 2),
                'period': period
            }
        except Exception:
            return {'has_seasonality': False, 'strength': 0}
    
    def detect_market_anomalies(self, df: pd.DataFrame, date_col: str,
                                 value_col: str, method: str = 'zscore') -> Dict[str, Any]:
        """Detect anomalies in market data."""
        try:
            df = df.copy()
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.dropna(subset=[date_col, value_col])
            df = df.sort_values(date_col)
            
            values = df[value_col].values.astype(float)
            
            anomalies = []
            if method == 'zscore':
                mean = np.mean(values)
                std = np.std(values)
                if std > 0:
                    z_scores = np.abs((values - mean) / std)
                    anomaly_indices = np.where(z_scores > 2.5)[0]
                    for idx in anomaly_indices:
                        anomalies.append({
                            'index': int(idx),
                            'date': df[date_col].iloc[idx].strftime('%Y-%m-%d') if hasattr(df[date_col].iloc[idx], 'strftime') else str(df[date_col].iloc[idx]),
                            'value': round(float(values[idx]), 4),
                            'z_score': round(float(z_scores[idx]), 4),
                            'expected': round(float(mean), 4),
                            'deviation': round(float(values[idx] - mean), 4)
                        })
            
            elif method == 'iqr':
                q1 = np.percentile(values, 25)
                q3 = np.percentile(values, 75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                anomaly_indices = np.where((values < lower) | (values > upper))[0]
                for idx in anomaly_indices:
                    anomalies.append({
                        'index': int(idx),
                        'date': df[date_col].iloc[idx].strftime('%Y-%m-%d') if hasattr(df[date_col].iloc[idx], 'strftime') else str(df[date_col].iloc[idx]),
                        'value': round(float(values[idx]), 4),
                        'boundary': 'upper' if values[idx] > upper else 'lower',
                        'expected_range': [round(float(lower), 4), round(float(upper), 4)]
                    })
            
            return {
                'success': True,
                'method': method,
                'anomaly_count': len(anomalies),
                'anomalies': anomalies[:50],  # Limit output
                'anomaly_percentage': round(len(anomalies) / len(values) * 100, 2) if len(values) > 0 else 0
            }
        
        except Exception as e:
            return {'success': False, 'error': str(e)}


class StrategicIntelligence:
    """Strategic intelligence and recommendation engine."""
    
    def __init__(self):
        pass
    
    def generate_strategy_recommendations(self, competitor_data: Dict = None,
                                          market_data: Dict = None,
                                          consumer_data: Dict = None,
                                          opportunity_data: Dict = None) -> List[str]:
        """Generate strategic recommendations based on all intelligence."""
        recommendations = []
        
        # Market-based recommendations
        if market_data and market_data.get('success'):
            trend = market_data.get('trend', 'stable')
            if trend == 'increasing':
                recommendations.append("Growing market: Increase investment in capacity and market expansion.")
            elif trend == 'decreasing':
                recommendations.append("Declining market: Focus on efficiency, cost reduction, and market consolidation.")
            
            if market_data.get('seasonality', {}).get('has_seasonality'):
                recommendations.append(f"Seasonal patterns detected: Adjust inventory and marketing for {market_data['seasonality'].get('period', 7)}-day cycles.")
        
        # Competitor-based recommendations
        if competitor_data and competitor_data.get('success'):
            hhi = competitor_data.get('market_concentration', {})
            if hhi.get('concentration_level') == 'high':
                recommendations.append("Consolidated market: Focus on differentiation and premium positioning.")
            
            analyses = competitor_data.get('analyses', [])
            high_threats = [a for a in analyses if a.get('threat_score', 0) > 70]
            if high_threats:
                recommendations.append(f"High threat from {high_threats[0]['competitor_name']}: Develop defensive strategies.")
        
        # Consumer-based recommendations
        if consumer_data and consumer_data.get('success'):
            brand = consumer_data.get('brand_health_score', 50)
            if brand < 50:
                recommendations.append("Low brand health: Launch brand recovery and customer engagement programs.")
            
            trust = consumer_data.get('trust_score', 50)
            if trust < 50:
                recommendations.append("Trust deficit: Implement transparency initiatives and quality improvements.")
        
        # Opportunity-based recommendations
        if opportunity_data and opportunity_data.get('success'):
            count = opportunity_data.get('total_detected', 0)
            if count > 5:
                recommendations.append(f"Multiple opportunities ({count}) detected: Establish opportunity evaluation framework.")
        
        # General recommendations
        if not recommendations:
            recommendations.append("Maintain current strategic direction while monitoring market conditions.")
        
        recommendations.append("Implement continuous competitive intelligence monitoring.")
        recommendations.append("Develop scenario plans for key strategic uncertainties.")
        
        return recommendations
