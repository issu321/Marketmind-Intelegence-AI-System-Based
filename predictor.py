"""
MarketMind - ML Forecasting Engine
Provides time series forecasting using multiple ML models.
Models: Random Forest, XGBoost, LightGBM, CatBoost, Gradient Boosting
"""

import json
import warnings
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

MODEL_REGISTRY = {}

try:
    import xgboost as xgb
    MODEL_REGISTRY['xgboost'] = xgb.XGBRegressor
except ImportError:
    pass

try:
    import lightgbm as lgb
    MODEL_REGISTRY['lightgbm'] = lgb.LGBMRegressor
except ImportError:
    pass

try:
    from catboost import CatBoostRegressor
    MODEL_REGISTRY['catboost'] = CatBoostRegressor
except ImportError:
    pass

MODEL_REGISTRY['random_forest'] = RandomForestRegressor
MODEL_REGISTRY['gradient_boosting'] = GradientBoostingRegressor


class ForecastingEngine:
    """Enterprise-grade ML forecasting engine."""
    
    def __init__(self, model_type: str = 'auto', random_state: int = 42):
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = {}
        self.metrics = {}
        self.is_trained = False
        self.best_params = {}
        self.training_log = []
        self.trained_model_name = 'auto'
    
    def _create_date_features(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Create time-based features from dates."""
        features = pd.DataFrame(index=dates)
        features['year'] = dates.year
        features['month'] = dates.month
        features['quarter'] = dates.quarter
        features['day_of_year'] = dates.dayofyear
        features['day_of_month'] = dates.day
        features['day_of_week'] = dates.dayofweek
        features['week_of_year'] = dates.isocalendar().week.astype(int)
        features['is_month_start'] = dates.is_month_start.astype(int)
        features['is_month_end'] = dates.is_month_end.astype(int)
        features['is_quarter_start'] = dates.is_quarter_start.astype(int)
        features['is_quarter_end'] = dates.is_quarter_end.astype(int)
        features['is_year_start'] = dates.is_year_start.astype(int)
        features['is_year_end'] = dates.is_year_end.astype(int)
        features['days_from_epoch'] = (dates - pd.Timestamp('1970-01-01')).days
        
        # Cyclical encoding for month, day_of_week
        features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
        features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
        features['dow_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
        features['dow_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
        features['doy_sin'] = np.sin(2 * np.pi * features['day_of_year'] / 365)
        features['doy_cos'] = np.cos(2 * np.pi * features['day_of_year'] / 365)
        
        return features
    
    def _create_lag_features(self, values: np.ndarray, lags: List[int] = None) -> pd.DataFrame:
        """Create lag features from values."""
        if lags is None:
            lags = [1, 2, 3, 7, 14, 30]
        lags = [l for l in lags if l < len(values)]
        
        features = pd.DataFrame()
        for lag in lags:
            features[f'lag_{lag}'] = pd.Series(values).shift(lag).values
        
        # Rolling statistics
        for window in [3, 7, 14, 30]:
            if window < len(values):
                features[f'rolling_mean_{window}'] = pd.Series(values).rolling(window=window, min_periods=1).mean().values
                features[f'rolling_std_{window}'] = pd.Series(values).rolling(window=window, min_periods=1).std().fillna(0).values
                features[f'rolling_min_{window}'] = pd.Series(values).rolling(window=window, min_periods=1).min().values
                features[f'rolling_max_{window}'] = pd.Series(values).rolling(window=window, min_periods=1).max().values
        
        # Expanding statistics
        features['expanding_mean'] = pd.Series(values).expanding(min_periods=1).mean().values
        features['expanding_std'] = pd.Series(values).expanding(min_periods=1).std().fillna(0).values
        
        # Differences
        features['diff_1'] = pd.Series(values).diff(1).fillna(0).values
        features['diff_7'] = pd.Series(values).diff(7).fillna(0).values
        features['pct_change_1'] = pd.Series(values).pct_change(1).fillna(0).replace([np.inf, -np.inf], 0).values
        
        return features
    
    def _prepare_features(self, df: pd.DataFrame, date_col: str, value_col: str,
                          target_horizon: int = 1) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for model training."""
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)
        
        dates = pd.DatetimeIndex(df[date_col])
        values = df[value_col].values.astype(float)
        
        # Create features
        date_features = self._create_date_features(dates)
        lag_features = self._create_lag_features(values)
        
        # CRITICAL FIX: Reset indices to ensure proper alignment
        date_features = date_features.reset_index(drop=True)
        lag_features = lag_features.reset_index(drop=True)
        
        # Combine features
        X = pd.concat([date_features, lag_features], axis=1)
        
        # Create target (forecast horizon ahead)
        if target_horizon > 0:
            y = pd.Series(values).shift(-target_horizon).dropna().values
            X = X.iloc[:len(y)]
        else:
            y = values
        
        # Fill NaN values
        X = X.fillna(X.median())
        
        return X, pd.Series(y)
    
    def _get_model(self, model_type: str):
        """Get model instance by type."""
        if model_type not in MODEL_REGISTRY:
            model_type = 'random_forest'
        
        model_class = MODEL_REGISTRY[model_type]
        
        common_params = {
            'random_state': self.random_state,
            'n_jobs': -1
        }
        
        if model_type == 'random_forest':
            return model_class(n_estimators=200, max_depth=15, min_samples_split=5,
                             min_samples_leaf=2, **common_params)
        elif model_type == 'xgboost':
            return model_class(n_estimators=200, max_depth=8, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, **common_params)
        elif model_type == 'lightgbm':
            return model_class(n_estimators=200, max_depth=8, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, verbose=-1, **common_params)
        elif model_type == 'catboost':
            return model_class(iterations=200, depth=8, learning_rate=0.05,
                             verbose=False, random_seed=self.random_state)
        elif model_type == 'gradient_boosting':
            return model_class(n_estimators=200, max_depth=6, learning_rate=0.1,
                             min_samples_split=5, **common_params)
        
        return model_class(**common_params)
    
    def _evaluate_model(self, model, X_train: pd.DataFrame, y_train: pd.Series,
                        X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate model performance."""
        try:
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            mae = mean_absolute_error(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, predictions)
            
            # MAPE
            mask = y_test != 0
            mape = np.mean(np.abs((y_test[mask] - predictions[mask]) / y_test[mask])) * 100 if mask.any() else float('inf')
            
            # Directional accuracy
            if len(y_test) > 1:
                actual_dir = np.sign(np.diff(y_test))
                pred_dir = np.sign(np.diff(predictions))
                dir_accuracy = np.mean(actual_dir == pred_dir) * 100
            else:
                dir_accuracy = 0.0
            
            return {
                'mae': float(mae),
                'mse': float(mse),
                'rmse': float(rmse),
                'r2': float(r2),
                'mape': float(mape) if not np.isinf(mape) else 999.0,
                'directional_accuracy': float(dir_accuracy)
            }
        except Exception as e:
            self.training_log.append(f"Evaluation error: {str(e)}")
            return {'mae': float('inf'), 'mse': float('inf'), 'rmse': float('inf'),
                    'r2': -float('inf'), 'mape': float('inf'), 'directional_accuracy': 0.0}
    
    def _select_best_model(self, X: pd.DataFrame, y: pd.Series) -> Tuple[str, Any, Dict]:
        """Auto-select the best model using cross-validation."""
        if self.model_type != 'auto' and self.model_type in MODEL_REGISTRY:
            model = self._get_model(self.model_type)
            # Use a proper train/test split instead of evaluating on training data
            split_idx = int(len(X) * 0.8)
            if split_idx < 10:
                split_idx = len(X) // 2
            metrics = self._evaluate_model(model, X.iloc[:split_idx], y.iloc[:split_idx], 
                                          X.iloc[split_idx:], y.iloc[split_idx:])
            return self.model_type, model, metrics
        
        models_to_test = ['random_forest', 'gradient_boosting']
        if 'xgboost' in MODEL_REGISTRY:
            models_to_test.append('xgboost')
        if 'lightgbm' in MODEL_REGISTRY:
            models_to_test.append('lightgbm')
        if 'catboost' in MODEL_REGISTRY:
            models_to_test.append('catboost')
        
        best_score = float('inf')
        best_model_name = 'random_forest'
        best_model = None
        best_metrics = {}
        
        # Use time series split for validation
        tscv = TimeSeriesSplit(n_splits=min(3, len(X) // 10))
        
        for model_name in models_to_test:
            try:
                scores = []
                fold_metrics = []
                
                for train_idx, val_idx in tscv.split(X):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    model = self._get_model(model_name)
                    metrics = self._evaluate_model(model, X_train, y_train, X_val, y_val)
                    scores.append(metrics['rmse'])
                    fold_metrics.append(metrics)
                
                avg_rmse = np.mean(scores)
                self.training_log.append(f"{model_name}: avg RMSE = {avg_rmse:.4f}")
                
                if avg_rmse < best_score:
                    best_score = avg_rmse
                    best_model_name = model_name
                    # Average metrics across folds
                    best_metrics = {
                        k: float(np.mean([m[k] for m in fold_metrics]))
                        for k in fold_metrics[0].keys()
                    }
            except Exception as e:
                self.training_log.append(f"{model_name} failed: {str(e)}")
                continue
        
        # Train final model on full data
        best_model = self._get_model(best_model_name)
        self.training_log.append(f"Selected model: {best_model_name}")
        
        return best_model_name, best_model, best_metrics
    
    def train(self, df: pd.DataFrame, date_col: str, value_col: str,
              forecast_horizon: int = 1) -> Dict[str, Any]:
        """Train the forecasting model."""
        try:
            X, y = self._prepare_features(df, date_col, value_col, forecast_horizon)
            
            if len(X) < 10:
                raise ValueError("Insufficient data for training. Need at least 10 data points.")
            
            # Select and train best model
            model_name, self.model, self.metrics = self._select_best_model(X, y)
            self.trained_model_name = model_name
            
            # Train on full dataset
            self.model.fit(X, y)
            
            # Calculate feature importance
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
            elif hasattr(self.model, 'feature_importance_'):
                importances = self.model.feature_importance_
            else:
                importances = np.ones(len(X.columns)) / len(X.columns)
            
            self.feature_importance = dict(sorted(
                zip(X.columns, importances),
                key=lambda x: x[1],
                reverse=True
            )[:15])
            
            self.is_trained = True
            
            return {
                'success': True,
                'model_name': model_name,
                'metrics': self.metrics,
                'feature_importance': self.feature_importance,
                'training_log': self.training_log,
                'feature_count': len(X.columns),
                'training_samples': len(X)
            }
        
        except Exception as e:
            self.training_log.append(f"Training error: {traceback.format_exc()}")
            return {
                'success': False,
                'error': str(e),
                'training_log': self.training_log
            }
    
    def forecast(self, df: pd.DataFrame, date_col: str, value_col: str,
                 horizon: int = 30, confidence: float = 0.95) -> Dict[str, Any]:
        """Generate forecasts for the specified horizon."""
        if not self.is_trained and self.model is None:
            train_result = self.train(df, date_col, value_col)
            if not train_result['success']:
                return {'success': False, 'error': train_result.get('error', 'Training failed')}
        
        try:
            df = df.copy()
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(date_col).reset_index(drop=True)
            
            last_date = df[date_col].max()
            values = df[value_col].values.astype(float)
            
            # Determine frequency
            date_diffs = df[date_col].diff().dropna()
            if len(date_diffs) > 0:
                median_diff = date_diffs.median()
                median_days = median_diff.total_seconds() / 86400
                if median_days >= 28:
                    freq = 'MS'
                elif median_days >= 7:
                    freq = 'W-MON'
                else:
                    freq = 'D'
            else:
                freq = 'D'
            
            # Generate future dates
            future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq=freq)[1:]
            
            # Iterative forecasting
            current_values = list(values)
            predictions = []
            lower_bounds = []
            upper_bounds = []
            
            for i in range(horizon):
                # Create features for prediction
                pred_dates = pd.DatetimeIndex([future_dates[i]])
                date_features = self._create_date_features(pred_dates)
                lag_features = self._create_lag_features(np.array(current_values))
                
                # CRITICAL FIX: Align indices properly
                date_features = date_features.reset_index(drop=True)
                lag_features_last = lag_features.iloc[[-1]].reset_index(drop=True)
                
                X_pred = pd.concat([date_features, lag_features_last], axis=1)
                X_pred = X_pred.fillna(0)
                
                # Ensure column order matches training
                if hasattr(self.model, 'feature_names_in_'):
                    for col in self.model.feature_names_in_:
                        if col not in X_pred.columns:
                            X_pred[col] = 0
                    X_pred = X_pred[self.model.feature_names_in_]
                
                pred = float(self.model.predict(X_pred)[0])
                pred = max(0, pred)  # Ensure non-negative for most business metrics
                
                predictions.append(pred)
                current_values.append(pred)
                
                # Confidence intervals (using prediction variance)
                if len(values) > 10:
                    std = np.std(values[-min(30, len(values)):])
                    z_score = 1.96 if confidence >= 0.95 else 1.645 if confidence >= 0.90 else 1.28
                    margin = z_score * std * np.sqrt(1 + i * 0.05)
                    lower_bounds.append(max(0, pred - margin))
                    upper_bounds.append(pred + margin)
                else:
                    lower_bounds.append(pred * 0.9)
                    upper_bounds.append(pred * 1.1)
            
            # Prepare forecast results
            forecast_data = []
            for i in range(horizon):
                forecast_data.append({
                    'date': future_dates[i].strftime('%Y-%m-%d'),
                    'prediction': round(predictions[i], 4),
                    'lower_bound': round(lower_bounds[i], 4),
                    'upper_bound': round(upper_bounds[i], 4)
                })
            
            # Calculate forecast insights
            avg_prediction = np.mean(predictions)
            last_actual = values[-1]
            growth_projection = ((predictions[-1] - last_actual) / abs(last_actual) * 100) if last_actual != 0 else 0
            
            return {
                'success': True,
                'forecast_data': forecast_data,
                'model_name': self.trained_model_name,
                'metrics': self.metrics,
                'feature_importance': self.feature_importance,
                'summary': {
                    'horizon': horizon,
                    'avg_prediction': round(float(avg_prediction), 4),
                    'last_actual': round(float(last_actual), 4),
                    'growth_projection': round(float(growth_projection), 2),
                    'confidence_level': confidence,
                    'prediction_trend': 'increasing' if predictions[-1] > predictions[0] else 'decreasing' if predictions[-1] < predictions[0] else 'stable',
                    'max_prediction': round(float(max(predictions)), 4),
                    'min_prediction': round(float(min(predictions)), 4)
                },
                'insights': self._generate_forecast_insights(values, predictions, forecast_data)
            }
        
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def _generate_forecast_insights(self, historical: np.ndarray, predictions: List[float],
                                    forecast_data: List[Dict]) -> List[str]:
        """Generate human-readable insights from forecasts."""
        insights = []
        
        last_actual = historical[-1]
        first_pred = predictions[0]
        last_pred = predictions[-1]
        
        # Trend insight
        if last_pred > first_pred * 1.05:
            insights.append(f"The forecast indicates an upward trend with a projected increase of {((last_pred - first_pred) / first_pred * 100):.1f}% over the forecast period.")
        elif last_pred < first_pred * 0.95:
            insights.append(f"The forecast shows a downward trend with a projected decrease of {((first_pred - last_pred) / first_pred * 100):.1f}% over the forecast period.")
        else:
            insights.append("The forecast indicates stable performance with minimal variation over the forecast period.")
        
        # Growth vs historical
        historical_growth = np.mean(np.diff(historical[-min(30, len(historical)):]))
        predicted_growth = np.mean(np.diff(predictions))
        if predicted_growth > historical_growth * 1.2:
            insights.append("Predicted growth rate exceeds recent historical performance, suggesting positive momentum.")
        elif predicted_growth < historical_growth * 0.8:
            insights.append("Predicted growth rate is below recent historical trends, warranting attention.")
        
        # Volatility insight
        pred_volatility = np.std(predictions)
        hist_volatility = np.std(historical[-min(30, len(historical)):])
        if pred_volatility > hist_volatility * 1.5:
            insights.append("Higher volatility is expected in the forecast period compared to recent history.")
        elif pred_volatility < hist_volatility * 0.7:
            insights.append("The forecast period shows more stability compared to recent historical volatility.")
        
        # Seasonal pattern
        if len(predictions) >= 14:
            mid_point = len(predictions) // 2
            first_half_avg = np.mean(predictions[:mid_point])
            second_half_avg = np.mean(predictions[mid_point:])
            if abs(second_half_avg - first_half_avg) / max(first_half_avg, 0.001) > 0.1:
                insights.append(f"A {'strengthening' if second_half_avg > first_half_avg else 'weakening'} pattern is observed in the latter half of the forecast.")
        
        return insights
    
    def backtest(self, df: pd.DataFrame, date_col: str, value_col: str,
                 test_size: float = 0.2) -> Dict[str, Any]:
        """Perform backtesting on historical data."""
        try:
            df = df.copy()
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(date_col).reset_index(drop=True)
            
            split_idx = int(len(df) * (1 - test_size))
            train_df = df.iloc[:split_idx]
            test_df = df.iloc[split_idx:]
            
            # Train on training set
            X_train, y_train = self._prepare_features(train_df, date_col, value_col)
            
            if len(X_train) < 10:
                return {'success': False, 'error': 'Insufficient training data for backtesting'}
            
            model_name, model, _ = self._select_best_model(X_train, y_train)
            model.fit(X_train, y_train)
            
            # Predict on test set
            actual_values = test_df[value_col].values.astype(float)
            predictions = []
            
            current_values = list(train_df[value_col].values.astype(float))
            
            for i in range(len(test_df)):
                pred_dates = pd.DatetimeIndex([test_df[date_col].iloc[i]])
                date_features = self._create_date_features(pred_dates)
                lag_features = self._create_lag_features(np.array(current_values))
                
                # Fix alignment
                date_features = date_features.reset_index(drop=True)
                lag_features_last = lag_features.iloc[[-1]].reset_index(drop=True)
                
                X_pred = pd.concat([date_features, lag_features_last], axis=1)
                X_pred = X_pred.fillna(0)
                
                if hasattr(model, 'feature_names_in_'):
                    for col in model.feature_names_in_:
                        if col not in X_pred.columns:
                            X_pred[col] = 0
                    X_pred = X_pred[model.feature_names_in_]
                
                pred = float(model.predict(X_pred)[0])
                predictions.append(max(0, pred))
                current_values.append(actual_values[i])
            
            # Calculate metrics
            mae = mean_absolute_error(actual_values, predictions)
            rmse = np.sqrt(mean_squared_error(actual_values, predictions))
            r2 = r2_score(actual_values, predictions)
            
            mask = actual_values != 0
            mape = np.mean(np.abs((actual_values[mask] - np.array(predictions)[mask]) / actual_values[mask])) * 100 if mask.any() else 0
            
            actual_dir = np.sign(np.diff(actual_values))
            pred_dir = np.sign(np.diff(predictions))
            dir_accuracy = np.mean(actual_dir == pred_dir) * 100 if len(actual_dir) > 0 else 0
            
            return {
                'success': True,
                'model_name': model_name,
                'metrics': {
                    'mae': round(float(mae), 4),
                    'rmse': round(float(rmse), 4),
                    'r2': round(float(r2), 4),
                    'mape': round(float(mape), 2),
                    'directional_accuracy': round(float(dir_accuracy), 2)
                },
                'actual': [round(float(v), 4) for v in actual_values],
                'predicted': [round(float(v), 4) for v in predictions],
                'dates': [d.strftime('%Y-%m-%d') for d in test_df[date_col]]
            }
        
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_feature_importance_chart(self) -> Dict[str, Any]:
        """Get feature importance data for visualization."""
        if not self.feature_importance:
            return {}
        
        return {
            'features': list(self.feature_importance.keys()),
            'importance': list(self.feature_importance.values()),
            'colors': self._generate_importance_colors(len(self.feature_importance))
        }
    
    def _generate_importance_colors(self, n: int) -> List[str]:
        """Generate colors for feature importance chart."""
        colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6',
                  '#ec4899', '#06b6d4', '#84cc16', '#f97316', '#6366f1']
        return (colors * ((n // len(colors)) + 1))[:n]


class DemandForecaster(ForecastingEngine):
    """Specialized forecaster for demand prediction."""
    
    def forecast_demand(self, df: pd.DataFrame, date_col: str, demand_col: str,
                       product_col: str = None, horizon: int = 30) -> Dict[str, Any]:
        """Forecast demand with optional product segmentation."""
        if product_col and product_col in df.columns:
            results = {}
            for product in df[product_col].unique():
                product_df = df[df[product_col] == product].copy()
                if len(product_df) >= 10:
                    result = self.forecast(product_df, date_col, demand_col, horizon)
                    results[str(product)] = result
            return {'success': True, 'product_forecasts': results, 'type': 'segmented'}
        else:
            return self.forecast(df, date_col, demand_col, horizon)


class RevenueForecaster(ForecastingEngine):
    """Specialized forecaster for revenue prediction."""
    
    def forecast_revenue(self, df: pd.DataFrame, date_col: str, revenue_col: str,
                        segment_col: str = None, horizon: int = 90) -> Dict[str, Any]:
        """Forecast revenue with trend analysis."""
        result = self.forecast(df, date_col, revenue_col, horizon)
        
        if result.get('success'):
            predictions = [d['prediction'] for d in result['forecast_data']]
            
            # Calculate revenue metrics
            total_predicted = sum(predictions)
            avg_monthly = np.mean(predictions[:30]) if len(predictions) >= 30 else np.mean(predictions)
            
            result['revenue_metrics'] = {
                'total_predicted_revenue': round(float(total_predicted), 2),
                'avg_daily_revenue': round(float(np.mean(predictions)), 2),
                'projected_monthly_avg': round(float(avg_monthly * 30), 2),
                'revenue_growth_rate': round(float((predictions[-1] - predictions[0]) / max(predictions[0], 0.001) * 100), 2)
            }
        
        return result