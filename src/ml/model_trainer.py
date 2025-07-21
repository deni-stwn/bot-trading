import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
from typing import Dict, Tuple, Any
import logging

class MLModelTrainer:
    def __init__(self, model_save_dir: str = "models/trained_models"):
        self.model_save_dir = model_save_dir
        os.makedirs(model_save_dir, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100, 
                max_depth=10, 
                random_state=42
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'logistic': LogisticRegression(random_state=42)
        }
        
        self.best_model = None
        self.best_score = 0
        
    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train multiple ML models and select the best one"""
        results = {}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.logger.info(f"Training models on {len(X_train)} samples")
        
        for name, model in self.models.items():
            self.logger.info(f"Training {name}...")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                
                # Test predictions
                y_pred = model.predict(X_test)
                test_accuracy = accuracy_score(y_test, y_pred)
                
                # Store results
                results[name] = {
                    'model': model,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'test_accuracy': test_accuracy,
                    'classification_report': classification_report(y_test, y_pred),
                    'confusion_matrix': confusion_matrix(y_test, y_pred)
                }
                
                self.logger.info(f"{name} - CV: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
                self.logger.info(f"{name} - Test Accuracy: {test_accuracy:.4f}")
                
                # Update best model
                if cv_scores.mean() > self.best_score:
                    self.best_score = cv_scores.mean()
                    self.best_model = model
                    
            except Exception as e:
                self.logger.error(f"Error training {name}: {e}")
                continue
        
        # Save best model
        if self.best_model:
            model_path = os.path.join(self.model_save_dir, "best_model.pkl")
            joblib.dump(self.best_model, model_path)
            self.logger.info(f"Best model saved to {model_path}")
        
        return results
    
    def hyperparameter_tuning(self, X: pd.DataFrame, y: pd.Series, 
                            model_name: str = 'xgboost') -> Any:
        """Perform hyperparameter tuning for selected model"""
        self.logger.info(f"Hyperparameter tuning for {model_name}")
        
        if model_name == 'xgboost':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
            base_model = xgb.XGBClassifier(random_state=42)
            
        elif model_name == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            base_model = RandomForestClassifier(random_state=42)
        
        else:
            self.logger.error(f"Hyperparameter tuning not implemented for {model_name}")
            return None
        
        # Grid search
        grid_search = GridSearchCV(
            base_model, 
            param_grid, 
            cv=5, 
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        self.logger.info(f"Best parameters: {grid_search.best_params_}")
        self.logger.info(f"Best score: {grid_search.best_score_:.4f}")
        
        # Save tuned model
        tuned_model_path = os.path.join(self.model_save_dir, f"tuned_{model_name}.pkl")
        joblib.dump(grid_search.best_estimator_, tuned_model_path)
        
        return grid_search.best_estimator_
    
    def get_feature_importance(self, model: Any, feature_names: list) -> pd.DataFrame:
        """Get feature importance from trained model"""
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        else:
            self.logger.warning("Model doesn't have feature_importances_ attribute")
            return pd.DataFrame()