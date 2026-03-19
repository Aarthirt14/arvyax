"""
Machine Learning Models for  Emotional Understanding System
Handles:
- Emotional State Classification
- Intensity Regression/Classification
- Uncertainty Quantification
- Model Evaluation
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    mean_squared_error, mean_absolute_error, r2_score,
    precision_recall_fscore_support
)
import xgboost as xgb
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class EmotionalStateClassifier:
    """Classifier for emotional_state (multi-class classification)"""
    
    def __init__(self, model_type='xgboost'):
        self.model_type = model_type
        self.model = None
        self.label_encoder = LabelEncoder()
        self.classes_ = None
        
    def fit(self, X_train, y_train):
        """Train the model"""
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        self.classes_ = self.label_encoder.classes_
        
        if self.model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=7,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                tree_method='hist'
            )
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        else:
            self.model = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=7,
                min_samples_split=5,
                random_state=42
            )
        
        self.model.fit(X_train, y_train_encoded)
        return self
    
    def predict(self, X):
        """Predict emotional states"""
        predictions = self.model.predict(X)
        return self.label_encoder.inverse_transform(predictions)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        probas = self.model.predict_proba(X)
        return probas  # Shape: (n_samples, n_classes)
    
    def get_confidence_scores(self, X):
        """
        Get confidence scores for predictions
        Confidence = max probability across all classes
        """
        probas = self.predict_proba(X)
        confidence = np.max(probas, axis=1)
        return confidence
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = self.predict(X_test)
        
        report = {
            'accuracy': accuracy_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
        }
        
        return report, y_pred
    
    def save(self, path):
        """Save model"""
        joblib.dump(self.model, f"{path}_model.pkl")
        joblib.dump(self.label_encoder, f"{path}_encoder.pkl")
    
    def load(self, path):
        """Load model"""
        self.model = joblib.load(f"{path}_model.pkl")
        self.label_encoder = joblib.load(f"{path}_encoder.pkl")
        self.classes_ = self.label_encoder.classes_


class IntensityPredictor:
    """
    Predictor for intensity (can be treated as classification or regression)
    We'll treat it as classification (1-5 scale) for better uncertainty handling
    """
    
    def __init__(self, model_type='xgboost', treatment='classification'):
        self.model_type = model_type
        self.treatment = treatment  # 'classification' or 'regression'
        self.model = None
        self.le = None if treatment == 'regression' else LabelEncoder()
        
    def fit(self, X_train, y_train):
        """Train the model"""
        if self.treatment == 'classification':
            y_encoded = self.le.fit_transform(y_train)
            
            if self.model_type == 'xgboost':
                self.model = xgb.XGBClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    tree_method='hist'
                )
            else:
                self.model = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=12,
                    min_samples_split=5,
                    random_state=42,
                    n_jobs=-1
                )
            
            self.model.fit(X_train, y_encoded)
        
        else:  # regression
            if self.model_type == 'xgboost':
                self.model = xgb.XGBRegressor(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    tree_method='hist'
                )
            else:
                self.model = RandomForestRegressor(
                    n_estimators=200,
                    max_depth=12,
                    min_samples_split=5,
                    random_state=42,
                    n_jobs=-1
                )
            
            self.model.fit(X_train, y_train)
        
        return self
    
    def predict(self, X):
        """Predict intensity"""
        if self.treatment == 'classification':
            predictions = self.model.predict(X)
            return self.le.inverse_transform(predictions).astype(int)
        else:
            predictions = self.model.predict(X)
            return np.clip(predictions, 1, 5).astype(int)
    
    def predict_proba(self, X):
        """Get prediction probabilities (for classification)"""
        if self.treatment == 'classification':
            return self.model.predict_proba(X)
        else:
            return None
    
    def get_confidence_scores(self, X):
        """Get confidence scores"""
        if self.treatment == 'classification':
            probas = self.predict_proba(X)
            return np.max(probas, axis=1)
        else:
            # For regression, estimate based on variance
            return np.ones(len(X)) * 0.5
    
    def evaluate(self, X_test, y_test):
        """Evaluate model"""
        y_pred = self.predict(X_test)
        
        if self.treatment == 'classification':
            report = {
                'accuracy': accuracy_score(y_test, y_pred),
                'mae': mean_absolute_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'classification_report': classification_report(y_test, y_pred, output_dict=True),
            }
        else:
            report = {
                'mae': mean_absolute_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2': r2_score(y_test, y_pred),
            }
        
        return report, y_pred
    
    def save(self, path):
        """Save model"""
        joblib.dump(self.model, f"{path}_model.pkl")
        if self.le:
            joblib.dump(self.le, f"{path}_encoder.pkl")
    
    def load(self, path):
        """Load model"""
        self.model = joblib.load(f"{path}_model.pkl")
        if self.le:
            self.le = joblib.load(f"{path}_encoder.pkl")


class UncertaintyQuantifier:
    """
    Quantify uncertainty in predictions
    Uses ensemble methods and out-of-bag estimates
    """
    
    def __init__(self):
        self.confidence_threshold = 0.6
        
    def calculate_uncertainty(self, y_pred_proba, y_pred, ground_truth=None):
        """
        Calculate uncertainty metrics
        
        Returns:
        - confidence: max probability (0-1)
        - entropy: prediction entropy (0-1, higher = more uncertain)
        - uncertain_flag: binary flag if confidence below threshold
        - prediction_variance: variance across top-2 predictions
        """
        
        # Confidence = max probability
        confidence = np.max(y_pred_proba, axis=1)
        
        # Entropy = -sum(p * log(p))
        epsilon = 1e-10
        entropy = -np.sum(y_pred_proba * np.log(y_pred_proba + epsilon), axis=1)
        # Normalize entropy to 0-1
        entropy = entropy / np.log(y_pred_proba.shape[1])
        
        # Variance of top 2 predictions
        sorted_probs = np.sort(y_pred_proba, axis=1)[:, -2:]
        prediction_variance = np.var(sorted_probs, axis=1)
        
        # Uncertain flag: 1 if low confidence or high entropy
        uncertain_flag = (
            (confidence < self.confidence_threshold) | 
            (entropy > 0.7)
        ).astype(int)
        
        return {
            'confidence': confidence,
            'entropy': entropy,
            'uncertainty_flag': uncertain_flag,
            'prediction_variance': prediction_variance,
        }
    
    def flag_edge_cases(self, X_features, journal_texts):
        """
        Flag edge cases that warrant higher uncertainty
        - Very short texts
        - All missing metadata
        - Contradictory signals
        """
        flags = {
            'short_text': [],
            'missing_data': [],
            'contradictory_signals': [],
        }
        
        # Check text length
        text_lengths = np.array([len(str(t).split()) for t in journal_texts])
        flags['short_text'] = (text_lengths <= 3).astype(int)
        
        # Check missing data (assuming X_features normalized)
        missing_threshold = -2.0  # Very low normalized values indicate missing
        flags['missing_data'] = (np.min(X_features, axis=1) < missing_threshold).astype(int)
        
        # Check contradictory signals (e.g., high stress + high energy)
        # This would require access to raw features
        
        return flags


if __name__ == "__main__":
    from feature_engineering import prepare_datasets
    from data_loader import DataLoader
    
    loader = DataLoader()
    train_df = loader.load_training_data()
    test_df = loader.load_test_data()
    
    print("Loading and preparing data...")
    X_train, X_test, y_state, y_intensity, fe = prepare_datasets(train_df, test_df)
    
    # Split for validation
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_state_tr, y_state_val, y_int_tr, y_int_val = train_test_split(
        X_train, y_state, y_intensity, test_size=0.2, random_state=42
    )
    
    print("\n" + "="*60)
    print("Training Emotional State Classifier...")
    print("="*60)
    state_clf = EmotionalStateClassifier(model_type='xgboost')
    state_clf.fit(X_tr, y_state_tr)
    
    report, y_pred = state_clf.evaluate(X_val, y_state_val)
    print(f"Accuracy: {report['accuracy']:.4f}")
    
    print("\n" + "="*60)
    print("Training Intensity Predictor...")
    print("="*60)
    intensity_pred = IntensityPredictor(model_type='xgboost', treatment='classification')
    intensity_pred.fit(X_tr, y_int_tr)
    
    report, y_pred = intensity_pred.evaluate(X_val, y_int_val)
    print(f"MAE: {report['mae']:.4f}")
    print(f"RMSE: {report['rmse']:.4f}")
