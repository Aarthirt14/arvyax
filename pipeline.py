"""
Main Pipeline for  Emotional Understanding System
Orchestrates: Data Loading → Feature Engineering → Model Training → Prediction → Decision
"""

import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

from data_loader import DataLoader
from feature_engineering import FeatureEngineer, prepare_datasets
from models import EmotionalStateClassifier, IntensityPredictor, UncertaintyQuantifier
from decision_engine import batch_decide


class Pipeline:
    """
    Complete pipeline for emotional understanding and guidance
    """
    
    def __init__(self, output_dir='models'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.data_loader = DataLoader()
        self.feature_engineer = None
        self.state_clf = None
        self.intensity_pred = None
        self.uncertainty_quantifier = UncertaintyQuantifier()
        
        self.train_data = None
        self.test_data = None
        self.predictions = None
        
    def load_data(self, train_path=None, test_path=None):
        """Load training and test data"""
        print("Loading data...")
        self.train_data = self.data_loader.load_training_data(train_path)
        self.test_data = self.data_loader.load_test_data(test_path)
        
        print(f"Training data: {self.train_data.shape}")
        print(f"Test data: {self.test_data.shape}")
        return self
    
    def prepare_features(self):
        """Engineer features from raw data"""
        print("\nEngineering features...")
        
        X_train, X_test, y_state, y_intensity, self.feature_engineer = prepare_datasets(
            self.train_data, self.test_data
        )
        
        print(f"Feature vectors shape: {X_train.shape}")
        print(f"Features: {len(self.feature_engineer.feature_names)}")
        
        return X_train, X_test, y_state, y_intensity
    
    def train_models(self, X_train, y_state, y_intensity, val_split=0.2):
        """
        Train state classifier and intensity predictor
        """
        from sklearn.model_selection import train_test_split
        
        print("\n" + "="*70)
        print("TRAINING MODELS")
        print("="*70)
        
        # Split into train/val
        X_tr, X_val, y_state_tr, y_state_val, y_int_tr, y_int_val = train_test_split(
            X_train, y_state, y_intensity, 
            test_size=val_split, random_state=42
        )
        
        # Train emotional state classifier
        print("\n1. Training Emotional State Classifier")
        print("-" * 50)
        self.state_clf = EmotionalStateClassifier(model_type='xgboost')
        self.state_clf.fit(X_tr, y_state_tr)
        
        report, y_pred = self.state_clf.evaluate(X_val, y_state_val)
        print(f"Validation Accuracy: {report['accuracy']:.4f}")
        print(f"Classes: {self.state_clf.classes_}")
        
        # Train intensity predictor
        print("\n2. Training Intensity Predictor (Classification)")
        print("-" * 50)
        self.intensity_pred = IntensityPredictor(
            model_type='xgboost', 
            treatment='classification'
        )
        self.intensity_pred.fit(X_tr, y_int_tr)
        
        report, y_pred = self.intensity_pred.evaluate(X_val, y_int_val)
        print(f"Validation MAE: {report['mae']:.4f}")
        print(f"Validation RMSE: {report['rmse']:.4f}")
        
        # Save models
        self.state_clf.save(str(self.output_dir / "state_classifier"))
        self.intensity_pred.save(str(self.output_dir / "intensity_predictor"))
        
        print(f"\nModels saved to {self.output_dir}")
        
        return self
    
    def predict_on_test(self, X_test):
        """
        Generate predictions on test data with uncertainty
        """
        print("\n" + "="*70)
        print("GENERATING PREDICTIONS")
        print("="*70)
        
        # Predict states
        print("\nPredicting emotional states...")
        predicted_states = self.state_clf.predict(X_test)
        state_probas = self.state_clf.predict_proba(X_test)
        state_confidence = self.state_clf.get_confidence_scores(X_test)
        
        # Predict intensities
        print("Predicting intensities...")
        predicted_intensities = self.intensity_pred.predict(X_test)
        intensity_probas = self.intensity_pred.predict_proba(X_test)
        intensity_confidence = self.intensity_pred.get_confidence_scores(X_test)
        
        # Quantify uncertainty
        print("Quantifying uncertainty...")
        state_uncertainty = self.uncertainty_quantifier.calculate_uncertainty(
            state_probas, predicted_states
        )
        
        intensity_uncertainty = self.uncertainty_quantifier.calculate_uncertainty(
            intensity_probas, predicted_intensities
        )
        
        # Combine uncertainties
        combined_confidence = (state_confidence * 0.6 + intensity_confidence * 0.4)
        combined_uncertain_flag = np.maximum(
            state_uncertainty['uncertainty_flag'],
            intensity_uncertainty['uncertainty_flag']
        )
        
        # Build predictions dataframe
        self.predictions = pd.DataFrame({
            'id': self.test_data['id'],
            'predicted_state': predicted_states,
            'predicted_intensity': predicted_intensities,
            'state_confidence': state_confidence,
            'intensity_confidence': intensity_confidence,
            'combined_confidence': combined_confidence,
            'uncertain_flag': combined_uncertain_flag,
        })
        
        print(f"\nPredictions generated: {len(self.predictions)}")
        print(f"Uncertain samples: {combined_uncertain_flag.sum()} ({100*combined_uncertain_flag.mean():.1f}%)")
        
        return self.predictions
    
    def apply_decision_engine(self):
        """
        Apply decision engine to generate actions and timing
        """
        print("\n" + "="*70)
        print("APPLYING DECISION ENGINE")
        print("="*70)
        
        # Apply decision logic
        decision_results = batch_decide(self.predictions, self.test_data)
        
        # Merge with predictions
        self.predictions = pd.concat(
            [self.predictions, decision_results],
            axis=1
        )
        
        print(f"Actions assigned: {self.predictions['what_to_do'].nunique()} types")
        print(f"Timing assigned: {self.predictions['when_to_do'].nunique()} options")
        
        return self
    
    def generate_output_csv(self, output_file='predictions.csv'):
        """Save predictions in required format"""
        output_df = self.predictions[[
            'id', 'predicted_state', 'predicted_intensity',
            'combined_confidence', 'uncertain_flag',
            'what_to_do', 'when_to_do', 'supportive_message'
        ]].copy()
        
        output_df.columns = [
            'id', 'predicted_state', 'predicted_intensity',
            'confidence', 'uncertain_flag',
            'what_to_do', 'when_to_do', 'supportive_message'
        ]
        
        output_path = Path(output_file)
        output_df.to_csv(output_path, index=False)
        print(f"\nPredictions saved to {output_path}")
        
        # Print sample
        print("\nSample predictions:")
        print(output_df.head(10).to_string())
        
        return output_path
    
    def analyze_feature_importance(self):
        """Analyze which features matter most"""
        print("\n" + "="*70)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*70)
        
        # Get importance from state classifier
        try:
            feature_importance = pd.DataFrame({
                'feature': self.feature_engineer.feature_names,
                'importance': self.state_clf.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nTop 20 Most Important Features:")
            print(feature_importance.head(20).to_string())
            
            # Summary
            text_feature_importance = feature_importance[
                feature_importance['feature'].str.startswith('text_')
            ]['importance'].sum()
            meta_feature_importance = feature_importance[
                feature_importance['feature'].str.startswith('meta_')
            ]['importance'].sum()
            
            print(f"\nText Features Importance: {100*text_feature_importance:.1f}%")
            print(f"Metadata Features Importance: {100*meta_feature_importance:.1f}%")
            
            return feature_importance
        except Exception as e:
            print(f"Could not extract feature importance: {e}")
            return None
    
    def run_full_pipeline(self, train_path=None, test_path=None, output_file='predictions.csv'):
        """
        Run complete pipeline end-to-end
        """
        print("\n" + "="*70)
        print(" EMOTIONAL UNDERSTANDING SYSTEM - FULL PIPELINE")
        print("="*70)
        
        # Load data
        self.load_data(train_path, test_path)
        
        # Prepare features
        X_train, X_test, y_state, y_intensity = self.prepare_features()
        
        # Train models
        self.train_models(X_train, y_state, y_intensity)
        
        # Make predictions
        self.predict_on_test(X_test)
        
        # Apply decision engine
        self.apply_decision_engine()
        
        # Analyze features
        self.analyze_feature_importance()
        
        # Save output
        output_path = self.generate_output_csv(output_file)
        
        print("\n" + "="*70)
        print("PIPELINE COMPLETE")
        print("="*70)
        
        return self.predictions


if __name__ == "__main__":
    pipeline = Pipeline()
    
    # Run full pipeline
    predictions = pipeline.run_full_pipeline(
        train_path='Sample__reflective_dataset.xlsx',
        test_path='_test_inputs_120.xlsx',
        output_file='predictions.csv'
    )
