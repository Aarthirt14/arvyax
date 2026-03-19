"""
Feature Engineering for  Emotional Understanding System
Handles text features, metadata features, and feature engineering
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    def __init__(self):
        self.tfidf_vectorizer = None
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        
    def fit_transform(self, df, is_test=False):
        """
        Fit and transform features from dataframe
        is_test=True means we won't try to extract emotional_state/intensity
        """
        # Extract text features
        text_features = self._extract_text_features(df)
        
        # Extract metadata features  
        metadata_features = self._extract_metadata_features(df)
        
        # Combine
        combined_features = np.hstack([text_features, metadata_features])
        
        self.feature_names = (
            [f'text_{i}' for i in range(text_features.shape[1])] +
            [f'meta_{i}' for i in range(metadata_features.shape[1])]
        )
        
        return combined_features
    
    def transform(self, df):
        """Transform using fitted transformers"""
        text_features = self._extract_text_features(df, fit=False)
        metadata_features = self._extract_metadata_features(df, fit=False)
        combined_features = np.hstack([text_features, metadata_features])
        return combined_features
    
    def _extract_text_features(self, df, fit=True):
        """
        Extract features from journal_text:
        - TF-IDF vectors
        - Text length
        - Sentiment indicators
        - Question/punctuation density
        """
        texts = df['journal_text'].fillna("").astype(str)
        
        # TF-IDF vectorization
        if fit:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=50,
                min_df=2,
                max_df=0.8,
                ngram_range=(1, 2),
                stop_words='english'
            )
            tfidf_features = self.tfidf_vectorizer.fit_transform(texts).toarray()
        else:
            tfidf_features = self.tfidf_vectorizer.transform(texts).toarray()
        
        # Additional text features
        text_length = np.array([len(text.split()) for text in texts]).reshape(-1, 1)
        char_length = np.array([len(text) for text in texts]).reshape(-1, 1)
        
        # Sentiment indicators
        sentiment_scores = np.array([
            self._calculate_sentiment_score(text) for text in texts
        ]).reshape(-1, 1)
        
        # Punctuation/question density
        question_density = np.array([
            text.count('?') / max(len(text.split()), 1) for text in texts
        ]).reshape(-1, 1)
        
        exclamation_density = np.array([
            text.count('!') / max(len(text.split()), 1) for text in texts
        ]).reshape(-1, 1)
        
        # Negative word count
        negative_words = ['bad', 'wrong', 'hate', 'terrible', 'awful', 'sad', 'angry', 'frustrated', 'stressed', 'tired']
        negative_count = np.array([
            sum(text.lower().count(word) for word in negative_words)
            for text in texts
        ]).reshape(-1, 1)
        
        # Positive word count
        positive_words = ['good', 'great', 'love', 'happy', 'calm', 'peaceful', 'focused', 'energized', 'hope', 'better']
        positive_count = np.array([
            sum(text.lower().count(word) for word in positive_words)
            for text in texts
        ]).reshape(-1, 1)
        
        # Is text very short/vague
        is_short = (text_length <= 3).astype(int)
        
        # Combine all text features
        text_features = np.hstack([
            tfidf_features,
            text_length,
            char_length,
            sentiment_scores,
            question_density,
            exclamation_density,
            negative_count,
            positive_count,
            is_short
        ])
        
        return text_features
    
    def _calculate_sentiment_score(self, text):
        """
        Simple sentiment score based on positive/negative words
        Returns float between -1 and 1
        """
        if not text or len(text.split()) < 1:
            return 0.0
        
        positive_words = ['good', 'great', 'love', 'happy', 'calm', 'peaceful', 'focused', 
                         'energized', 'hope', 'better', 'amazing', 'wonderful', 'beautiful']
        negative_words = ['bad', 'wrong', 'hate', 'terrible', 'awful', 'sad', 'angry', 
                         'frustrated', 'stressed', 'tired', 'overwhelmed', 'anxious', 'depressed']
        
        text_lower = text.lower()
        pos_count = sum(text_lower.count(word) for word in positive_words)
        neg_count = sum(text_lower.count(word) for word in negative_words)
        
        if pos_count + neg_count == 0:
            return 0.0
        
        return (pos_count - neg_count) / (pos_count + neg_count)
    
    def _extract_metadata_features(self, df, fit=True):
        """
        Extract features from metadata:
        - Encode categorical variables
        - Scale numeric variables
        - Create interaction features
        """
        features_list = []
        
        # Numeric features - scale them
        numeric_cols = ['duration_min', 'sleep_hours', 'energy_level', 'stress_level']
        numeric_data = df[numeric_cols].fillna(df[numeric_cols].median()).values
        
        if fit:
            self.scalers['numeric'] = StandardScaler()
            scaled_numeric = self.scalers['numeric'].fit_transform(numeric_data)
        else:
            scaled_numeric = self.scalers['numeric'].transform(numeric_data)
        
        features_list.append(scaled_numeric)
        
        # Categorical features - encode them
        categorical_cols = ['ambience_type', 'time_of_day', 'previous_day_mood', 
                           'face_emotion_hint', 'reflection_quality']
        
        for col in categorical_cols:
            if col in df.columns:
                data = df[col].fillna('unknown').astype(str)
                
                if fit:
                    encoder = LabelEncoder()
                    encoded = encoder.fit_transform(data).reshape(-1, 1)
                    self.encoders[col] = encoder
                else:
                    encoded = self.encoders[col].transform(data).reshape(-1, 1)
                
                features_list.append(encoded)
        
        # Interaction features
        # Sleep * Stress (fatigue from stress)
        sleep_stress_interaction = (numeric_data[:, 1] * numeric_data[:, 3]).reshape(-1, 1)
        features_list.append(sleep_stress_interaction)
        
        # Energy * Duration (motivation)
        energy_duration_interaction = (numeric_data[:, 2] * numeric_data[:, 0]).reshape(-1, 1)
        features_list.append(energy_duration_interaction)
        
        # Stress / Energy (difficulty)
        stress_energy_ratio = (numeric_data[:, 3] / np.maximum(numeric_data[:, 2], 1)).reshape(-1, 1)
        features_list.append(stress_energy_ratio)
        
        metadata_features = np.hstack(features_list)
        return metadata_features
    
    def get_feature_importance_baseline(self, X, y):
        """
        Calculate baseline feature importance using correlation with target
        """
        from sklearn.ensemble import RandomForestClassifier
        
        # Only works if we have labels
        if y is None or len(y) == 0:
            return None
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        rf.fit(X, y)
        
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance


def prepare_datasets(train_df, test_df):
    """
    Prepare train and test datasets with feature engineering
    Returns: X_train_text, X_train_meta, y_train, X_test_text, X_test_meta
    """
    fe = FeatureEngineer()
    
    # Fit on training data
    X_train = fe.fit_transform(train_df, is_test=False)
    
    # Transform test data
    X_test = fe.transform(test_df)
    
    # Extract labels (only available in training)
    y_train_state = train_df['emotional_state'].astype(str)
    y_train_intensity = train_df['intensity'].astype(int)
    
    return X_train, X_test, y_train_state, y_train_intensity, fe


if __name__ == "__main__":
    from data_loader import DataLoader
    
    loader = DataLoader()
    train_df = loader.load_training_data()
    test_df = loader.load_test_data()
    
    print("Training data shape:", train_df.shape)
    print("Test data shape:", test_df.shape)
    
    # Test feature engineering
    X_train, X_test, y_state, y_intensity, fe = prepare_datasets(train_df, test_df)
    
    print(f"\nFeatures extracted:")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"Number of features: {len(fe.feature_names)}")
    print(f"Target - Emotional States: {len(np.unique(y_state))}")
    print(f"Target - Intensities: {np.unique(y_intensity)}")
