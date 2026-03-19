"""
Label Noise Handling for 
Identifies and handles mislabeled samples in training data
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, List


class LabelNoiseDetector:
    """
    Detects potentially mislabeled samples
    Uses multiple strategies:
    1. Self-training confidence
    2. Duplicate/similar sample disagreement
    3. Feature-label disagreement
    """
    
    def __init__(self, contamination_rate: float = 0.05):
        """
        Args:
            contamination_rate: Expected fraction of mislabeled samples (5% default)
        """
        self.contamination_rate = contamination_rate
        self.noise_scores = None
        self.noisy_indices = []
    
    def detect_noise(self, X: np.ndarray, y: np.ndarray, 
                     method: str = 'ensemble') -> Tuple[np.ndarray, List[int]]:
        """
        Detect noisy labels using ensemble method
        
        Returns:
            noise_scores: Array of noise scores (higher = more likely mislabeled)
            noisy_indices: Indices of samples flagged as noisy
        """
        
        if method == 'ensemble':
            scores = self._ensemble_noise_detection(X, y)
        elif method == 'knn':
            scores = self._knn_noise_detection(X, y)
        elif method == 'entropy':
            scores = self._entropy_noise_detection(X, y)
        else:
            scores = self._ensemble_noise_detection(X, y)
        
        self.noise_scores = scores
        
        # Flag top contamination_rate samples as noisy
        n_noisy = max(1, int(len(y) * self.contamination_rate))
        self.noisy_indices = np.argsort(scores)[-n_noisy:].tolist()
        
        return scores, self.noisy_indices
    
    def _ensemble_noise_detection(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Train model and get self-training uncertainty
        High uncertainty on training data = likely noisy label
        """
        # Train on all data
        rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        rf.fit(X, y)
        
        # Get prediction confidence (max probability)
        probas = rf.predict_proba(X)
        confidence = np.max(probas, axis=1)
        
        # Noise score = 1 - confidence (inverted)
        noise_score = 1 - confidence
        
        return noise_score
    
    def _knn_noise_detection(self, X: np.ndarray, y: np.ndarray, k: int = 5) -> np.ndarray:
        """
        KNN-based detection: if k nearest neighbors have different label, likely noisy
        """
        from sklearn.neighbors import NearestNeighbors
        
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
        distances, indices = nbrs.kneighbors(X)
        
        noise_scores = np.zeros(len(y))
        
        for i, neighbors_idx in enumerate(indices):
            neighbor_labels = y[neighbors_idx[1:]]  # Skip self
            label_agreement = (neighbor_labels == y[i]).mean()
            noise_scores[i] = 1 - label_agreement
        
        return noise_scores
    
    def _entropy_noise_detection(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Entropy-based: high entropy outputs = uncertain predictions = likely noisy
        """
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        probas = rf.predict_proba(X)
        
        # Calculate entropy
        epsilon = 1e-10
        entropy = -np.sum(probas * np.log(probas + epsilon), axis=1)
        entropy = entropy / np.log(len(np.unique(y)))  # Normalize
        
        return entropy
    
    def get_clean_dataset(self, X: np.ndarray, y: np.ndarray, 
                         remove_noisy: bool = False,
                         weight_noisy: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get dataset after noise handling
        
        Args:
            X, y: Original data
            remove_noisy: If True, remove noisy samples. If False, weight them.
            weight_noisy: If True, return sample weights
        
        Returns:
            X_clean, y_clean, sample_weights
        """
        if self.noise_scores is None:
            self.detect_noise(X, y)
        
        if remove_noisy:
            # Remove noisy samples
            clean_mask = ~np.isin(np.arange(len(y)), self.noisy_indices)
            X_clean = X[clean_mask]
            y_clean = y[clean_mask]
            sample_weights = np.ones(len(y_clean))
        else:
            # Keep all, but weight noisy samples down
            X_clean = X
            y_clean = y
            sample_weights = 1 - self.noise_scores  # Weight down noisy samples
            sample_weights = sample_weights / sample_weights.mean()  # Normalize
        
        return X_clean, y_clean, sample_weights
    
    def print_noise_report(self, y: np.ndarray, sample_mask: np.ndarray = None):
        """Print noise detection report"""
        print("\n" + "="*70)
        print("LABEL NOISE DETECTION REPORT")
        print("="*70)
        
        if self.noise_scores is None:
            print("No noise detection performed yet")
            return
        
        n_noisy = len(self.noisy_indices)
        n_total = len(y)
        
        print(f"\nTotal samples analyzed: {n_total}")
        print(f"Samples flagged as noisy: {n_noisy} ({100*n_noisy/n_total:.1f}%)")
        print(f"Noise score range: [{self.noise_scores.min():.3f}, {self.noise_scores.max():.3f}]")
        
        # Show label distribution of noisy samples
        if len(self.noisy_indices) > 0:
            noisy_labels = y[self.noisy_indices]
            print(f"\nNoisy sample label distribution:")
            for label in np.unique(y):
                count = (noisy_labels == label).sum()
                pct = 100 * count / len(noisy_labels)
                print(f"  {label}: {count} ({pct:.1f}%)")
        
        # Show examples of high-noise samples
        print(f"\nTop 5 samples with highest noise scores:")
        top_noisy_idx = np.argsort(self.noise_scores)[-5:][::-1]
        for rank, idx in enumerate(top_noisy_idx, 1):
            print(f"  {rank}. Index {idx}: Label '{y[idx]}', Noise score {self.noise_scores[idx]:.3f}")


class NoiseRobustTrainer:
    """
    Train models robust to label noise
    Uses techniques: sample weighting, confidence threshold, noisy label correction
    """
    
    def __init__(self, base_model_class, contamination_rate: float = 0.05):
        self.base_model_class = base_model_class
        self.contamination_rate = contamination_rate
        self.noise_detector = LabelNoiseDetector(contamination_rate)
        self.model = None
        self.sample_weights = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, handle_noise: str = 'weight') -> 'NoiseRobustTrainer':
        """
        Train model with noise handling
        
        Args:
            handle_noise: 'weight' (down-weight noisy), 'remove' (delete noisy), 'none'
        """
        
        if handle_noise != 'none':
            print(f"\nDetecting label noise ({handle_noise} strategy)...")
            self.noise_detector.detect_noise(X, y)
            self.noise_detector.print_noise_report(y)
            
            X_clean, y_clean, sample_weights = self.noise_detector.get_clean_dataset(
                X, y, 
                remove_noisy=(handle_noise == 'remove'),
                weight_noisy=True
            )
            self.sample_weights = sample_weights
        else:
            X_clean, y_clean = X, y
            sample_weights = None
        
        print(f"\nTraining model with {len(X_clean)} samples...")
        self.model = self.base_model_class
        
        # Train with sample weights
        if sample_weights is not None:
            self.model.fit(X_clean, y_clean, sample_weight=sample_weights)
        else:
            self.model.fit(X_clean, y_clean)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        return self.model.predict_proba(X)


if __name__ == "__main__":
    # Test label noise detection
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    
    # Generate synthetic data
    X, y = make_classification(n_samples=200, n_features=20, n_classes=3, 
                               n_informative=10, random_state=42)
    
    # Artificially add noise
    noisy_fraction = 0.1
    n_noisy = int(len(y) * noisy_fraction)
    noisy_indices = np.random.choice(len(y), n_noisy, replace=False)
    y_noisy = y.copy()
    for idx in noisy_indices:
        # Flip to wrong label
        wrong_labels = [l for l in np.unique(y) if l != y[idx]]
        y_noisy[idx] = np.random.choice(wrong_labels)
    
    # Detect noise
    detector = LabelNoiseDetector(contamination_rate=0.15)
    scores, noisy_idx = detector.detect_noise(X, y_noisy)
    detector.print_noise_report(y_noisy)
    
    print(f"\nActual noisy indices: {sorted(noisy_indices)}")
    print(f"Detected noisy indices: {sorted(noisy_idx)}")
    print(f"Detection accuracy: {len(set(noisy_indices) & set(noisy_idx)) / len(noisy_indices) * 100:.1f}%")
