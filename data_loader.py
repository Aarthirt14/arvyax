"""
Data loader for  Emotional Understanding System
Fetches data from CSVs or generates synthetic data for testing
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

class DataLoader:
    def __init__(self, data_dir='data'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
    def load_training_data(self, csv_path=None):
        """
        Load training data from CSV/Excel or Google Sheets export
        Expected columns: id, journal_text, ambience_type, duration_min, sleep_hours,
                         energy_level, stress_level, time_of_day, previous_day_mood,
                         face_emotion_hint, reflection_quality, emotional_state, intensity
        """
        try:
            if csv_path and Path(csv_path).exists():
                if csv_path.endswith('.xlsx') or csv_path.endswith('.xls'):
                    df = pd.read_excel(csv_path)
                else:
                    df = pd.read_csv(csv_path)
            else:
                # Try to load from standard location
                excel_path = self.data_dir.parent / 'Sample__reflective_dataset.xlsx'
                if excel_path.exists():
                    df = pd.read_excel(excel_path)
                else:
                    df = self.generate_synthetic_data(n_samples=500, seed=42)
            
            if df is None or df.empty:
                raise ValueError("Data loading resulted in empty dataset")
            
            return self.validate_and_clean_data(df)
        except Exception as e:
            print(f"Error loading training data: {e}")
            return self.generate_synthetic_data(n_samples=500, seed=42)
    
    def load_test_data(self, csv_path=None):
        """Load test data - note: may not have emotional_state and intensity"""
        if csv_path and Path(csv_path).exists():
            if csv_path.endswith('.xlsx') or csv_path.endswith('.xls'):
                df = pd.read_excel(csv_path)
            else:
                df = pd.read_csv(csv_path)
        else:
            # Try to load from standard location
            excel_path = self.data_dir.parent / '_test_inputs_120.xlsx'
            if excel_path.exists():
                df = pd.read_excel(excel_path)
            else:
                df = self.generate_synthetic_data(n_samples=100, seed=123, is_test=True)
        
        return self.validate_and_clean_data(df, is_test=True)
    
    def generate_synthetic_data(self, n_samples=500, seed=42, is_test=False):
        """Generate realistic synthetic data for training/testing"""
        np.random.seed(seed)
        
        # Sample data
        sample_reflections = [
            "I felt the trees surrounding me, grounded. Nothing else mattered.",
            "The ocean waves helped me clear my mind. Went back feeling hopeful.",
            "Rainy day reflection made me think about past mistakes. Still anxious.",
            "Mountain air was crisp. Energy back. Ready to focus.",
            "Café time helped. Little overwhelmed still but better.",
            "Ok",
            "Fine I guess",
            "Felt calm after forest walk",
            "Very stressed, needed a break",
            "Focused and energized today",
            "Confused about my goals lately",
            "Sleep-deprived, everything feels harder",
            "Morning was good, afternoon got tough",
            "Reflection felt pointless today",
            "",  # Missing reflection
        ]
        
        # Sleep hours needs special handling (list comprehension)
        sleep_hours = [np.clip(np.random.normal(6.5, 2), 2, 12) for _ in range(n_samples)]
        
        data = {
            'id': np.arange(n_samples),
            'journal_text': np.random.choice(sample_reflections, n_samples),
            'ambience_type': np.random.choice(['forest', 'ocean', 'rain', 'mountain', 'café'], n_samples),
            'duration_min': np.random.choice([10, 15, 20, 30, 45, 60], n_samples),
            'sleep_hours': sleep_hours,
            'energy_level': np.random.randint(1, 6, n_samples),
            'stress_level': np.random.randint(1, 6, n_samples),
            'time_of_day': np.random.choice(['morning', 'afternoon', 'evening', 'night'], n_samples),
            'previous_day_mood': np.random.choice(['positive', 'neutral', 'negative'], n_samples),
            'face_emotion_hint': np.random.choice(['happy', 'neutral', 'sad', 'tense', 'confused'], n_samples),
            'reflection_quality': np.random.choice(['low', 'medium', 'high'], n_samples),
        }
        
        df = pd.DataFrame(data)
        
        if not is_test:
            # Add labels for training
            df['emotional_state'] = df.apply(self._label_emotional_state, axis=1)
            df['intensity'] = df.apply(self._label_intensity, axis=1)
        
        return df
    
    def _label_emotional_state(self, row):
        """Generate emotional state based on features"""
        states = {
            (1, 2): 'calm',
            (2, 1): 'peaceful',
            (5, 1): 'focused',
            (5, 2): 'energized',
            (3, 3): 'neutral',
            (1, 5): 'anxious',
            (2, 5): 'overwhelmed',
            (5, 5): 'restless',
            (2, 2): 'content',
            (1, 1): 'tired',
        }
        
        stress_level = row['stress_level']
        energy_level = row['energy_level']
        
        # Simple heuristic
        if stress_level <= 2 and energy_level >= 4:
            return 'energized'
        elif stress_level <= 2 and energy_level <= 2:
            return 'peaceful'
        elif stress_level >= 4:
            return 'anxious'
        elif energy_level <= 2:
            return 'tired'
        else:
            return 'neutral'
    
    def _label_intensity(self, row):
        """Generate intensity based on features"""
        # Intensity is correlation of emotion magnitude
        stress_normalized = row['stress_level'] / 5.0
        energy_normalized = row['energy_level'] / 5.0
        
        # High stress or energy extremes = higher intensity
        avg_intensity = (stress_normalized + (1 - energy_normalized)) / 2.0
        intensity = int(np.clip(avg_intensity * 5, 1, 5))
        
        # Add some randomness
        intensity = np.clip(intensity + np.random.randint(-1, 2), 1, 5)
        return intensity
    
    def validate_and_clean_data(self, df, is_test=False):
        """Validate and clean data"""
        # Fill missing journal texts
        df['journal_text'] = df['journal_text'].fillna("No reflection provided")
        
        # Ensure all numeric columns are numeric
        numeric_cols = ['duration_min', 'sleep_hours', 'energy_level', 'stress_level']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].median())
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['id'], keep='first')
        
        return df.reset_index(drop=True)


def create_data_splits(df, test_size=0.2, random_state=42):
    """Split data into train/val sets"""
    from sklearn.model_selection import train_test_split
    
    train, val = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    return train, val


if __name__ == "__main__":
    loader = DataLoader()
    
    # Test loading
    train_df = loader.load_training_data()
    print(f"Training data shape: {train_df.shape}")
    print(f"Columns: {train_df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(train_df.head())
    
    test_df = loader.load_test_data()
    print(f"\nTest data shape: {test_df.shape}")
    print(f"\nTest columns: {test_df.columns.tolist()}")
