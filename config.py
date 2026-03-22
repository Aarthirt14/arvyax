"""
Configuration for  Emotional Understanding System
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Environment Settings
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
API_PORT = int(os.getenv('API_PORT', 8000))
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 32))
RANDOM_SEED = int(os.getenv('RANDOM_SEED', 42))

# Emotional States
EMOTIONAL_STATES = [
    'calm', 'content', 'energized', 'peaceful', 'focused',
    'restless', 'anxious', 'overwhelmed', 'tired', 'neutral'
]

# Intensity Scale
INTENSITY_RANGE = (1, 5)

# Decision Actions
ACTIONS = [
    'box_breathing',      # Controlled breathing (4-4-4-4 pattern)
    'journaling',         # Expressive writing
    'grounding',          # 5-4-3-2-1 technique
    'deep_work',          # Focused work session
    'yoga',               # Gentle movement
    'sound_therapy',      # Music/ambient sounds
    'light_planning',     # Quick planning/organization
    'rest',               # Sleep/downtime
    'movement',           # Exercise/walking
    'pause',              # Micro-break
]

# Timing Options
TIMING_OPTIONS = [
    'now',                # Immediate
    'within_15_min',      # Next 15 minutes
    'later_today',        # Afternoon/evening
    'tonight',            # Before bed
    'tomorrow_morning',   # Next morning
]

# Ambience Types
AMBIENCE_TYPES = [
    'forest', 'ocean', 'rain', 'mountain', 'café'
]

# Time of Day Categories
TIME_OF_DAY = [
    'morning', 'afternoon', 'evening', 'night'
]

# Model Configuration
MODEL_CONFIG = {
    'text_embedding_dim': 100,
    'hidden_dim': 128,
    'dropout': 0.3,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 50,
    'random_state': 42,
    'test_size': 0.2,
}

# Feature Scaling
FEATURE_SCALING_PARAMS = {
    'sleep_hours': (0, 12),
    'energy_level': (1, 5),
    'stress_level': (1, 5),
    'duration_min': (0, 60),
}

# Uncertainty Thresholds
UNCERTAINTY_CONFIG = {
    'confidence_threshold': 0.6,  # Below this = uncertain flag
    'entropy_threshold': 0.8,     # Max entropy for confidence
}

# Decision Logic Weights
DECISION_WEIGHTS = {
    'state_importance': 0.40,
    'intensity_importance': 0.25,
    'stress_importance': 0.15,
    'energy_importance': 0.15,
    'sleep_importance': 0.05,
}

# Confidence Boosters/Reducers
CONFIDENCE_FACTORS = {
    'text_length': {'short': -0.15, 'medium': 0.0, 'long': 0.15},
    'data_completeness': {'missing_1': -0.1, 'missing_2+': -0.25},
    'prediction_agreement': {'high': 0.2, 'medium': 0.0, 'low': -0.2},
}
