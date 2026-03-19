"""
Decision Engine for  Emotional Understanding System
Determines: What to do + When to do it
Based on: Predicted state, intensity, stress, energy, time of day
"""

import numpy as np
import pandas as pd
from config import ACTIONS, TIMING_OPTIONS


class DecisionEngine:
    """
    Core logic to decide:
    1. What action should user take?
    2. When should they take it?
    """
    
    def __init__(self):
        pass  # Methods are called as needed
        
    def decide(self, emotional_state, intensity, stress_level, energy_level, 
               time_of_day, sleep_hours, raw_features=None):
        """
        Main decision function
        
        Args:
            emotional_state: predicted emotional state (str)
            intensity: intensity level (1-5)
            stress_level: stress level (1-5)
            energy_level: energy level (1-5)
            time_of_day: time of day (morning/afternoon/evening/night)
            sleep_hours: hours of sleep
            raw_features: optional dict with additional context
        
        Returns:
            {
                'what_to_do': action,
                'when_to_do': timing,
                'rationale': explanation,
                'supportive_message': human-friendly message
            }
        """
        
        # Determine action
        action = self._decide_action(
            emotional_state, intensity, stress_level, energy_level
        )
        
        # Determine timing
        timing = self._decide_timing(
            emotional_state, intensity, time_of_day, energy_level, sleep_hours
        )
        
        # Build rationale
        rationale = self._build_rationale(
            emotional_state, intensity, stress_level, energy_level, action, timing
        )
        
        # Generate supportive message
        message = self._generate_supportive_message(
            emotional_state, intensity, action, timing
        )
        
        return {
            'what_to_do': action,
            'when_to_do': timing,
            'rationale': rationale,
            'supportive_message': message,
        }
    
    def _define_action_rules(self):
        """
        Decision tree for actions
        Maps: (state, intensity, stress, energy) -> action
        """
        rules = {}
        
        # High stress cases -> calming actions
        # (stress >= 4) -> breathing, grounding, rest
        
        # Low energy cases -> rest or light activities
        # (energy <= 2) -> rest, pause, yoga
        
        # High energy cases -> focused work
        # (energy >= 4, stress <= 2) -> deep_work, movement
        
        # Anxious states -> grounding techniques
        # 'anxious' or 'restless' -> box_breathing, grounding
        
        # Overwhelmed -> break it down
        # 'overwhelmed' -> light_planning, pause, grounding
        
        return rules
    
    def _decide_action(self, emotional_state, intensity, stress_level, energy_level):
        """
        Decide what action to take using business logic
        Priority: State > Stress > Energy
        """
        
        # Rule hierarchy
        
        # Rule 1: Overwhelmed/Anxious -> Immediate grounding
        if emotional_state in ['overwhelmed', 'anxious', 'restless']:
            if stress_level >= 4:
                return 'grounding'  # 5-4-3-2-1 technique
            elif intensity >= 4:
                return 'box_breathing'
            else:
                return 'pause'
        
        # Rule 2: Tired/Low energy -> Rest
        if emotional_state == 'tired' or energy_level <= 2:
            if stress_level >= 4:
                return 'rest'
            elif emotional_state == 'tired':
                return 'rest'
            else:
                return 'pause'
        
        # Rule 3: Focused/Energized + Low stress -> Deep work
        if emotional_state in ['focused', 'energized']:
            if stress_level <= 2:
                return 'deep_work'
            elif energy_level >= 4:
                return 'light_planning'
            else:
                return 'movement'
        
        # Rule 4: Calm/Peaceful + Low stress -> Journaling/Yoga
        if emotional_state in ['calm', 'peaceful', 'content']:
            if intensity <= 2:
                return 'journaling'
            elif stress_level >= 3:
                return 'yoga'
            else:
                return 'sound_therapy'
        
        # Rule 5: Neutral/Default cases
        if emotional_state in ['neutral', 'content']:
            if stress_level >= 4:
                return 'light_planning'  # Gain control
            elif energy_level <= 2:
                return 'rest'
            else:
                return 'movement'
        
        # Default fallback
        return 'pause'
    
    def _decide_timing(self, emotional_state, intensity, time_of_day, 
                      energy_level, sleep_hours):
        """
        Decide when to do the action
        """
        
        # Rule 1: If very stressed/anxious -> NOW (immediate intervention)
        if emotional_state in ['overwhelmed', 'anxious'] and intensity >= 4:
            return 'now'
        
        # Rule 2: If tired and it's evening/night -> tonight
        if emotional_state == 'tired':
            if time_of_day in ['evening', 'night']:
                return 'tonight'
            elif time_of_day == 'afternoon':
                return 'later_today'
            else:
                return 'tonight'
        
        # Rule 3: If low sleep hours and morning -> tomorrow_morning
        if sleep_hours <= 4 and time_of_day == 'morning':
            return 'rest'  # Actually should be "now" but emphasize rest
            return 'now'
        
        # Rule 4: If good energy in morning -> deep work NOW
        if time_of_day == 'morning' and energy_level >= 4:
            return 'now'
        
        # Rule 5: If afternoon slump -> later_today or tonight
        if time_of_day == 'afternoon':
            if energy_level <= 2:
                return 'later_today'
            elif intensity >= 3:
                return 'within_15_min'
            else:
                return 'later_today'
        
        # Rule 6: Evening -> prepare for night
        if time_of_day in ['evening', 'night']:
            if emotional_state in ['calm', 'peaceful']:
                return 'tonight'
            elif intensity >= 3:
                return 'within_15_min'
            else:
                return 'tonight'
        
        # Default
        return 'within_15_min'
    
    def _build_rationale(self, state, intensity, stress, energy, action, timing):
        """Build explanation for decision"""
        rationale = f"Based on emotional state '{state}' "
        rationale += f"(intensity: {intensity}/5), "
        rationale += f"stress level {stress}/5, "
        rationale += f"energy level {energy}/5: "
        rationale += f"recommended action is '{action}' "
        rationale += f"to be done ({timing})."
        return rationale
    
    def _generate_supportive_message(self, state, intensity, action, timing):
        """Generate human-friendly supportive message"""
        
        messages = {
            ('anxious', 'box_breathing', 'now'): 
                "You seem a bit restless right now. Let's slow things down with some controlled breathing. Try a 4-4-4-4 box breathing exercise right now.",
            
            ('overwhelmed', 'grounding', 'now'):
                "Things feel like a lot right now. Let's ground yourself with the 5-4-3-2-1 technique. Notice 5 things you see, 4 you can touch, 3 you hear, 2 you smell, 1 you taste.",
            
            ('tired', 'rest', 'tonight'):
                "Your energy is low. You need rest. Plan to get to bed soon and give yourself proper sleep tonight.",
            
            ('focused', 'deep_work', 'now'):
                "You're in a focused state! Use this window of clarity for your important work. Strike while you're in the zone.",
            
            ('calm', 'journaling', 'later_today'):
                "You're in a peaceful state. Try journaling to explore your thoughts more deeply when you have time later.",
            
            ('peaceful', 'yoga', 'within_15_min'):
                "You're feeling peaceful. A gentle yoga or stretch session in the next 15 minutes will help you maintain this calm.",
            
            ('neutral', 'movement', 'within_15_min'):
                "You're in a stable place. A bit of movement or a short walk in the next 15 minutes could help you feel more grounded.",
        }
        
        # Look for matching message
        key = (state, action, timing)
        if key in messages:
            return messages[key]
        
        # Partial matching
        for (s, a, t), msg in messages.items():
            if s == state and a == action:
                return msg
        
        # Generate generic message
        action_descriptions = {
            'box_breathing': 'a controlled breathing exercise',
            'journaling': 'expressive writing',
            'grounding': 'a grounding technique to reconnect with the present',
            'deep_work': 'focused work on your goals',
            'yoga': 'gentle stretching or yoga',
            'sound_therapy': 'calming music or nature sounds',
            'light_planning': 'organizing your tasks',
            'rest': 'rest and recovery',
            'movement': 'some gentle movement',
            'pause': 'a short break',
        }
        
        action_desc = action_descriptions.get(action, action)
        timing_descriptions = {
            'now': 'right now',
            'within_15_min': 'in the next 15 minutes',
            'later_today': 'later today',
            'tonight': 'tonight',
            'tomorrow_morning': 'tomorrow morning',
        }
        timing_desc = timing_descriptions.get(timing, timing)
        
        return f"You're feeling {state}. Try {action_desc} {timing_desc} to help yourself feel better."


def batch_decide(predictions_df, feature_data):
    """
    Apply decision engine to batch of predictions
    
    Args:
        predictions_df: DataFrame with predicted_state, predicted_intensity
        feature_data: DataFrame with stress_level, energy_level, time_of_day, sleep_hours
    
    Returns:
        DataFrame with what_to_do, when_to_do, supportive_message
    """
    engine = DecisionEngine()
    
    results = []
    for idx in predictions_df.index:
        decision = engine.decide(
            emotional_state=predictions_df.loc[idx, 'predicted_state'],
            intensity=int(predictions_df.loc[idx, 'predicted_intensity']),
            stress_level=int(feature_data.loc[idx, 'stress_level']),
            energy_level=int(feature_data.loc[idx, 'energy_level']),
            time_of_day=feature_data.loc[idx, 'time_of_day'],
            sleep_hours=feature_data.loc[idx, 'sleep_hours'],
        )
        
        results.append({
            'what_to_do': decision['what_to_do'],
            'when_to_do': decision['when_to_do'],
            'supportive_message': decision['supportive_message'],
        })
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    engine = DecisionEngine()
    
    # Test cases
    test_cases = [
        ('anxious', 4, 4, 2, 'afternoon', 5.5),
        ('tired', 2, 2, 1, 'evening', 3),
        ('focused', 4, 2, 5, 'morning', 7),
        ('calm', 2, 1, 3, 'afternoon', 6),
        ('overwhelmed', 5, 5, 2, 'morning', 4),
    ]
    
    print("Decision Engine Test Cases:")
    print("="*80)
    
    for state, intensity, stress, energy, time, sleep in test_cases:
        decision = engine.decide(
            emotional_state=state,
            intensity=intensity,
            stress_level=stress,
            energy_level=energy,
            time_of_day=time,
            sleep_hours=sleep,
        )
        
        print(f"\nState: {state}, Intensity: {intensity}, Stress: {stress}, Energy: {energy}")
        print(f"Time: {time}, Sleep: {sleep}h")
        print(f"→ Action: {decision['what_to_do']}")
        print(f"→ Timing: {decision['when_to_do']}")
        print(f"→ Message: {decision['supportive_message']}")
