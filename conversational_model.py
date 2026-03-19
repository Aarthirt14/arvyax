"""
Lightweight Conversational Model for 
Generates human-like supportive messages without heavy LLMs
Uses templates + context-aware interpolation
"""

import random
from typing import Dict, List, Tuple


class LightweightConversationalModel:
    """
    Lightweight SLM for generating natural supportive messages
    No external LLM calls, runs locally, ~1KB model size
    """
    
    def __init__(self):
        self.message_templates = self._load_templates()
        self.response_patterns = self._load_patterns()
    
    def _load_templates(self) -> Dict[str, List[str]]:
        """
        Template-based message generation
        Each state has multiple response templates with placeholders
        """
        return {
            'anxious': [
                "You seem a bit restless right now. Let's slow things down with some controlled breathing.",
                "Anxiety is showing up for you. Taking a moment to ground yourself can help.",
                "I notice you're feeling anxious. A grounding technique might help you feel more stable.",
                "Your nervous system is activated. Let's bring it back to center with some breathing.",
            ],
            'overwhelmed': [
                "Things feel like a lot right now. Breaking this down into smaller pieces will help.",
                "You're carrying a lot. Let's pause and return to the present moment.",
                "Overwhelming feelings are normal. A moment of stillness can help shift this.",
                "I sense you're stretched thin. Let's create some space to breathe.",
            ],
            'tired': [
                "Your energy is low. Rest and recovery are what you need right now.",
                "Tiredness is your body's signal. Honor it with proper rest.",
                "Sleep is medicine. Your body is asking for rest—give it what it needs.",
                "You need recharge time. Prioritize rest and see how you feel tomorrow.",
            ],
            'focused': [
                "You're in a clear, focused state! Use this window to tackle what matters.",
                "Great focus energy here. Strike while you're in the zone.",
                "Your mind is sharp right now. Channel this toward your goals.",
                "This is prime time for deep work. Make the most of this focused state.",
            ],
            'calm': [
                "You're in a peaceful place. Journaling can help explore this further.",
                "Calmness is a gift. Writing might deepen this insight.",
                "This quiet clarity is valuable. Reflection can help you learn more about yourself.",
                "You're grounded and present. This is fertile ground for self-understanding.",
            ],
            'peaceful': [
                "Beautiful peaceful energy. A gentle practice like yoga will complement this.",
                "You've found your peace. Gentle movement can help sustain this.",
                "This serenity is worth protecting. Yoga or stretching can help you hold onto it.",
                "You're in harmony with yourself. Let gentle movement support this state.",
            ],
            'energized': [
                "Your energy is high! Channel it into something meaningful.",
                "This vitality is powerful. Direct it toward your priorities.",
                "You're vibrating at a high frequency. Use this for productivity or creativity.",
                "This energy is a gift. Make something meaningful with it.",
            ],
            'restless': [
                "Restlessness wants to move. Try a walk or some light movement.",
                "Your body wants motion. Movement will help settle this restless feeling.",
                "Restless energy needs an outlet. Exercise or dancing could be perfect.",
                "This restlessness is asking for flow. Physical activity will help.",
            ],
            'neutral': [
                "You're in a neutral state—a good moment for reflection.",
                "Nothing overwhelming, nothing exciting—space for stillness and observation.",
                "This neutral ground is actually peaceful. Use it for self-check-in.",
                "You're balanced right now. This is a good moment to pause and reflect.",
            ],
            'content': [
                "Contentment is showing up. Savor this feeling and let it settle into your body.",
                "This quiet satisfaction is real. Take a moment to truly feel it.",
                "You're in a place of genuine satisfaction. This matters—recognize it.",
                "Contentment is often overlooked but so vital. Stay with this feeling.",
            ],
        }
    
    def _load_patterns(self) -> Dict[str, List[Tuple]]:
        """
        Context-aware response patterns
        (intensity, stress, energy) → response adjustments
        """
        return {
            'high_intensity': [
                " This feeling is intense—lean into your support system.",
                " The intensity you're feeling is real. Acknowledge it.",
                " Such strong feelings deserve attention. Take them seriously.",
            ],
            'high_stress_low_energy': [
                " Given the stress you're carrying, rest is not optional—it's necessary.",
                " Your stress level is high and energy is low. This calls for recovery time.",
                " This combination needs compassion. Be gentle with yourself.",
            ],
            'high_energy_high_stress': [
                " Your energy can help you handle this pressure—channel it wisely.",
                " High stress with high energy: be careful not to burn out.",
                " This is intense. Use your energy strategically, not frantically.",
            ],
            'low_stress_high_energy': [
                " This is an ideal state—low stress, high energy. Use it well.",
                " You've found the sweet spot. Make something meaningful happen.",
                " This is the best foundation for your goals. Strike now.",
            ],
        }
    
    def generate_message(self, state: str, intensity: int, stress: int, 
                        energy: int, time_of_day: str = None) -> str:
        """
        Generate conversational, supportive message
        
        Args:
            state: Emotional state
            intensity: 1-5 scale
            stress: 1-5 stress level
            energy: 1-5 energy level
            time_of_day: Optional context
        
        Returns:
            Natural, supportive message
        """
        # Get base template for state
        if state not in self.message_templates:
            state = 'neutral'
        
        base_message = random.choice(self.message_templates[state])
        
        # Add context-aware patterns
        context_pattern = self._get_context_pattern(intensity, stress, energy)
        if context_pattern:
            base_message += context_pattern
        
        # Add time-of-day context
        if time_of_day:
            time_context = self._get_time_context(time_of_day, state, energy)
            if time_context:
                base_message += " " + time_context
        
        # Add closing affirmation
        closing = self._get_closing(state, intensity)
        base_message += " " + closing
        
        return base_message
    
    def _get_context_pattern(self, intensity: int, stress: int, energy: int) -> str:
        """Get pattern based on signal combination"""
        if stress >= 4 and energy <= 2:
            return random.choice(self.response_patterns['high_stress_low_energy'])
        elif stress >= 4 and energy >= 4:
            return random.choice(self.response_patterns['high_energy_high_stress'])
        elif stress <= 2 and energy >= 4:
            return random.choice(self.response_patterns['low_stress_high_energy'])
        elif intensity >= 4:
            return random.choice(self.response_patterns['high_intensity'])
        return ""
    
    def _get_time_context(self, time_of_day: str, state: str, energy: int) -> str:
        """Add time-aware context"""
        time_contexts = {
            'morning': {
                'tired': "Start your day with rest and gentle movement.",
                'anxious': "Morning anxiety is common. Ground yourself before the day begins.",
                'energized': "Morning energy is powerful. Use it to set intentions.",
                'default': "Morning is a fresh start. Make it count.",
            },
            'afternoon': {
                'tired': "The afternoon slump is real. A short break or walk could help.",
                'overwhelmed': "Afternoon overwhelm often peaks. Step back and reassess.",
                'energized': "Afternoon energy surge! Ride this wave productively.",
                'default': "Afternoon check-in: how are you really doing?",
            },
            'evening': {
                'energized': "Evening energy—be careful not to stay wired into the night.",
                'overwhelmed': "Evening wind-down is important. Start transitioning now.",
                'tired': "Your body is asking for rest. Respect that signal.",
                'default': "Evening is for settling and reflecting.",
            },
            'night': {
                'tired': "Sleep is calling. Listen to your body.",
                'anxious': "Nighttime anxiety often amplifies. Practice calming techniques.",
                'energized': "Evening/night energy—wind down gradually so you can sleep.",
                'default': "Prepare your body and mind for rest.",
            },
        }
        
        time_dict = time_contexts.get(time_of_day, {})
        return time_dict.get(state, time_dict.get('default', ''))
    
    def _get_closing(self, state: str, intensity: int) -> str:
        """Get closing affirmation"""
        closings = {
            'anxious': "You've got this.",
            'overwhelmed': "One step at a time.",
            'tired': "Rest well.",
            'focused': "Make it matter.",
            'calm': "Stay with this.",
            'peaceful': "Breathe deeply.",
            'energized': "Channel it wisely.",
            'restless': "Let it flow.",
            'neutral': "You're doing okay.",
            'content': "Hold onto this.",
        }
        
        closing = closings.get(state, "You're doing okay.")
        
        # Adjust intensity
        if intensity >= 4:
            closing += " This matters."
        
        return closing


# Test the model
if __name__ == "__main__":
    model = LightweightConversationalModel()
    
    test_cases = [
        ('anxious', 4, 4, 2, 'afternoon'),
        ('tired', 2, 2, 1, 'evening'),
        ('focused', 4, 2, 5, 'morning'),
        ('overwhelmed', 5, 5, 2, 'morning'),
        ('calm', 2, 1, 3, 'afternoon'),
    ]
    
    print("Lightweight Conversational Model - Test Output:")
    print("=" * 80)
    
    for state, intensity, stress, energy, time in test_cases:
        message = model.generate_message(state, intensity, stress, energy, time)
        print(f"\nState: {state}, Intensity: {intensity}, Stress: {stress}, Energy: {energy}, Time: {time}")
        print(f"Message: {message}")
