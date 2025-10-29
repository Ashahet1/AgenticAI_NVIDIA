# simple_extractor.py
# Simple regex-based extraction BEFORE running ParsingAgent

import re

class SimpleExtractor:
    """
    Extract obvious information from user message using simple patterns
    This runs BEFORE ParsingAgent to catch 90% of cases
    """
    
    def __init__(self):
        # Common exercises
        self.exercises = [
            'squat', 'squats', 'deadlift', 'deadlifts', 'bench press', 'bench',
            'overhead press', 'ohp', 'pull up', 'pullup', 'pull-up', 'pullups',
            'chin up', 'chinup', 'chin-up', 'chinups', 'row', 'rows', 'barbell row',
            'lunge', 'lunges', 'leg press', 'leg curl', 'leg extension',
            'plank', 'push up', 'pushup', 'push-up', 'pushups',
            'bicep curl', 'biceps curl', 'tricep extension', 'lat pulldown',
            'dip', 'dips', 'clean', 'snatch', 'running', 'jogging', 'sprinting'
        ]
        
        # Body parts
        self.body_parts = {
            'knee': 'knee', 'knees': 'knee',
            'shoulder': 'shoulder', 'shoulders': 'shoulder',
            'back': 'back', 'lower back': 'lower back', 'upper back': 'upper back',
            'elbow': 'elbow', 'elbows': 'elbow',
            'wrist': 'wrist', 'wrists': 'wrist',
            'hip': 'hip', 'hips': 'hip',
            'ankle': 'ankle', 'ankles': 'ankle',
            'neck': 'neck',
            'chest': 'chest',
            'hamstring': 'hamstring', 'hamstrings': 'hamstring',
            'quad': 'quadriceps', 'quads': 'quadriceps', 'quadriceps': 'quadriceps',
            'calf': 'calf', 'calves': 'calf',
            'glute': 'glute', 'glutes': 'glute'
        }
        
        # Timing patterns
        self.timing_patterns = {
            'during': ['during', 'while', 'when', 'as i'],
            'after': ['after', 'following', 'post', 'later', 'next day', 'day after'],
            'before': ['before', 'prior to']
        }
    
    def extract(self, text):
        """
        Extract exercise, pain_location, pain_timing from text
        Returns dict with extracted fields (or "unknown" if not found)
        """
        text_lower = text.lower()
        
        result = {
            'exercise': 'unknown',
            'pain_location': 'unknown',
            'pain_timing': 'unknown',
            'pain_side': 'unknown'
        }
        
        # Extract exercise
        for exercise in self.exercises:
            if exercise in text_lower:
                result['exercise'] = exercise
                print(f"   [SimpleExtractor] Found exercise: {exercise}")
                break
        
        # Extract pain location
        for body_part, normalized in self.body_parts.items():
            if body_part in text_lower:
                result['pain_location'] = normalized
                print(f"   [SimpleExtractor] Found pain location: {normalized}")
                break
        
        # Extract side (left/right)
        if 'right' in text_lower:
            result['pain_side'] = 'right'
            result['pain_location'] = f"right {result['pain_location']}"
            print(f"   [SimpleExtractor] Found side: right")
        elif 'left' in text_lower:
            result['pain_side'] = 'left'
            result['pain_location'] = f"left {result['pain_location']}"
            print(f"   [SimpleExtractor] Found side: left")
        
        # Extract timing
        for timing_key, patterns in self.timing_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    # Get more context
                    if 'bottom' in text_lower or 'descent' in text_lower or 'lowering' in text_lower or 'coming up' in text_lower:
                        result['pain_timing'] = f"during movement (specific phase)"
                    elif timing_key == 'during':
                        result['pain_timing'] = 'during the movement'
                    elif timing_key == 'after':
                        result['pain_timing'] = 'after the workout'
                    else:
                        result['pain_timing'] = timing_key
                    print(f"   [SimpleExtractor] Found timing: {result['pain_timing']}")
                    break
            if result['pain_timing'] != 'unknown':
                break
        
        return result