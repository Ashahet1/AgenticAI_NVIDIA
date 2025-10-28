# conversation_manager.py

class ConversationManager:
    """
    Manages multi-turn conversation state with smart adaptive questioning
    - Asks required fields first
    - Then asks MOST IMPORTANT optional fields
    - If still unclear, asks MORE optional fields
    - Never overwhelms user with all questions at once
    """
    
    def __init__(self):
        self.conversation_history = []
        self.collected_data = {}
        
        # REQUIRED fields (must have these)
        self.required_fields = [
            "exercise",
            "pain_location",
            "pain_timing"
        ]
        
        # OPTIONAL fields - TIERED by importance
        # Tier 1: Most critical for diagnosis (ask first if needed)
        self.optional_tier1 = [
            "pain_side",           # left vs right - critical for diagnosis
            "pain_intensity",      # severity matters
            "pain_type",           # sharp vs dull - mechanism clue
            "movement_phase",      # when in the rep - biomechanics
            "duration_since_onset" # acute vs chronic
        ]
        
        # Tier 2: Important but secondary (ask if Tier 1 didn't clarify)
        self.optional_tier2 = [
            "previous_injuries",
            "training_experience",
            "equipment",
            "self_treatment_actions",
            "improvement_since"
        ]
        
        # Tier 3: Nice to have (ask only if really stuck)
        self.optional_tier3 = [
            "surface_type",
            "environment",
            "repetition_scheme",
            "sleep_quality",
            "hydration_level",
            "training_frequency",
            "associated_symptoms",
            "availability_for_training",
            "preferred_exercise_types",
            "goal"
        ]
        
        # All optional fields combined
        self.optional_fields = self.optional_tier1 + self.optional_tier2 + self.optional_tier3
        
        # Track which questions we've already asked
        self.questions_asked = []
        
        # Track how many optional questions we've asked so far
        self.optional_questions_count = 0
        
        # Limits
        self.max_optional_tier1_questions = 2  # Ask max 2 from tier 1
        self.max_optional_tier2_questions = 2  # Then max 2 from tier 2
        self.max_optional_tier3_questions = 1  # Then max 1 from tier 3
    
    def add_message(self, role, content, agent_name=None):
        """
        Add a message to conversation history
        
        Args:
            role: "user" or "agent"
            content: message text
            agent_name: which agent sent it (if agent)
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": self._get_timestamp()
        }
        
        if agent_name:
            message["agent"] = agent_name
        
        self.conversation_history.append(message)
        return message
    
    def update_collected_data(self, new_data):
        """
        Merge new extracted data into collected data
        """
        self.collected_data.update(new_data)
    
    def get_missing_required_fields(self):
        """
        Check what required fields are still missing
        
        Returns:
            List of missing field names
        """
        missing = []
        for field in self.required_fields:
            value = self.collected_data.get(field, "")
            
            # Check if missing or has placeholder values
            if not value or value in ["unknown", "unspecified", "not specified"]:
                missing.append(field)
        
        return missing
    
    def get_missing_optional_fields(self, tier=None):
        """
        Check what optional fields are missing
        
        Args:
            tier: 1, 2, or 3 to get specific tier, None for all
            
        Returns:
            List of missing field names
        """
        if tier == 1:
            fields_to_check = self.optional_tier1
        elif tier == 2:
            fields_to_check = self.optional_tier2
        elif tier == 3:
            fields_to_check = self.optional_tier3
        else:
            fields_to_check = self.optional_fields
        
        missing = []
        for field in fields_to_check:
            value = self.collected_data.get(field, "")
            
            # Check if missing or placeholder
            if not value or value in ["unknown", "unspecified", "not specified"]:
                # Don't include if we've already asked about it
                if field not in self.questions_asked:
                    missing.append(field)
        
        return missing
    
    def needs_clarification(self):
        """
        Determine if we need to ask user for more info
        
        Returns:
            bool: True if missing required fields OR should ask optional fields
        """
        # First priority: missing required fields
        if len(self.get_missing_required_fields()) > 0:
            return True
        
        # Second priority: should we ask some optional questions?
        return self.should_ask_optional_questions()
    
    def should_ask_optional_questions(self):
        """
        Decide if we should ask optional questions
        
        Logic:
        - If we haven't asked any tier 1 questions yet, and there are missing tier 1 fields → YES
        - If we asked tier 1 but still need clarity, ask tier 2 → YES
        - If we asked tier 1 and 2, but still stuck, ask tier 3 → YES
        - Otherwise → NO
        
        Returns:
            bool: True if should ask optional questions
        """
        # Check tier 1
        tier1_missing = self.get_missing_optional_fields(tier=1)
        tier1_asked = sum(1 for q in self.questions_asked if q in self.optional_tier1)
        
        if tier1_missing and tier1_asked < self.max_optional_tier1_questions:
            return True
        
        # Check tier 2 (only if tier 1 is done)
        if tier1_asked >= self.max_optional_tier1_questions:
            tier2_missing = self.get_missing_optional_fields(tier=2)
            tier2_asked = sum(1 for q in self.questions_asked if q in self.optional_tier2)
            
            if tier2_missing and tier2_asked < self.max_optional_tier2_questions:
                return True
        
        # Check tier 3 (only if tier 1 and 2 are done)
        tier2_asked = sum(1 for q in self.questions_asked if q in self.optional_tier2)
        if (tier1_asked >= self.max_optional_tier1_questions and 
            tier2_asked >= self.max_optional_tier2_questions):
            
            tier3_missing = self.get_missing_optional_fields(tier=3)
            tier3_asked = sum(1 for q in self.questions_asked if q in self.optional_tier3)
            
            if tier3_missing and tier3_asked < self.max_optional_tier3_questions:
                return True
        
        return False
    
    def generate_clarifying_question(self):
        """
        Generate a natural clarifying question
        
        Priority:
        1. Required fields
        2. Tier 1 optional fields
        3. Tier 2 optional fields
        4. Tier 3 optional fields
        
        Returns:
            str: Question to ask user, or None if nothing to ask
        """
        # PRIORITY 1: Required fields
        missing_required = self.get_missing_required_fields()
        if missing_required:
            field = missing_required[0]
            question = self._get_question_for_field(field)
            self.questions_asked.append(field)
            return question
        
        # PRIORITY 2: Tier 1 optional fields
        tier1_missing = self.get_missing_optional_fields(tier=1)
        tier1_asked = sum(1 for q in self.questions_asked if q in self.optional_tier1)
        
        if tier1_missing and tier1_asked < self.max_optional_tier1_questions:
            field = tier1_missing[0]
            question = self._get_question_for_field(field)
            self.questions_asked.append(field)
            self.optional_questions_count += 1
            return question
        
        # PRIORITY 3: Tier 2 optional fields
        tier2_missing = self.get_missing_optional_fields(tier=2)
        tier2_asked = sum(1 for q in self.questions_asked if q in self.optional_tier2)
        
        if tier2_missing and tier2_asked < self.max_optional_tier2_questions:
            field = tier2_missing[0]
            question = self._get_question_for_field(field)
            self.questions_asked.append(field)
            self.optional_questions_count += 1
            return question
        
        # PRIORITY 4: Tier 3 optional fields (rarely get here)
        tier3_missing = self.get_missing_optional_fields(tier=3)
        tier3_asked = sum(1 for q in self.questions_asked if q in self.optional_tier3)
        
        if tier3_missing and tier3_asked < self.max_optional_tier3_questions:
            field = tier3_missing[0]
            question = self._get_question_for_field(field)
            self.questions_asked.append(field)
            self.optional_questions_count += 1
            return question
        
        # No more questions to ask
        return None
    
    def _get_question_for_field(self, field):
        """
        Generate natural question text for a specific field
        """
        questions = {
            # Required
            "exercise": "What exercise were you doing when this happened?",
            "pain_location": "Where exactly do you feel the pain? (e.g., right knee, left shoulder, lower back)",
            "pain_timing": "When exactly does the pain occur? (e.g., during the movement, after workout, next day)",
            
            # Optional Tier 1
            "pain_side": "Which side? Left or right?",
            "pain_intensity": "How intense is the pain? (mild, moderate, severe)",
            "pain_type": "How would you describe the pain? (sharp, dull, aching, burning, stabbing)",
            "movement_phase": "At what point in the movement does it hurt? (e.g., at the bottom, during the lift, at the top)",
            "duration_since_onset": "How long ago did this start? (e.g., today, 3 days ago, 2 weeks ago)",
            
            # Optional Tier 2
            "previous_injuries": "Have you had similar injuries or issues before in this area?",
            "training_experience": "How long have you been training with this exercise?",
            "equipment": "What equipment were you using? (e.g., barbell, dumbbells, machine)",
            "self_treatment_actions": "Have you tried anything to treat it? (ice, rest, stretching, etc.)",
            "improvement_since": "Has it gotten better, worse, or stayed the same since it started?",
            
            # Optional Tier 3
            "surface_type": "What surface were you training on? (gym floor, grass, concrete, etc.)",
            "environment": "Where were you training? (gym, home, outdoors)",
            "repetition_scheme": "How many reps/sets were you doing?",
            "sleep_quality": "How has your sleep been recently? (good, poor, average)",
            "hydration_level": "Have you been staying well hydrated?",
            "training_frequency": "How often do you train per week?",
            "associated_symptoms": "Any other symptoms? (swelling, stiffness, numbness, weakness)",
            "availability_for_training": "How much time can you dedicate to recovery exercises per day?",
            "preferred_exercise_types": "What types of exercises do you prefer for rehab? (stretching, strengthening, mobility)",
            "goal": "What's your main goal? (recover quickly, prevent re-injury, return to full performance)"
        }
        
        return questions.get(field, f"Can you tell me more about {field.replace('_', ' ')}?")
    
    def has_minimum_required_info(self):
        """
        Check if we have the absolute minimum to proceed
        (all required fields + at least some optional context)
        """
        # Must have all required fields
        if len(self.get_missing_required_fields()) > 0:
            return False
        
        # Nice to have at least 1-2 tier 1 optional fields
        tier1_filled = sum(1 for field in self.optional_tier1 
                          if self.collected_data.get(field) and 
                          self.collected_data.get(field) not in ["unknown", "unspecified"])
        
        # If we have required fields + at least 1 optional field, we can proceed
        return tier1_filled >= 1 or self.optional_questions_count >= 2
    
    def force_proceed(self):
        """
        User wants to proceed even without full info
        Stop asking questions
        """
        self.optional_questions_count = 999  # High number to stop asking
    
    def get_conversation_summary(self):
        """
        Get a text summary of the conversation so far
        """
        summary_parts = []
        
        for msg in self.conversation_history:
            if msg["role"] == "user":
                summary_parts.append(f"User: {msg['content']}")
            else:
                agent = msg.get("agent", "Agent")
                summary_parts.append(f"{agent}: {msg['content']}")
        
        return "\n".join(summary_parts)
    
    def get_state_snapshot(self):
        """
        Get complete state for saving/loading
        """
        return {
            "conversation_history": self.conversation_history,
            "collected_data": self.collected_data,
            "missing_required": self.get_missing_required_fields(),
            "missing_optional_tier1": self.get_missing_optional_fields(tier=1),
            "questions_asked": self.questions_asked,
            "optional_questions_count": self.optional_questions_count,
            "has_minimum_info": self.has_minimum_required_info()
        }
    
    def _get_timestamp(self):
        """Helper to get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()


# Test the ConversationManager
if __name__ == "__main__":
    print("Testing ConversationManager with tiered questioning...\n")
    
    cm = ConversationManager()
    
    # Simulate conversation flow
    print("=== ROUND 1: User gives vague input ===")
    cm.add_message("user", "My knee hurts")
    cm.update_collected_data({
        "exercise": "unknown",
        "pain_location": "knee",
        "pain_timing": "unknown"
    })
    
    print(f"Missing required: {cm.get_missing_required_fields()}")
    print(f"Needs clarification: {cm.needs_clarification()}")
    
    if cm.needs_clarification():
        q1 = cm.generate_clarifying_question()
        print(f"Question 1: {q1}\n")
        cm.add_message("agent", q1, "ParsingAgent")
    
    print("=== ROUND 2: User provides exercise ===")
    cm.add_message("user", "Squats")
    cm.update_collected_data({"exercise": "squat"})
    
    print(f"Missing required: {cm.get_missing_required_fields()}")
    
    if cm.needs_clarification():
        q2 = cm.generate_clarifying_question()
        print(f"Question 2: {q2}\n")
        cm.add_message("agent", q2, "ParsingAgent")
    
    print("=== ROUND 3: User provides timing ===")
    cm.add_message("user", "During the upward motion")
    cm.update_collected_data({"pain_timing": "during ascent"})
    
    print(f"Missing required: {cm.get_missing_required_fields()}")
    print(f"Should ask optional: {cm.should_ask_optional_questions()}")
    
    if cm.needs_clarification():
        q3 = cm.generate_clarifying_question()
        print(f"Question 3 (Tier 1 optional): {q3}\n")
        cm.add_message("agent", q3, "ParsingAgent")
    
    print("=== ROUND 4: User provides side ===")
    cm.add_message("user", "Right knee")
    cm.update_collected_data({"pain_side": "right"})
    
    if cm.needs_clarification():
        q4 = cm.generate_clarifying_question()
        print(f"Question 4 (Tier 1 optional): {q4}\n")
        cm.add_message("agent", q4, "ParsingAgent")
    
    print("=== ROUND 5: User provides intensity ===")
    cm.add_message("user", "Sharp pain, pretty severe")
    cm.update_collected_data({"pain_intensity": "severe", "pain_type": "sharp"})
    
    print(f"\n=== FINAL STATE ===")
    print(f"Has minimum info: {cm.has_minimum_required_info()}")
    print(f"Collected data: {cm.collected_data}")
    print(f"Questions asked: {cm.questions_asked}")
    print(f"Optional questions count: {cm.optional_questions_count}")
    
    print("\n✅ ConversationManager test complete!")
