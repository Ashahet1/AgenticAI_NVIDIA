# conversation_manager.py

import random

class ConversationManager:
    def __init__(self):
        self.conversation_history = []
        self.collected_data = {}
        
        self.required_fields = ["exercise", "pain_location", "pain_timing"]
        
        self.optional_tier1 = ["pain_side", "pain_intensity", "pain_type", "movement_phase", "duration_since_onset"]
        self.optional_tier2 = ["previous_injuries", "training_experience", "equipment", "self_treatment_actions", "improvement_since"]
        self.optional_tier3 = ["surface_type", "environment", "repetition_scheme", "sleep_quality", "hydration_level", "training_frequency", "associated_symptoms"]
        
        self.optional_fields = self.optional_tier1 + self.optional_tier2 + self.optional_tier3
        self.questions_asked = []
        
        self.max_optional_questions = random.randint(3, 4)
        self.optional_questions_count = 0
        
        print(f"💬 ConvMgr: Will ask {self.max_optional_questions} optional questions")
    
    def add_message(self, role, content, agent_name=None):
        message = {"role": role, "content": content, "timestamp": self._get_timestamp()}
        if agent_name:
            message["agent"] = agent_name
        self.conversation_history.append(message)
        return message
    
    def update_collected_data(self, new_data):
        self.collected_data.update(new_data)
        print(f"💬 ConvMgr: Updated data -> {list(self.collected_data.keys())}")
    
    def get_missing_required_fields(self):
        missing = []
        for field in self.required_fields:
            value = self.collected_data.get(field, "")
            if not value or value in ["unknown", "unspecified", "not specified"]:
                missing.append(field)
        return missing
    
    def get_missing_optional_fields(self, tier=None):
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
            if not value or value in ["unknown", "unspecified", "not specified"]:
                if field not in self.questions_asked:
                    missing.append(field)
        return missing
    
    def needs_clarification(self):
        missing_req = len(self.get_missing_required_fields())
        should_ask_opt = self.should_ask_optional_questions()
        print(f"💬 ConvMgr: needs_clarification? req={missing_req>0}, opt={should_ask_opt}")
        return missing_req > 0 or should_ask_opt
    
    def should_ask_optional_questions(self):
        if self.optional_questions_count >= self.max_optional_questions:
            print(f"💬 ConvMgr: Reached optional limit ({self.optional_questions_count}/{self.max_optional_questions})")
            return False
        missing_optional = self.get_missing_optional_fields()
        result = len(missing_optional) > 0
        print(f"💬 ConvMgr: should_ask_optional? {result} (asked {self.optional_questions_count}/{self.max_optional_questions})")
        return result
    
    def generate_clarifying_question(self):
        # Required first
        missing_required = self.get_missing_required_fields()
        if missing_required:
            field = missing_required[0]
            question = self._get_question_for_field(field)
            self.questions_asked.append(field)
            print(f"💬 ConvMgr: Asking required -> {field}")
            return question
        
        # Then optional
        missing_optional = self.get_missing_optional_fields()
        if missing_optional and self.optional_questions_count < self.max_optional_questions:
            field = random.choice(missing_optional)
            question = self._get_question_for_field(field)
            self.questions_asked.append(field)
            self.optional_questions_count += 1
            print(f"💬 ConvMgr: Asking optional {self.optional_questions_count}/{self.max_optional_questions} -> {field}")
            return question
        
        return None
    
    def _get_question_for_field(self, field):
        questions = {
            "exercise": "What exercise were you doing?",
            "pain_location": "Where exactly do you feel the pain?",
            "pain_timing": "When does the pain occur?",
            "pain_side": "Which side? Left or right?",
            "pain_intensity": "How intense is the pain? (mild, moderate, severe)",
            "pain_type": "How would you describe the pain? (sharp, dull, aching, burning)",
            "movement_phase": "At what point in the movement does it hurt?",
            "duration_since_onset": "How long ago did this start?",
            "previous_injuries": "Have you had similar issues before?",
            "training_experience": "How long have you been training with this exercise?",
            "equipment": "What equipment were you using?",
            "self_treatment_actions": "Have you tried anything to treat it?",
            "improvement_since": "Has it gotten better, worse, or stayed the same?",
            "surface_type": "What surface were you training on?",
            "environment": "Where were you training?",
            "repetition_scheme": "How many reps/sets were you doing?",
            "sleep_quality": "How has your sleep been recently?",
            "hydration_level": "Have you been staying well hydrated?",
            "training_frequency": "How often do you train per week?",
            "associated_symptoms": "Any other symptoms? (swelling, stiffness, numbness)"
        }
        return questions.get(field, f"Can you tell me more about {field.replace('_', ' ')}?")
    
    def has_minimum_required_info(self):
        if len(self.get_missing_required_fields()) > 0:
            return False
        optional_filled = sum(1 for field in self.optional_fields 
                             if self.collected_data.get(field) and 
                             self.collected_data.get(field) not in ["unknown", "unspecified"])
        return optional_filled >= 2 or self.optional_questions_count >= self.max_optional_questions
    
    def force_proceed(self):
        self.optional_questions_count = 999
    
    def get_conversation_summary(self):
        summary_parts = []
        for msg in self.conversation_history:
            if msg["role"] == "user":
                summary_parts.append(f"User: {msg['content']}")
            else:
                agent = msg.get("agent", "Agent")
                summary_parts.append(f"{agent}: {msg['content']}")
        return "\n".join(summary_parts)
    
    def get_state_snapshot(self):
        return {
            "conversation_history": self.conversation_history,
            "collected_data": self.collected_data,
            "missing_required": self.get_missing_required_fields(),
            "questions_asked": self.questions_asked,
            "optional_questions_count": self.optional_questions_count
        }
    
    def _get_timestamp(self):
        from datetime import datetime
        return datetime.now().isoformat()
