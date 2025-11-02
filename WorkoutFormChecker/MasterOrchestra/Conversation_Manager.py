# conversation_manager.py

import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConversationManager:
    def __init__(self, llm_client=None):
        self.conversation_history = []
        self.collected_data = {}
        self.llm_client = llm_client  # ✅ For generating smart questions
        self.required_fields = ["exercise", "pain_location", "pain_timing"]
        
        self.optional_tier1 = ["pain_side", "pain_intensity", "pain_type", "movement_phase", "duration_since_onset"]
        self.optional_tier2 = ["previous_injuries", "training_experience", "equipment", "self_treatment_actions", "improvement_since"]
        self.optional_tier3 = ["surface_type", "environment", "repetition_scheme", "sleep_quality", "hydration_level", "training_frequency", "associated_symptoms"]
        # Shuffle optional tiers to randomize question order slightly
        self.optional_tier1 = random.sample(self.optional_tier1, len(self.optional_tier1))
        self.optional_tier2 = random.sample(self.optional_tier2, len(self.optional_tier2))
        self.optional_tier3 = random.sample(self.optional_tier3, len(self.optional_tier3))
        self.optional_fields = self.optional_tier1 + self.optional_tier2 + self.optional_tier3

        self.optional_fields = self.optional_tier1 + self.optional_tier2 + self.optional_tier3
        self.questions_asked = []
        self.question_examples = {
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
        self.max_optional_questions = random.randint(2, 3)  # ✅ Reduced from 3-4
        self.optional_questions_count = 0
        
        logger.info(f"💬 ConvMgr: Will ask {self.max_optional_questions} optional questions")
    
    def add_message(self, role, content, agent_name=None):
        message = {"role": role, "content": content}
        if agent_name:
            message["agent"] = agent_name
        self.conversation_history.append(message)
        return message
    
    def update_collected_data(self, new_data):
        for k, v in new_data.items():
            if isinstance(v, str):
                v = v.strip()
            if v and v not in ["unknown", "unspecified", "not specified"]:
                self.collected_data[k] = v
        logger.info(f"💬 ConvMgr: Updated data -> {list(self.collected_data.keys())}")
    
    def get_missing_required_fields(self):
        missing = []
        for field in self.required_fields:
            value = self.collected_data.get(field, "")
            cmp = value.strip().lower() if isinstance(value, str) else str(value).strip().lower()
            if not cmp or cmp in ["unknown", "unspecified", "not specified"]:
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
            cmp = value.strip().lower() if isinstance(value, str) else str(value).strip().lower()
            if (not cmp or cmp in ["unknown", "unspecified", "not specified"]) and field not in self.questions_asked:
                missing.append(field)
        return missing
    
    def needs_clarification(self):
        missing_req = len(self.get_missing_required_fields())
        should_ask_opt = self.should_ask_optional_questions()
        logger.info(f"💬 ConvMgr: needs_clarification? req={missing_req>0}, opt={should_ask_opt}")
        return missing_req > 0 or should_ask_opt

    def should_ask_optional_questions(self):
        if self.optional_questions_count >= self.max_optional_questions:
            logger.info(f"💬 ConvMgr: Reached optional limit ({self.optional_questions_count}/{self.max_optional_questions})")
            return False
        missing_optional = self.get_missing_optional_fields()
        return len(missing_optional) > 0
    
    def _build_few_shot_prompt(self, missing_fields):
        """You are a helpful assistant collecting information about a user's workout pain.\n"
            "Make your questions *conversational* and avoid repeating the same phrasing.\n"
            "Rephrase creatively while staying medically relevant.\n"""
        examples = "\n".join(
            f"{field}: {question}" for field, question in self.question_examples.items()
        )
        collected = "\n".join(
            f"{k}: {v}" for k, v in self.collected_data.items() if v
        )
        prompt = (
            "You are a helpful assistant collecting information about a user's workout pain.\n"
            "Here are example questions for each field:\n"
            f"{examples}\n\n"
            "Here is the information collected so far:\n"
            f"{collected if collected else '(none)'}\n\n"
            f"Please ask the next best, most natural question to fill in missing information. "
            f"Only ask about one missing field at a time. "
            f"Missing fields: {', '.join(missing_fields)}\n\n"
            f"Generate a conversational, natural question (max 20 words):"
        )
        return prompt
    
    def generate_clarifying_question(self):
        missing_required = self.get_missing_required_fields()
        missing_optional = self.get_missing_optional_fields()
        missing_fields = missing_required or (
            missing_optional if self.optional_questions_count < self.max_optional_questions else []
        )

        if not missing_fields:
            logger.info("💬 ConvMgr: No clarifying questions to ask")
            return None

        # ✅ Use LLM to generate natural question
        if self.llm_client:
            try:
                prompt = self._build_few_shot_prompt(missing_fields)
                 # 🧠 Simple semantic inference before asking
                # 🧠 Smart field skip logic
                utterance = " ".join(v.lower() for v in self.collected_data.values() if isinstance(v, str))

                # Skip redundant pain type/intensity questions
                pain_descriptors = ["sharp", "dull", "burning", "aching", "tingling"]
                intensity_descriptors = ["mild", "moderate", "severe"]

                if any(desc in utterance for desc in pain_descriptors):
                    if "pain_type" in missing_fields:
                        missing_fields.remove("pain_type")

                if any(desc in utterance for desc in intensity_descriptors):
                    if "pain_intensity" in missing_fields:
                        missing_fields.remove("pain_intensity")

                # Skip side question if direction already known
                if "left" in utterance or "right" in utterance or "both" in utterance:
                    if "pain_side" in missing_fields:
                        missing_fields.remove("pain_side")


                logger.info("💬 ConvMgr: Calling LLM for question...")
                question = self.llm_client.call_llm(prompt, max_tokens=64)
                
                if not question:
                    logger.warning("⚠️ ConvMgr: LLM returned empty. Falling back.")
                    return self._get_question_for_field(missing_fields[0])
                
                self.questions_asked.append(missing_fields[0])
                if not missing_required:
                    self.optional_questions_count += 1
                
                logger.info(f"✅ ConvMgr: Generated question: {question[:50]}")
                return question.strip()
                
            except Exception as e:
                logger.error(f"❌ ConvMgr: LLM call FAILED: {e}")
                return self._get_question_for_field(missing_fields[0])
        else:
            # Fallback to hardcoded
            field = missing_fields[0]
            question = self._get_question_for_field(field)
            self.questions_asked.append(field)
            if not missing_required:
                self.optional_questions_count += 1
            logger.info(f"💬 ConvMgr: (Fallback) Asking -> {field}")
            return question
        
    def _get_question_for_field(self, field):
        """Fallback hardcoded questions"""
        return self.question_examples.get(field, f"Can you tell me more about {field.replace('_', ' ')}?")
    
    def has_minimum_required_info(self):
        if len(self.get_missing_required_fields()) > 0:
            return False
        optional_filled = sum(
            1 for field in self.optional_fields
            if (self.collected_data.get(field) and 
                str(self.collected_data.get(field)).strip().lower() not in ["unknown", "unspecified"])
        )
        return optional_filled >= 2 or self.optional_questions_count >= self.max_optional_questions
    
    def force_proceed(self):
        self.optional_questions_count = 999

# ====================== TEST SECTION ======================
if __name__ == "__main__":
    print("🧪 Testing ConversationManager...\n")

    # 1️⃣ Fake LLM client for testing (simulates call_llm)
    class FakeLLM:
        def call_llm(self, prompt, max_tokens=64):
            print("\n🧠 [FakeLLM] Prompt received:")
            print(prompt[:300] + "...\n")
            return "What exercise were you performing when the pain started?"

    # 2️⃣ Create instance
    cm = ConversationManager(llm_client=FakeLLM())

    # 3️⃣ Simulate adding some partial user data
    cm.update_collected_data({
        "exercise": "",
        "pain_location": "right knee",
        "pain_timing": ""
    })

    print("\n➡️ Missing required fields:", cm.get_missing_required_fields())
    print("➡️ Needs clarification?", cm.needs_clarification())

    # 4️⃣ Generate a clarifying question
    question = cm.generate_clarifying_question()
    print("\n💬 Clarifying question generated:", question)

    # 5️⃣ Add more data and test again
    cm.update_collected_data({
        "exercise": "squat",
        "pain_timing": "during ascent"
    })
    print("\n➡️ Missing required fields (after update):", cm.get_missing_required_fields())
    print("➡️ Should ask optional questions?", cm.should_ask_optional_questions())

    # 6️⃣ Simulate asking optional questions
    for i in range(3):
        q = cm.generate_clarifying_question()
        print(f"Optional Q{i+1}: {q}")

    print("\n✅ Test complete. Conversation state summary:")
 
