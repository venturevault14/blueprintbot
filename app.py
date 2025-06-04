from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Request
import os
import json
import logging
import re
from datetime import datetime
from typing import List, Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import Pinecone (make it optional for Vercel)
try:
    from pinecone import Pinecone
    PINECONE_AVAILABLE = True
    logger.info("✅ Pinecone client available")
except ImportError:
    PINECONE_AVAILABLE = False
    logger.warning("⚠️ Pinecone not available - continuing without it")

# ========================================
# Configuration from Environment Variables
# ========================================

class Config:
    def __init__(self):
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.OPENAI_API_KEY_2 = os.getenv("OPENAI_API_KEY_2")  # For treatment/citations
        self.PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        self.PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east1-gcp")
        self.ASSISTANT_ID = os.getenv("ASSISTANT_ID", "asst_pAhSF6XJsj60efD9GEVdEG5n")
        self.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
        self.INDEX_NAME = os.getenv("INDEX_NAME", "triage-index")
        self.TREATMENT_INDEX_NAME = os.getenv("TREATMENT_INDEX_NAME", "triage-index-treatment")
        self.MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
        self.MIN_SYMPTOMS_FOR_PINECONE = int(os.getenv("MIN_SYMPTOMS_FOR_PINECONE", "3"))
        self.MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "3"))
        self.NIGERIA_EMERGENCY_HOTLINE = os.getenv("EMERGENCY_HOTLINE", "112")
        self.PINECONE_SCORE_THRESHOLD = float(os.getenv("PINECONE_SCORE_THRESHOLD", "0.88"))
        self.PORT = int(os.getenv("PORT", "8000"))

    def validate(self):
        """Validate required environment variables"""
        required_vars = ["OPENAI_API_KEY"]
        missing = [var for var in required_vars if not getattr(self, var)]
        if missing:
            raise ValueError(f"Missing required environment variables: {missing}")

config = Config()

# Validate configuration
try:
    config.validate()
    logger.info("Configuration validated successfully")
except ValueError as e:
    logger.error(f"Configuration error: {e}")
    raise

# ========================================
# Global Constants
# ========================================

RED_FLAGS = [
    "bullet wound", "gunshot", "profuse bleeding", "crushing chest pain",
    "sudden shortness of breath", "loss of consciousness", "slurred speech", "seizure",
    "head trauma", "neck trauma", "high fever with stiff neck", "uncontrolled vomiting",
    "severe allergic reaction", "anaphylaxis", "difficulty breathing", "persistent cough with blood",
    "severe abdominal pain", "sudden vision loss", "chest tightness with sweating",
    "blood in urine", "inability to pass urine", "sharp abdominal pain", "intermenstrual bleeding"
]

# ========================================
# FastAPI App
# ========================================

app = FastAPI(
    title="Medical Triage Assistant API",
    description="AI-powered medical triage assistant for symptom assessment, condition suggestions, and citations",
    version="1.1.2"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================================
# Client Management (Singleton Pattern)
# ========================================

class ClientManager:
    def __init__(self):
        self._openai_client = None
        self._openai_client_2 = None  # For treatment/citations
        self._pinecone_client = None
        self._pinecone_index = None
        self._treatment_index = None

    async def get_openai_client(self):
        """Get OpenAI client - always create fresh for serverless"""
        try:
            logger.info("🔑 Creating fresh OpenAI client for serverless...")
            client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
            logger.info("🧪 Testing OpenAI connection...")
            await client.models.list()
            logger.info("✅ OpenAI client created and tested successfully")
            return client
        except Exception as e:
            logger.error(f"❌ Failed to create OpenAI client: {e}")
            raise HTTPException(status_code=503, detail=f"OpenAI connection failed: {str(e)}")

    async def get_openai_client_2(self):
        """Get secondary OpenAI client for treatment & citations processing"""
        try:
            api_key = config.OPENAI_API_KEY_2 or config.OPENAI_API_KEY
            logger.info("🔑 Creating fresh OpenAI client 2 for treatment/citations processing...")
            client = AsyncOpenAI(api_key=api_key)
            await client.models.list()
            logger.info("✅ OpenAI client 2 created and tested successfully")
            return client
        except Exception as e:
            logger.error(f"❌ Failed to create OpenAI client 2: {e}")
            raise HTTPException(status_code=503, detail=f"Treatment/citations OpenAI connection failed: {str(e)}")

    def get_pinecone_index(self):
        """Get Pinecone index - cache for performance"""
        if not PINECONE_AVAILABLE or not config.PINECONE_API_KEY:
            return None
        if self._pinecone_index is None:
            try:
                logger.info("🔍 Initializing Pinecone client...")
                self._pinecone_client = Pinecone(api_key=config.PINECONE_API_KEY)
                logger.info(f"🔗 Connecting to Pinecone index: {config.INDEX_NAME}")
                self._pinecone_index = self._pinecone_client.Index(name=config.INDEX_NAME)
                stats = self._pinecone_index.describe_index_stats()
                logger.info(f"✅ Pinecone connected successfully - {stats.total_vector_count} vectors")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Pinecone: {e}")
                return None
        return self._pinecone_index

    def get_treatment_index(self):
        """Get Pinecone treatment index - cache for performance"""
        if not PINECONE_AVAILABLE or not config.PINECONE_API_KEY:
            return None
        if self._treatment_index is None:
            try:
                if self._pinecone_client is None:
                    logger.info("🔍 Initializing Pinecone client for treatment index...")
                    self._pinecone_client = Pinecone(api_key=config.PINECONE_API_KEY)
                logger.info(f"🔗 Connecting to treatment index: {config.TREATMENT_INDEX_NAME}")
                self._treatment_index = self._pinecone_client.Index(name=config.TREATMENT_INDEX_NAME)
                stats = self._treatment_index.describe_index_stats()
                logger.info(f"✅ Treatment index connected successfully - {stats.total_vector_count} vectors")
            except Exception as e:
                logger.error(f"❌ Failed to initialize treatment index: {e}")
                return None
        return self._treatment_index

client_manager = ClientManager()

# ========================================
# Pydantic Models
# ========================================

class TriageRequest(BaseModel):
    description: str = Field(..., min_length=1, max_length=1000)
    thread_id: Optional[str] = Field(None)

class ConditionInfo(BaseModel):
    name: str
    description: str
    file_citation: str

class TriageInfo(BaseModel):
    type: str
    location: str

class TriageResponse(BaseModel):
    text: str
    possible_conditions: List[ConditionInfo] = []
    safety_measures: List[str] = []
    disease_names: List[str] = []
    citations: List[str] = []
    triage: TriageInfo
    send_sos: bool = False
    follow_up_questions: List[str] = []
    thread_id: str
    symptoms_count: int = 0
    should_query_pinecone: bool = False

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    services: Dict[str, str]

# ========================================
# Utility Functions
# ========================================

def extract_symptoms_fallback(text: str) -> List[str]:
    """Fallback keyword-based symptom extraction"""
    text_lower = text.lower()
    symptoms = []
    symptom_patterns = {
        "headache": ["headache", "head pain", "migraine"],
        "nausea": ["nausea", "nauseous", "feeling sick", "sick to stomach"],
        "fever": ["fever", "temperature", "hot", "feverish"],
        "cough": ["cough", "coughing"],
        "sore throat": ["sore throat", "throat pain", "throat hurts"],
        "stomach pain": ["stomach pain", "stomach ache", "abdominal pain", "belly pain"],
        "diarrhea": ["diarrhea", "loose stool", "watery stool"],
        "constipation": ["constipation", "not stooling", "can't poop", "no bowel movement"],
        "vomiting": ["vomiting", "throwing up", "vomit"],
        "dizziness": ["dizzy", "dizziness", "lightheaded"],
        "fatigue": ["tired", "fatigue", "exhausted", "weak"],
        "chest pain": ["chest pain", "chest hurts"],
        "shortness of breath": ["shortness of breath", "hard to breathe", "can't breathe"],
        "back pain": ["back pain", "backache"],
        "joint pain": ["joint pain", "joints hurt"],
        "runny nose": ["runny nose", "stuffy nose", "congestion"],
        "sour taste": ["sour taste", "bitter taste", "metallic taste"]
    }
    for symptom, patterns in symptom_patterns.items():
        if any(pattern in text_lower for pattern in patterns):
            symptoms.append(symptom)
    return symptoms

async def validate_thread(thread_id: str, client=None) -> bool:
    """Check if an OpenAI thread ID is valid - with retry logic"""
    if not thread_id or not thread_id.strip():
        return False
    if client is None:
        try:
            client = await client_manager.get_openai_client()
        except Exception as e:
            logger.error(f"❌ Could not get OpenAI client for thread validation: {e}")
            return False
    for attempt in range(2):
        try:
            await client.beta.threads.retrieve(thread_id=thread_id.strip())
            logger.info(f"✅ Thread {thread_id} validated successfully (attempt {attempt + 1})")
            return True
        except Exception as e:
            logger.error(f"❌ Thread validation attempt {attempt + 1} failed for {thread_id}: {e}")
            if attempt == 0:
                import asyncio
                await asyncio.sleep(0.5)
    return False

async def get_thread_context(thread_id: str, client=None) -> Dict:
    """Retrieve and analyze thread context"""
    try:
        if client is None:
            client = await client_manager.get_openai_client()
        logger.info(f"📚 Retrieving context for thread: {thread_id}")
        messages = await client.beta.threads.messages.list(thread_id=thread_id, order='asc', limit=100)
        user_messages, all_symptoms, max_severity = [], [], 0
        logger.info(f"📊 Found {len(messages.data)} total messages in thread")
        for i, msg in enumerate(messages.data):
            if msg.role == "user":
                content = ""
                if msg.content and hasattr(msg.content[0], "text"):
                    content = msg.content[0].text.value
                if content:
                    user_messages.append(content)
                    logger.info(f"👤 User message {i+1}: '{content[:50]}...'")
                    try:
                        symptom_data = await extract_symptoms_comprehensive(content, client)
                        extracted_symptoms = symptom_data["symptoms"]
                        severity = symptom_data["severity"]
                        logger.info(f"🩺 Extracted from message {i+1}: {extracted_symptoms} (severity: {severity})")
                        all_symptoms.extend(extracted_symptoms)
                        max_severity = max(max_severity, severity)
                    except Exception as e:
                        logger.error(f"❌ Failed to extract symptoms from message {i+1}: {e}")
        unique_symptoms = list(dict.fromkeys([s.lower().strip() for s in all_symptoms if s.strip()]))
        context = {"user_messages": user_messages, "all_symptoms": unique_symptoms, "max_severity": max_severity}
        logger.info(f"📋 Thread context summary:")
        logger.info(f"   - User messages: {len(user_messages)}")
        logger.info(f"   - Raw symptoms extracted: {all_symptoms}")
        logger.info(f"   - Unique symptoms: {unique_symptoms}")
        logger.info(f"   - Max severity: {max_severity}")
        return context
    except Exception as e:
        logger.error(f"❌ Error retrieving thread context for {thread_id}: {e}")
        return {"user_messages": [], "all_symptoms": [], "max_severity": 0}

async def extract_symptoms_comprehensive(description: str, client=None) -> Dict:
    """Extract symptoms, duration, and severity from description"""
    try:
        if client is None:
            client = await client_manager.get_openai_client()
        description_lower = description.lower()
        logger.info(f"🔍 Extracting symptoms from: '{description[:100]}...'")
        severity = 0
        descriptive_severity = {
            "excruciating": 10, "unbearable": 10, "extremely painful": 10,
            "severe": 8, "very painful": 8, "crushing": 8, "sharp": 8,
            "painful": 6, "moderate": 6, "mild": 4
        }
        for term, score in descriptive_severity.items():
            if term in description_lower:
                severity = score
                logger.info(f"📊 Found severity indicator '{term}': {score}")
                break
        pain_scale_match = re.search(r"pain\s+(\d+)/10", description_lower)
        if pain_scale_match:
            severity = int(pain_scale_match.group(1))
            logger.info(f"📊 Found numeric pain scale: {severity}/10")
        prompt = (
            "Extract specific medical symptoms from this text. Return ONLY a JSON array of symptom strings. "
            "Be precise and use standard medical terminology. Focus on physical symptoms only. "
            "Do NOT wrap in markdown code blocks. Return raw JSON only. "
            "Example: [\"chest pain\", \"shortness of breath\", \"nausea\"]\n\n"
            f"Text: \"{description}\""
        )
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt}, {"role": "user", "content": "Extract symptoms now."}],
            temperature=0,
            max_tokens=200
        )
        raw_response = response.choices[0].message.content.strip()
        logger.info(f"🤖 GPT raw response: {raw_response[:100]}...")
        if raw_response.startswith("```json"):
            json_start = raw_response.find('[')
            json_end = raw_response.rfind(']') + 1
            if json_start != -1 and json_end > json_start:
                json_str = raw_response[json_start:json_end]
                logger.info(f"🔧 Extracted JSON from markdown: {json_str}")
            else:
                logger.warning("⚠️ Could not extract JSON from markdown block")
                json_str = "[]"
        elif raw_response.startswith("```"):
            lines = raw_response.split('\n')
            json_lines = [line for line in lines if line.strip() and not line.startswith('```')]
            json_str = '\n'.join(json_lines)
            logger.info(f"🔧 Extracted JSON from code block: {json_str}")
        else:
            json_str = raw_response
        try:
            symptoms = json.loads(json_str)
            if not isinstance(symptoms, list):
                logger.warning(f"⚠️ GPT returned non-list: {symptoms}")
                symptoms = []
            else:
                logger.info(f"✅ Successfully parsed {len(symptoms)} symptoms")
        except json.JSONDecodeError as e:
            logger.warning(f"⚠️ JSON parsing failed: {e}")
            logger.warning(f"⚠️ Raw content: {raw_response}")
            logger.info("🔄 Falling back to keyword-based symptom extraction")
            symptoms = extract_symptoms_fallback(description)
            logger.info(f"🔄 Fallback extracted: {symptoms}")
        unique_symptoms = []
        for symptom in symptoms:
            if isinstance(symptom, str) and symptom.strip():
                clean_symptom = symptom.lower().strip()
                if clean_symptom not in [s.lower() for s in unique_symptoms]:
                    unique_symptoms.append(clean_symptom)
        result = {"symptoms": unique_symptoms, "severity": severity}
        logger.info(f"✅ Extraction result: {result}")
        return result
    except Exception as e:
        logger.error(f"❌ Error extracting symptoms: {e}")
        return {"symptoms": [], "severity": 0}

def is_red_flag(text: str, severity: int = 0) -> bool:
    """Check for emergency red flags"""
    text_lower = text.lower()
    for flag in RED_FLAGS:
        if flag in text_lower:
            logger.info(f"🚨 Red flag detected: {flag}")
            return True
    if severity >= 8:
        critical_symptoms = ["chest pain", "abdominal pain", "breathing", "consciousness"]
        if any(symptom in text_lower for symptom in critical_symptoms):
            logger.info(f"🚨 High severity ({severity}) with critical symptom detected")
            return True
    return False

async def classify_intent_with_gpt(message: str, client=None) -> str:
    """Classify user intent"""
    prompt = (
        "You are a classifier. Label the user message with exactly one word: "
        "GREETING, THANKS, INFO_REQUEST, SYMPTOM_REPORT, or OTHER.\n\n"
        f"User: \"{message.strip()}\"\n"
        "Answer with exactly one label, and nothing else."
    )
    try:
        if client is None:
            client = await client_manager.get_openai_client()
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=4,
        )
        label = response.choices[0].message.content.strip().upper()
        valid_labels = {"GREETING", "THANKS", "INFO_REQUEST", "SYMPTOM_REPORT", "OTHER"}
        if label not in valid_labels:
            return "OTHER"
        return label
    except Exception as e:
        logger.error(f"Error in intent classification: {e}")
        return "OTHER"

async def detect_treatment_intent(message: str, context: Dict, client=None) -> bool:
    """Detect if user wants treatment/next-step advice using GPT - ONLY after conditions shown"""
    user_messages = context.get("user_messages", [])
    all_symptoms = context.get("all_symptoms", [])
    if len(all_symptoms) < config.MIN_SYMPTOMS_FOR_PINECONE:
        logger.info(f"🚫 Too few symptoms ({len(all_symptoms)}) for treatment intent")
        return False
    treatment_keywords = [
        "yes", "sure", "please", "okay", "ok", "yep", "yeah", "yup",
        "give me advice", "what should i do", "treatment", "recommendations",
        "next steps", "help me", "what to do"
    ]
    message_lower = message.lower().strip()
    if not any(keyword in message_lower for keyword in treatment_keywords):
        logger.info(f"🚫 No treatment keywords found in: '{message[:50]}...'")
        return False
    symptom_indicators = [
        "i have", "i feel", "i am", "my", "the pain", "it hurts", 
        "started", "began", "since", "yesterday", "today", "morning",
        "stomach", "head", "chest", "back", "throat", "fever", "cough"
    ]
    symptom_count = sum
