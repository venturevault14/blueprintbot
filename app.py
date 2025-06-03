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
        self.OPENAI_API_KEY_2 = os.getenv("OPENAI_API_KEY_2")  # For treatment processing
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
        self.PINECONE_SCORE_THRESHOLD = float(os.getenv("PINECONE_SCORE_THRESHOLD", "0.88"))  # Updated to 0.88
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
    description="AI-powered medical triage assistant for symptom assessment and condition suggestions",
    version="1.0.0"
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
        self._openai_client_2 = None  # For treatment processing
        self._pinecone_client = None
        self._pinecone_index = None
        self._treatment_index = None

    async def get_openai_client(self):
        """Get OpenAI client - always create fresh for serverless"""
        try:
            logger.info("🔑 Creating fresh OpenAI client for serverless...")
            client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
            
            # Quick connection test
            logger.info("🧪 Testing OpenAI connection...")
            await client.models.list()
            logger.info("✅ OpenAI client created and tested successfully")
            
            return client
            
        except Exception as e:
            logger.error(f"❌ Failed to create OpenAI client: {e}")
            raise HTTPException(status_code=503, detail=f"OpenAI connection failed: {str(e)}")

    async def get_openai_client_2(self):
        """Get secondary OpenAI client for treatment processing"""
        try:
            # Use secondary API key if available, otherwise fall back to primary
            api_key = config.OPENAI_API_KEY_2 or config.OPENAI_API_KEY
            logger.info("🔑 Creating fresh OpenAI client 2 for treatment processing...")
            client = AsyncOpenAI(api_key=api_key)
            
            # Quick connection test
            await client.models.list()
            logger.info("✅ OpenAI client 2 created and tested successfully")
            
            return client
            
        except Exception as e:
            logger.error(f"❌ Failed to create OpenAI client 2: {e}")
            raise HTTPException(status_code=503, detail=f"Treatment OpenAI connection failed: {str(e)}")

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
                
                # Test the connection
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
                # Initialize client if not already done
                if self._pinecone_client is None:
                    logger.info("🔍 Initializing Pinecone client for treatment index...")
                    self._pinecone_client = Pinecone(api_key=config.PINECONE_API_KEY)
                
                logger.info(f"🔗 Connecting to treatment index: {config.TREATMENT_INDEX_NAME}")
                self._treatment_index = self._pinecone_client.Index(name=config.TREATMENT_INDEX_NAME)
                
                # Test the connection
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
    
    # Common symptom keywords
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
    
    # Use provided client or get fresh one
    if client is None:
        try:
            client = await client_manager.get_openai_client()
        except Exception as e:
            logger.error(f"❌ Could not get OpenAI client for thread validation: {e}")
            return False
    
    # Try to retrieve the thread with retry
    for attempt in range(2):
        try:
            await client.beta.threads.retrieve(thread_id=thread_id.strip())
            logger.info(f"✅ Thread {thread_id} validated successfully (attempt {attempt + 1})")
            return True
        except Exception as e:
            logger.error(f"❌ Thread validation attempt {attempt + 1} failed for {thread_id}: {e}")
            if attempt == 0:
                # Try once more with a small delay
                import asyncio
                await asyncio.sleep(0.5)
            
    return False

async def get_thread_context(thread_id: str, client=None) -> Dict:
    """Retrieve and analyze thread context"""
    try:
        if client is None:
            client = await client_manager.get_openai_client()
            
        logger.info(f"📚 Retrieving context for thread: {thread_id}")
        
        messages = await client.beta.threads.messages.list(
            thread_id=thread_id,
            order='asc',
            limit=100
        )

        user_messages = []
        all_symptoms = []
        max_severity = 0
        
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

        # Deduplicate symptoms
        unique_symptoms = list(dict.fromkeys([s.lower().strip() for s in all_symptoms if s.strip()]))
        
        context = {
            "user_messages": user_messages,
            "all_symptoms": unique_symptoms,
            "max_severity": max_severity
        }
        
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

        # Extract severity
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

        # Check for numeric pain scale
        pain_scale_match = re.search(r"pain\s+(\d+)/10", description_lower)
        if pain_scale_match:
            severity = int(pain_scale_match.group(1))
            logger.info(f"📊 Found numeric pain scale: {severity}/10")

        # Extract symptoms using GPT
        prompt = (
            "Extract specific medical symptoms from this text. Return ONLY a JSON array of symptom strings. "
            "Be precise and use standard medical terminology. Focus on physical symptoms only. "
            "Do NOT wrap in markdown code blocks. Return raw JSON only. "
            "Example: [\"chest pain\", \"shortness of breath\", \"nausea\"]\n\n"
            f"Text: \"{description}\""
        )

        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Extract symptoms now."}
            ],
            temperature=0,
            max_tokens=200
        )

        try:
            raw_response = response.choices[0].message.content.strip()
            logger.info(f"🤖 GPT raw response: {raw_response[:100]}...")
            
            # Handle markdown code blocks
            if raw_response.startswith("```json"):
                # Extract JSON from markdown code block
                json_start = raw_response.find('[')
                json_end = raw_response.rfind(']') + 1
                if json_start != -1 and json_end > json_start:
                    json_str = raw_response[json_start:json_end]
                    logger.info(f"🔧 Extracted JSON from markdown: {json_str}")
                else:
                    logger.warning("⚠️ Could not extract JSON from markdown block")
                    json_str = "[]"
            elif raw_response.startswith("```"):
                # Handle other code block formats
                lines = raw_response.split('\n')
                json_lines = [line for line in lines if line.strip() and not line.startswith('```')]
                json_str = '\n'.join(json_lines)
                logger.info(f"🔧 Extracted JSON from code block: {json_str}")
            else:
                json_str = raw_response
            
            symptoms = json.loads(json_str)
            if not isinstance(symptoms, list):
                logger.warning(f"⚠️ GPT returned non-list: {symptoms}")
                symptoms = []
            else:
                logger.info(f"✅ Successfully parsed {len(symptoms)} symptoms")
                
        except json.JSONDecodeError as e:
            logger.warning(f"⚠️ JSON parsing failed: {e}")
            logger.warning(f"⚠️ Raw content: {raw_response}")
            
            # Fallback: try to extract symptoms using keyword matching
            logger.info("🔄 Falling back to keyword-based symptom extraction")
            symptoms = extract_symptoms_fallback(description)
            logger.info(f"🔄 Fallback extracted: {symptoms}")

        # Clean and deduplicate symptoms
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

    # Check explicit red flags
    for flag in RED_FLAGS:
        if flag in text_lower:
            logger.info(f"🚨 Red flag detected: {flag}")
            return True

    # Check high severity with critical symptoms
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
    
    # First check: Has the user already been shown conditions?
    # Only detect treatment intent if we've already provided an assessment
    user_messages = context.get("user_messages", [])
    all_symptoms = context.get("all_symptoms", [])
    
    # If we haven't shown conditions yet (less than MIN_SYMPTOMS or early conversation), 
    # this CAN'T be a treatment request
    if len(all_symptoms) < config.MIN_SYMPTOMS_FOR_PINECONE:
        logger.info(f"🚫 Too few symptoms ({len(all_symptoms)}) for treatment intent")
        return False
    
    # Look for treatment-specific keywords first
    treatment_keywords = [
        "yes", "sure", "please", "okay", "ok", "yep", "yeah", "yup",
        "give me advice", "what should i do", "treatment", "recommendations",
        "next steps", "help me", "what to do"
    ]
    
    message_lower = message.lower().strip()
    
    # If the message doesn't contain any treatment-indicating words, it's not a treatment request
    if not any(keyword in message_lower for keyword in treatment_keywords):
        logger.info(f"🚫 No treatment keywords found in: '{message[:50]}...'")
        return False
    
    # If it's mostly about symptoms, it's not a treatment request
    symptom_indicators = [
        "i have", "i feel", "i am", "my", "the pain", "it hurts", 
        "started", "began", "since", "yesterday", "today", "morning",
        "stomach", "head", "chest", "back", "throat", "fever", "cough"
    ]
    
    symptom_count = sum(1 for indicator in symptom_indicators if indicator in message_lower)
    if symptom_count >= 2:  # If 2+ symptom indicators, it's probably symptom reporting
        logger.info(f"🚫 Too many symptom indicators ({symptom_count}) - likely symptom report")
        return False

    # Now use GPT for final determination, but with a stricter prompt
    prompt = (
        "You are analyzing a user's response in a medical conversation. "
        "The user has already described symptoms and been shown possible medical conditions. "
        "Now they were asked: 'Would you like next-step recommendations (treatment tips, what to do)?'"
        "Determine if this response means YES (wants treatment advice) or NO (doesn't want advice). "
        "Only return 'YES' if they are clearly requesting treatment/next-step advice. "
        "If they're describing MORE symptoms or asking other questions, return 'NO'. "
        "Return only 'YES' or 'NO' - nothing else.\n\n"
        f"User response: \"{message.strip()}\""
    )

    try:
        if client is None:
            client = await client_manager.get_openai_client()
            
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=3,
        )
        
        result = response.choices[0].message.content.strip().upper()
        wants_treatment = result == "YES"
        
        logger.info(f"🎯 Treatment intent detection: '{message[:50]}...' → {wants_treatment}")
        return wants_treatment

    except Exception as e:
        logger.error(f"Error in treatment intent detection: {e}")
        # Default to False if unsure
        return False

async def should_query_pinecone_database(context: Dict, conversation_depth: int = 0) -> bool:
    """Determine if we should query the medical database - more conversational approach"""
    all_symptoms = context.get("all_symptoms", [])
    symptom_count = len(all_symptoms)
    max_severity = context.get("max_severity", 0)
    full_text = " ".join(context.get("user_messages", [])).lower()
    user_message_count = len(context.get("user_messages", []))

    # Always query for emergencies
    if is_red_flag(full_text, max_severity):
        logger.info("🚨 Emergency detected - querying Pinecone immediately")
        return True

    # For explicit "what could this be" requests - always query
    condition_keywords = [
        "what might be", "what could be", "what is", "what do i have",
        "diagnosis", "condition", "disease", "what's wrong with me"
    ]
    if any(keyword in full_text for keyword in condition_keywords):
        logger.info("🔍 Explicit condition request - querying Pinecone")
        return True

    # NEW: More conversational approach
    # If user has 3+ symptoms but it's still early in conversation (≤2 exchanges)
    # Ask for more context first to be more thorough
    if symptom_count >= config.MIN_SYMPTOMS_FOR_PINECONE:
        if user_message_count <= 2:
            logger.info(f"🤔 Have {symptom_count} symptoms but only {user_message_count} exchanges - gathering more context first")
            return False
        else:
            logger.info(f"💡 Have {symptom_count} symptoms and {user_message_count} exchanges - ready for assessment")
            return True

    logger.info(f"📝 Continuing conversation: {symptom_count} symptoms, {user_message_count} messages")
    return False

async def should_offer_treatment(context: Dict) -> bool:
    """Determine if we should offer treatment advice after showing conditions"""
    symptoms = context.get("all_symptoms", [])
    return len(symptoms) >= config.MIN_SYMPTOMS_FOR_PINECONE

async def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Get embeddings from OpenAI"""
    client = await client_manager.get_openai_client()
    for attempt in range(config.MAX_RETRIES):
        try:
            response = await client.embeddings.create(
                input=texts,
                model=config.EMBEDDING_MODEL
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Embedding attempt {attempt+1}/{config.MAX_RETRIES} failed: {e}")
            if attempt == config.MAX_RETRIES - 1:
                return None
    return None

async def get_treatment_embeddings(texts: List[str]) -> List[List[float]]:
    """Get embeddings for treatment queries using secondary OpenAI client"""
    client = await client_manager.get_openai_client_2()
    
    for attempt in range(config.MAX_RETRIES):
        try:
            response = await client.embeddings.create(
                input=texts,
                model=config.EMBEDDING_MODEL
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Treatment embedding attempt {attempt+1}/{config.MAX_RETRIES} failed: {e}")
            if attempt == config.MAX_RETRIES - 1:
                raise HTTPException(status_code=503, detail=f"Failed to generate treatment embeddings: {str(e)}")
    
    raise HTTPException(status_code=503, detail="Failed to generate treatment embeddings")

async def query_index(query_text: str, symptoms: List[str], context: Dict, top_k: int = 50) -> List[Dict]:
    """Query Pinecone index for medical conditions - IMPROVED for variety"""
    try:
        pinecone_index = client_manager.get_pinecone_index()
        if not pinecone_index:
            logger.warning("Pinecone index not available")
            return []
            
        query_embedding = await get_embeddings([query_text])
        if not query_embedding:
            logger.error("Failed to generate query embedding")
            return []

        # Query with more results to get variety
        response = pinecone_index.query(
            vector=query_embedding[0],
            top_k=top_k,
            include_metadata=True
        )

        matches = response.get("matches", [])
        logger.info(f"Pinecone returned {len(matches)} matches")

        # Filter by score threshold (using 0.88 as requested)
        filtered_matches = [
            match for match in matches 
            if match.get("score", 0) >= config.PINECONE_SCORE_THRESHOLD
        ]
        
        logger.info(f"✅ Selected {len(filtered_matches)} conditions after filtering (score >= {config.PINECONE_SCORE_THRESHOLD})")
        return filtered_matches

    except Exception as e:
        logger.error(f"❌ Error querying index: {e}")
        return []

async def query_treatment_index(symptoms: List[str], context: Dict, top_k: int = 5) -> List[Dict]:
    """Query Pinecone treatment index for Q&A snippets"""
    try:
        # Get treatment index
        treatment_index = client_manager.get_treatment_index()
        if not treatment_index:
            logger.warning("Treatment index not available")
            return []
        
        # Create query text from symptoms
        symptoms_text = ", ".join(symptoms)
        query_text = f"I have {symptoms_text} - what should I do?"
        
        logger.info(f"💊 Querying treatment index with: {query_text}")
        
        # Get embeddings using secondary client
        query_embedding = await get_treatment_embeddings([query_text])
        
        response = treatment_index.query(
            vector=query_embedding[0],
            top_k=top_k,
            include_metadata=True
        )

        matches = response.get("matches", [])
        logger.info(f"📊 Treatment index returned {len(matches)} matches")

        # Filter by score threshold (can be lower for treatment as they're more specific)
        filtered_matches = [
            match for match in matches
            if match.get("score", 0) >= 0.7  # Lower threshold for treatment
        ]
        
        logger.info(f"✅ Selected {len(filtered_matches)} treatment snippets after filtering")
        return filtered_matches

    except Exception as e:
        logger.error(f"❌ Error querying treatment index: {e}")
        return []

async def generate_condition_description(condition_name: str, symptoms: List[str]) -> str:
    """Generate patient-friendly condition description"""
    try:
        client = await client_manager.get_openai_client()
        symptoms_text = ", ".join(symptoms)
        
        prompt = (
            "Explain this medical condition in simple, patient-friendly language. "
            "Use 2-3 sentences. Explain what it is and how it relates to their symptoms. "
            "Do not provide treatment advice."
        )

        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Explain {condition_name} to a patient with: {symptoms_text}"}
            ],
            temperature=0.3,
            max_tokens=150
        )

        description = response.choices[0].message.content.strip()
        if description and len(description) > 20:
            return description

        # Fallback
        return f"{condition_name} may be related to your symptoms. Please consult a healthcare professional for proper evaluation."

    except Exception as e:
        logger.error(f"Error generating description for {condition_name}: {e}")
        return f"{condition_name} may be related to your symptoms. Please consult a healthcare professional."

async def rank_conditions(matches: List[Dict], symptoms: List[str], context: Dict) -> List[ConditionInfo]:
    """Rank and format condition information - Show ALL conditions that meet score threshold"""
    try:
        conditions = []
        seen_diseases = set()  # Track unique disease names
        
        for match in matches:  # Process ALL matches that already passed score filter
            disease_name = match["metadata"].get("disease", "Unknown").title()
            score = match.get("score", 0)
            
            # Skip duplicates
            if disease_name in seen_diseases:
                continue
                
            seen_diseases.add(disease_name)
            description = await generate_condition_description(disease_name, symptoms)

            conditions.append(ConditionInfo(
                name=disease_name,
                description=description,
                file_citation="medical_database.json"
            ))
            
            logger.info(f"✅ Added condition: {disease_name} (score: {score:.3f})")

        logger.info(f"✅ Ranked {len(conditions)} unique conditions meeting threshold: {[c.name for c in conditions]}")
        return conditions

    except Exception as e:
        logger.error(f"Error ranking conditions: {e}")
        return []

async def synthesize_treatment_advice(treatment_matches: List[Dict], symptoms: List[str], context: Dict) -> str:
    """Use GPT to synthesize treatment Q&A snippets into cohesive advice"""
    try:
        client = await client_manager.get_openai_client_2()
        
        # Extract Q&A content directly from Pinecone metadata
        qa_snippets = []
        
        logger.info(f"🔍 Processing {len(treatment_matches)} treatment matches...")
        
        for i, match in enumerate(treatment_matches):
            metadata = match.get("metadata", {})
            
            # Debug: Log what we actually get from Pinecone
            logger.info(f"📋 Match {i+1} metadata keys: {list(metadata.keys())}")
            
            # Your data has the Q&A text in metadata["text"]
            if "text" in metadata:
                qa_text = metadata["text"]
                qa_snippets.append(qa_text)
                logger.info(f"✅ Match {i+1}: Found Q&A text ({len(qa_text)} chars)")
                logger.info(f"📝 Preview: {qa_text[:100]}...")
            else:
                logger.warning(f"⚠️ Match {i+1}: No 'text' field in metadata")
                # Log the full metadata structure for debugging
                logger.info(f"📋 Full metadata: {metadata}")
        
        if not qa_snippets:
            logger.error("❌ No Q&A snippets extracted from treatment matches")
            return "I recommend consulting with a healthcare provider for personalized treatment advice."
        
        logger.info(f"✅ Successfully extracted {len(qa_snippets)} Q&A snippets")
        
        # Create prompt for GPT
        symptoms_text = ", ".join(symptoms)
        
        prompt = (
            "You are a medical assistant. Below are Q&A excerpts from a medical database "
            "that match the patient's symptoms. Create ONE unified set of treatment recommendations.\n\n"
            f"Patient's symptoms: {symptoms_text}\n\n"
            "Relevant Q&A excerpts:\n\n"
            + "\n\n---\n\n".join(qa_snippets) +
            "\n\nBased on these excerpts, provide unified treatment recommendations:"
        )
        
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful medical assistant providing treatment guidance."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=400
        )
        
        treatment_advice = response.choices[0].message.content.strip()
        logger.info(f"✅ Generated treatment advice: {len(treatment_advice)} characters")
        
        return treatment_advice
        
    except Exception as e:
        logger.error(f"❌ Error synthesizing treatment advice: {e}")
        return "I recommend consulting with a healthcare provider for personalized treatment advice."

async def generate_follow_up_questions(context: Dict, client=None) -> List[str]:
    """Generate contextual follow-up questions based on symptoms and conversation stage"""
    try:
        if client is None:
            client = await client_manager.get_openai_client()
            
        symptoms = context.get("all_symptoms", [])
        user_messages = context.get("user_messages", [])
        conversation_depth = len(user_messages)
        symptoms_text = ", ".join(symptoms) if symptoms else "general symptoms"

        # Create a more detailed prompt based on conversation stage
        if conversation_depth <= 1:
            # Early conversation - gather basic details
            prompt = (
                "You are an experienced triage nurse. The patient has mentioned these symptoms: {symptoms}. "
                "Generate 2-3 warm, empathetic follow-up questions to gather essential details. "
                "Focus on: timing, severity, triggers, and associated symptoms. "
                "Return JSON: {{'questions': ['question1', 'question2']}}"
            ).format(symptoms=symptoms_text)
        else:
            # Deeper conversation - gather specific context
            recent_messages = " ".join(user_messages[-2:])
            prompt = (
                "You are an experienced triage nurse. The patient has been telling you about: {symptoms}. "
                "Recent conversation: '{recent}'. "
                "Generate 2-3 specific, targeted questions to understand their condition better before assessment. "
                "Ask about things like: impact on daily life, what makes it better/worse, other related symptoms, "
                "medical history relevance, or specific characteristics. "
                "Return JSON: {{'questions': ['question1', 'question2']}}"
            ).format(symptoms=symptoms_text, recent=recent_messages)

        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Current symptoms: {symptoms_text}"}
            ],
            temperature=0.4,
            max_tokens=200
        )

        try:
            result = json.loads(response.choices[0].message.content.strip())
            questions = result.get("questions", [])
            logger.info(f"💬 Generated {len(questions)} follow-up questions")
            return questions
        except json.JSONDecodeError:
            logger.warning("Failed to parse follow-up questions JSON")
            return [
                "Can you tell me more about when this started?",
                "How would you rate the severity on a scale of 1-10?",
                "Is there anything that makes it better or worse?"
            ]

    except Exception as e:
        logger.error(f"Error generating follow-up questions: {e}")
        return [
            "Can you describe your symptoms in more detail?",
            "When did you first notice these symptoms?"
        ]

# ========================================
# Response Generators
# ========================================

async def add_message_to_thread(thread_id: str, content: str, role: str = "assistant", client=None):
    """Helper to add message to thread"""
    try:
        if client is None:
            client = await client_manager.get_openai_client()
        await client.beta.threads.messages.create(
            thread_id=thread_id,
            role=role,
            content=content
        )
    except Exception as e:
        logger.error(f"Error adding message to thread {thread_id}: {e}")

def serialize_response_data(response_data: dict) -> str:
    """Safely serialize response data to JSON string"""
    try:
        # Convert any Pydantic models to dictionaries
        serializable_data = {}
        for key, value in response_data.items():
            if hasattr(value, 'dict'):  # Pydantic model
                serializable_data[key] = value.dict()
            elif isinstance(value, list):
                # Handle lists that might contain Pydantic models
                serializable_data[key] = [
                    item.dict() if hasattr(item, 'dict') else item 
                    for item in value
                ]
            else:
                serializable_data[key] = value
        
        return json.dumps(serializable_data)
    except Exception as e:
        logger.error(f"Error serializing response data: {e}")
        return json.dumps({"error": "Failed to serialize response"})

async def generate_greeting_response(thread_id: str, client=None) -> TriageResponse:
    """Generate friendly greeting response"""
    response_data = {
        "text": (
            "Hello! 👋 I'm here to help you with any health concerns you might have. "
            "I'm a medical triage assistant, which means I can help you figure out if your symptoms "
            "need immediate attention, suggest what might be causing them, and point you to the right "
            "kind of care.\n\n"
            "Feel free to tell me about any symptoms you're experiencing, or ask me anything about your health!"
        ),
        "possible_conditions": [],
        "safety_measures": [],
        "triage": TriageInfo(type="", location="Unknown"),
        "send_sos": False,
        "follow_up_questions": [
            "How are you feeling today?",
            "Is there anything health-related I can help you with?"
        ],
        "thread_id": thread_id,
        "symptoms_count": 0,
        "should_query_pinecone": False
    }

    await add_message_to_thread(thread_id, serialize_response_data(response_data), client=client)
    return TriageResponse(**response_data)

async def generate_thanks_response(thread_id: str, client=None) -> TriageResponse:
    """Generate warm thank you response"""
    response_data = {
        "text": (
            "You're very welcome! 😊 I'm really glad I could help. "
            "If you have any other health questions or concerns—now or later—please don't hesitate to ask. "
            "Your health and well-being are important, and I'm here whenever you need guidance.\n\n"
            "Take care of yourself!"
        ),
        "possible_conditions": [],
        "safety_measures": [],
        "triage": TriageInfo(type="", location="Unknown"),
        "send_sos": False,
        "follow_up_questions": [
            "Is there anything else I can help you with?",
            "Feel free to reach out anytime you have health questions!"
        ],
        "thread_id": thread_id,
        "symptoms_count": 0,
        "should_query_pinecone": False
    }

    await add_message_to_thread(thread_id, serialize_response_data(response_data), client=client)
    return TriageResponse(**response_data)

async def generate_info_request_response(thread_id: str, client=None) -> TriageResponse:
    """Generate informative response about capabilities"""
    response_data = {
        "text": (
            "Great question! I'm your medical triage assistant, and I'm here to help you navigate health concerns. "
            "Here's what I can do for you:\n\n"
            "🔍 **Symptom Assessment**: Tell me what you're experiencing, and I'll help you understand what it might be\n"
            "🏥 **Care Recommendations**: I'll suggest whether you should see a doctor, visit urgent care, or if it's an emergency\n"
            "📍 **Find Care**: I can help you locate clinics and hospitals in Nigeria\n"
            "❓ **Answer Questions**: Ask me about symptoms, conditions, or general health concerns\n\n"
            "Just describe how you're feeling in your own words—like you would to a friend. "
            "For example: 'I have a headache that won't go away' or 'I'm feeling dizzy and nauseous.'\n\n"
            "What can I help you with today?"
        ),
        "possible_conditions": [],
        "safety_measures": [],
        "triage": TriageInfo(type="", location="Unknown"),
        "send_sos": False,
        "follow_up_questions": [
            "Do you have any symptoms you'd like me to look at?",
            "What's your city if you need to find nearby healthcare?"
        ],
        "thread_id": thread_id,
        "symptoms_count": 0,
        "should_query_pinecone": False
    }

    await add_message_to_thread(thread_id, serialize_response_data(response_data), client=client)
    return TriageResponse(**response_data)

async def generate_conversational_response(context: Dict, is_emergency: bool, thread_id: str, client=None) -> TriageResponse:
    """Generate conversational medical response that feels like talking to a real nurse"""
    try:
        symptoms = context["all_symptoms"]
        symptom_count = len(symptoms)
        user_messages = context.get("user_messages", [])
        conversation_depth = len(user_messages)
        
        # Check if this is a treatment request (user wants next-step advice)
        # ONLY check for treatment intent if we have enough symptoms
        latest_message = user_messages[-1] if user_messages else ""
        wants_treatment = False
        
        if symptom_count >= config.MIN_SYMPTOMS_FOR_PINECONE:
            wants_treatment = await detect_treatment_intent(latest_message, context, client)
        
        should_query = await should_query_pinecone_database(context, conversation_depth)

        logger.info(f"🗣️ Generating response: {symptom_count} symptoms, depth {conversation_depth}, query: {should_query}, treatment: {wants_treatment}")

        possible_conditions = []
        treatment_advice = ""
        
        # TREATMENT PHASE - User wants next-step advice AFTER seeing conditions
        if wants_treatment and symptom_count >= config.MIN_SYMPTOMS_FOR_PINECONE:
            logger.info("💊 Processing treatment request...")
            try:
                # Query treatment index
                treatment_matches = await query_treatment_index(symptoms, context)
                treatment_advice = await synthesize_treatment_advice(treatment_matches, symptoms, context)
                
                response_data = {
                    "text": (
                        f"Here's what I recommend for your symptoms ({', '.join(symptoms)}):\n\n"
                        f"{treatment_advice}\n\n"
                        "*This is general guidance. Please consult a healthcare professional for personalized treatment.*"
                    ),
                    "possible_conditions": [],
                    "safety_measures": ["Follow the recommended steps above", "Monitor your symptoms", "Seek professional care if symptoms worsen"],
                    "triage": TriageInfo(type="self_care", location="Unknown"),
                    "send_sos": False,
                    "follow_up_questions": [
                        "Do you have any questions about these recommendations?",
                        "Is there anything else I can help you with?"
                    ],
                    "thread_id": thread_id,
                    "symptoms_count": symptom_count,
                    "should_query_pinecone": False  # Don't query again unless new symptoms
                }
                
                await add_message_to_thread(thread_id, serialize_response_data(response_data), client=client)
                return TriageResponse(**response_data)
                
            except Exception as e:
                logger.error(f"❌ Treatment processing failed: {e}")
                # Fall back to regular response
                wants_treatment = False

        # CONDITION ASSESSMENT PHASE - Query Pinecone when we have enough symptoms
        if should_query and not wants_treatment:
            logger.info(f"🔍 Querying Pinecone for {symptom_count} symptoms: {symptoms}")
            query_text = f"Symptoms: {', '.join(symptoms)}"
            matches = await query_index(query_text, symptoms, context)
            possible_conditions = await rank_conditions(matches, symptoms, context)

        # Generate response based on phase
        if is_emergency:
            text_parts = [
                "🚨 Based on what you've told me, this sounds like it could be a medical emergency.",
                f"Please call {config.NIGERIA_EMERGENCY_HOTLINE} immediately or go to your nearest hospital emergency room.",
                "Don't wait - it's better to be safe when it comes to your health."
            ]
            safety_measures = [
                f"Call {config.NIGERIA_EMERGENCY_HOTLINE} immediately",
                "Do not drive yourself - call for help",
                "Stay calm and follow emergency operator instructions"
            ]
            triage_type = "hospital"
            send_sos = True
            follow_up_questions = []
            
        elif should_query and possible_conditions and not wants_treatment:
            # Assessment with conditions - OFFER TREATMENT
            symptoms_text = ", ".join(symptoms)
            text_parts = [
                f"Thank you for sharing all that information with me. Based on your symptoms - {symptoms_text} - I can see why you're concerned.",
                "",
                "Here are the conditions that could potentially match what you're experiencing:"
            ]
            
            for i, condition in enumerate(possible_conditions, 1):
                text_parts.append(f"**{i}. {condition.name}**")
                text_parts.append(f"   {condition.description}")
                text_parts.append("")

            text_parts.extend([
                "Now, I want to be clear that this isn't a diagnosis - only a healthcare provider who can examine you properly can determine that.",
                "",
                "**Would you like some next-step recommendations (precautions, treatment tips, what to do immediately) for any of these conditions?**"
            ])
            
            safety_measures = [
                "Monitor your symptoms closely",
                "Stay hydrated and get adequate rest", 
                "See a healthcare provider for proper evaluation",
                "Seek immediate care if symptoms worsen"
            ]
            triage_type = "clinic"
            send_sos = False
            follow_up_questions = [
                "Would you like specific treatment recommendations?",
                "Do you have any questions about these possibilities?"
            ]
            
        else:
            # Conversation continuation - gather more context
            symptoms_text = ", ".join(symptoms) if symptoms else "what you're experiencing"
            
            if conversation_depth <= 1:
                # Early conversation
                text_parts = [
                    f"I understand you're dealing with {symptoms_text}, and I want to help you figure out the best next steps.",
                    "",
                    "To give you the most helpful guidance, I'd like to understand your situation better:"
                ]
            else:
                # Deeper conversation
                text_parts = [
                    f"Thank you for the additional details about {symptoms_text}.",
                    "",
                    "I'm getting a clearer picture of what's going on. Let me ask a few more specific questions to help guide my recommendations:"
                ]
            
            # Get and add follow-up questions
            follow_up_questions = await generate_follow_up_questions(context, client)
            
            for i, question in enumerate(follow_up_questions, 1):
                text_parts.append(f"{i}. {question}")
            
            text_parts.extend([
                "",
                "Take your time answering - the more I understand about your situation, the better I can help guide you to the right care."
            ])
            
            safety_measures = [
                "Continue monitoring your symptoms",
                "Stay hydrated and rest as needed",
                "Contact emergency services if symptoms suddenly worsen"
            ]
            triage_type = "clinic"
            send_sos = False

        response_data = {
            "text": "\n".join(text_parts),
            "possible_conditions": possible_conditions,
            "safety_measures": safety_measures,
            "triage": TriageInfo(type=triage_type, location="Unknown"),
            "send_sos": send_sos,
            "follow_up_questions": follow_up_questions,
            "thread_id": thread_id,
            "symptoms_count": symptom_count,
            "should_query_pinecone": should_query
        }

        await add_message_to_thread(thread_id, serialize_response_data(response_data), client=client)
        return TriageResponse(**response_data)

    except Exception as e:
        logger.error(f"❌ Error in generate_conversational_response: {e}")
        # Emergency fallback
        return TriageResponse(
            text=f"I'm experiencing technical difficulties right now. If this is urgent, please call {config.NIGERIA_EMERGENCY_HOTLINE} immediately.",
            possible_conditions=[],
            safety_measures=["Seek immediate medical attention if urgent"],
            triage=TriageInfo(type="hospital", location="Unknown"),
            send_sos=True,
            follow_up_questions=[],
            thread_id=thread_id,
            symptoms_count=len(context.get("all_symptoms", [])),
            should_query_pinecone=False
        )

# ========================================
# API Endpoints
# ========================================

@app.post("/triage", response_model=TriageResponse)
async def triage(request: TriageRequest):
    """Main triage endpoint with enhanced intent classification and treatment support"""
    try:
        # Get OpenAI client (initialize if needed)
        client = await client_manager.get_openai_client()
        
        description = request.description.strip()
        thread_id = request.thread_id

        logger.info(f"🚀 Triage request: '{description[:50]}...', provided thread: {thread_id}")

        # Handle thread validation and management
        use_existing_thread = False
        if thread_id and thread_id.strip():
            # Try to validate the provided thread
            if await validate_thread(thread_id.strip(), client):
                use_existing_thread = True
                thread_id = thread_id.strip()
                logger.info(f"✅ Using existing thread: {thread_id}")
            else:
                logger.warning(f"⚠️ Invalid thread provided: {thread_id}, creating new one")
                use_existing_thread = False

        # Create new thread only if we don't have a valid existing one
        if not use_existing_thread:
            try:
                new_thread = await client.beta.threads.create()
                thread_id = new_thread.id
                logger.info(f"🆕 Created new thread: {thread_id}")
            except Exception as e:
                logger.error(f"❌ Failed to create OpenAI thread: {e}")
                raise HTTPException(
                    status_code=503,
                    detail="Unable to create conversation thread. Please try again."
                )

        # Add user message to thread
        try:
            await client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content=description
            )
            logger.info(f"✅ Added user message to thread {thread_id}")
        except Exception as e:
            logger.error(f"❌ Failed to add message to thread: {e}")
            # Continue anyway - we can still provide a response

        # Quick GPT-based intent classification
        intent_label = await classify_intent_with_gpt(description, client)
        logger.info(f"🎯 Intent classified as: {intent_label}")

        # Handle conversational intents with friendly responses
        if intent_label == "GREETING":
            return await generate_greeting_response(thread_id, client)
        elif intent_label == "THANKS":
            return await generate_thanks_response(thread_id, client)
        elif intent_label == "INFO_REQUEST":
            return await generate_info_request_response(thread_id, client)

        # For SYMPTOM_REPORT and OTHER, continue with existing medical logic
        elif intent_label in ["SYMPTOM_REPORT", "OTHER"]:
            # Get thread context - this should now include ALL previous messages
            context = await get_thread_context(thread_id, client)

            symptom_count = len(context["all_symptoms"])
            max_severity = context["max_severity"]
            is_emergency = is_red_flag(" ".join(context["user_messages"]), max_severity)

            logger.info(f"🩺 Medical analysis:")
            logger.info(f"   - Total symptoms: {symptom_count} ({context['all_symptoms']})")
            logger.info(f"   - Emergency: {is_emergency}")

            # Route to conversational response generator
            return await generate_conversational_response(context, is_emergency, thread_id, client)

        # Fallback
        else:
            return await generate_greeting_response(thread_id, client)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in triage endpoint: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)}"
        )

@app.post("/test-treatment")
async def test_treatment():
    """Test treatment system with sample data"""
    try:
        logger.info("🧪 Testing treatment system...")
        
        # Sample symptoms for testing
        test_symptoms = ["headache", "nausea", "light sensitivity"]
        test_context = {
            "all_symptoms": test_symptoms,
            "user_messages": ["I have a headache", "I also feel nauseous", "and bright lights hurt my eyes", "yes please give me advice"]
        }
        
        # Test treatment intent detection
        wants_treatment = await detect_treatment_intent("yes please give me advice", test_context)
        logger.info(f"Treatment intent detected: {wants_treatment}")
        
        # Test treatment index query
        treatment_matches = await query_treatment_index(test_symptoms, test_context)
        logger.info(f"Found {len(treatment_matches)} treatment matches")
        
        # Test synthesis
        if treatment_matches:
            advice = await synthesize_treatment_advice(treatment_matches, test_symptoms, test_context)
            logger.info(f"Generated advice: {advice[:100]}...")
        else:
            advice = "No treatment matches found"
        
        return {
            "status": "success",
            "treatment_intent": wants_treatment,
            "matches_found": len(treatment_matches),
            "treatment_advice": advice,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Treatment test failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "failed",
                "error": str(e),
                "error_type": type(e).__name__,
                "timestamp": datetime.utcnow().isoformat()
            }
        )

@app.get("/debug/treatment-index")
async def debug_treatment_index():
    """Debug endpoint to check treatment index status"""
    try:
        treatment_index = client_manager.get_treatment_index()
        if not treatment_index:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unavailable",
                    "message": "Treatment index not available",
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
        
        stats = treatment_index.describe_index_stats()
        
        return {
            "status": "healthy",
            "index_name": config.TREATMENT_INDEX_NAME,
            "total_vectors": stats.total_vector_count,
            "dimension": stats.dimension,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Treatment index debug failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )

@app.get("/debug/treatment-sample")
async def debug_treatment_sample():
    """Debug endpoint to see treatment index sample data"""
    try:
        treatment_index = client_manager.get_treatment_index()
        if not treatment_index:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unavailable",
                    "message": "Treatment index not available"
                }
            )
        
        # Query for a sample to see metadata structure
        dummy_embedding = [0.1] * 1536  # Dummy embedding to get sample data
        
        response = treatment_index.query(
            vector=dummy_embedding,
            top_k=3,
            include_metadata=True
        )
        
        matches = response.get("matches", [])
        
        samples = []
        for i, match in enumerate(matches):
            metadata = match.get("metadata", {})
            text_content = metadata.get("text", "No text field found")
            
            samples.append({
                "match_index": i,
                "score": match.get("score", 0),
                "metadata_keys": list(metadata.keys()),
                "has_text_field": "text" in metadata,
                "text_preview": text_content[:300] + "..." if len(text_content) > 300 else text_content,
                "text_length": len(text_content) if isinstance(text_content, str) else 0
            })
        
        return {
            "status": "success",
            "total_samples": len(samples),
            "samples": samples,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Treatment sample debug failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )

@app.get("/debug/thread/{thread_id}")
async def debug_thread(thread_id: str):
    """Debug endpoint to inspect thread contents"""
    try:
        logger.info(f"🔍 Debugging thread: {thread_id}")
        client = await client_manager.get_openai_client()
        
        # Get all messages from the thread
        messages = await client.beta.threads.messages.list(
            thread_id=thread_id,
            order='asc',
            limit=100
        )
        
        debug_info = {
            "thread_id": thread_id,
            "total_messages": len(messages.data),
            "messages": []
        }
        
        all_symptoms = []
        max_severity = 0
        
        for i, msg in enumerate(messages.data):
            message_info = {
                "index": i,
                "role": msg.role,
                "timestamp": msg.created_at,
                "content": ""
            }
            
            if msg.content and len(msg.content) > 0:
                if hasattr(msg.content[0], "text"):
                    content = msg.content[0].text.value
                    message_info["content"] = content[:200] + "..." if len(content) > 200 else content
                    
                    # Extract symptoms from user messages
                    if msg.role == "user":
                        try:
                            symptom_data = await extract_symptoms_comprehensive(content, client)
                            message_info["extracted_symptoms"] = symptom_data["symptoms"]
                            message_info["severity"] = symptom_data["severity"]
                            all_symptoms.extend(symptom_data["symptoms"])
                            max_severity = max(max_severity, symptom_data["severity"])
                        except Exception as e:
                            message_info["symptom_extraction_error"] = str(e)
            
            debug_info["messages"].append(message_info)
        
        # Deduplicate symptoms
        unique_symptoms = list(dict.fromkeys([s.lower().strip() for s in all_symptoms if s.strip()]))
        
        debug_info["summary"] = {
            "unique_symptoms": unique_symptoms,
            "symptom_count": len(unique_symptoms),
            "max_severity": max_severity,
            "should_query_pinecone": len(unique_symptoms) >= config.MIN_SYMPTOMS_FOR_PINECONE
        }
        
        return debug_info
        
    except Exception as e:
        logger.error(f"❌ Debug thread failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "thread_id": thread_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint - tests all required services"""
    try:
        # Test OpenAI
        openai_status = "healthy"
        try:
            client = await client_manager.get_openai_client()
        except Exception as e:
            openai_status = f"unhealthy: {str(e)}"

        # Test Pinecone (optional)
        pinecone_status = "not_available"
        if PINECONE_AVAILABLE and config.PINECONE_API_KEY:
            try:
                index = client_manager.get_pinecone_index()
                if index:
                    pinecone_status = "healthy"
                else:
                    pinecone_status = "unavailable"
            except Exception:
                pinecone_status = "unhealthy"

        # Test Treatment Index (optional)
        treatment_status = "not_available"
        if PINECONE_AVAILABLE and config.PINECONE_API_KEY:
            try:
                treatment_index = client_manager.get_treatment_index()
                if treatment_index:
                    treatment_status = "healthy"
                else:
                    treatment_status = "unavailable"
            except Exception:
                treatment_status = "unhealthy"

        overall_status = "healthy" if openai_status == "healthy" else "degraded"

        return HealthResponse(
            status=overall_status,
            timestamp=datetime.utcnow().isoformat(),
            version="1.0.0",
            services={
                "openai": openai_status,
                "pinecone": pinecone_status,
                "treatment_index": treatment_status
            }
        )
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.utcnow().isoformat(),
            version="1.0.0",
            services={"error": str(e)}
        )

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Medical Triage Assistant API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "features": ["symptom_assessment", "condition_suggestions", "treatment_advice"]
    }

# ========================================
# Error Handlers
# ========================================

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred",
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# Export for Vercel
handler = app
