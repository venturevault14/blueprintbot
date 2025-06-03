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

# Import Pinecone (required, not optional)
try:
    from pinecone import Pinecone
    logger.info("✅ Pinecone client imported successfully")
except ImportError:
    logger.error("❌ Pinecone client not found. Please install: pip install pinecone")
    raise ImportError("Pinecone is required. Please install: pip install pinecone")

# ========================================
# Configuration from Environment Variables
# ========================================

class Config:
    def __init__(self):
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        self.PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east1-gcp")
        self.ASSISTANT_ID = os.getenv("ASSISTANT_ID", "asst_pAhSF6XJsj60efD9GEVdEG5n")
        self.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
        self.INDEX_NAME = os.getenv("INDEX_NAME", "triage-index")
        self.MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
        self.MIN_SYMPTOMS_FOR_PINECONE = int(os.getenv("MIN_SYMPTOMS_FOR_PINECONE", "3"))
        self.MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "3"))
        self.NIGERIA_EMERGENCY_HOTLINE = os.getenv("EMERGENCY_HOTLINE", "112")
        self.PINECONE_SCORE_THRESHOLD = float(os.getenv("PINECONE_SCORE_THRESHOLD", "0.8"))
        self.PORT = int(os.getenv("PORT", "8000"))

    def validate(self):
        """Validate required environment variables"""
        required_vars = ["OPENAI_API_KEY", "PINECONE_API_KEY"]
        missing = [var for var in required_vars if not getattr(self, var)]
        if missing:
            raise ValueError(f"Missing required environment variables: {missing}")

config = Config()

# Validate configuration - fail fast if missing required vars
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
# Client Management (Always Available)
# ========================================

class ClientManager:
    def __init__(self):
        self._openai_client = None
        self._pinecone_client = None
        self._pinecone_index = None

    async def get_openai_client(self):
        """Get OpenAI client - always create fresh for serverless"""
        # Always create a fresh client for serverless reliability
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

    def get_pinecone_index(self):
        """Get Pinecone index - cache for performance"""
        if self._pinecone_index is None:
            if not config.PINECONE_API_KEY:
                raise HTTPException(status_code=503, detail="Pinecone API key not configured")
            
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
                raise HTTPException(
                    status_code=503, 
                    detail=f"Failed to connect to Pinecone: {str(e)}"
                )
        
        return self._pinecone_index

# Global client manager
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
            "Extract specific medical symptoms from this text. Return a JSON array of symptom strings. "
            "Be precise and use standard medical terminology. Focus on physical symptoms only. "
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
            symptoms = json.loads(response.choices[0].message.content.strip())
            if not isinstance(symptoms, list):
                logger.warning(f"⚠️ GPT returned non-list: {symptoms}")
                symptoms = []
        except json.JSONDecodeError as e:
            logger.warning(f"⚠️ GPT returned invalid JSON: {response.choices[0].message.content}")
            symptoms = []

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

async def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Get embeddings from OpenAI - ALWAYS SUCCEED OR FAIL"""
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
                raise HTTPException(status_code=503, detail=f"Failed to generate embeddings: {str(e)}")
    
    # This should never be reached due to the raise above
    raise HTTPException(status_code=503, detail="Failed to generate embeddings")

async def query_index(query_text: str, symptoms: List[str], context: Dict, top_k: int = 50) -> List[Dict]:
    """Query Pinecone index for medical conditions - ALWAYS SUCCEED OR FAIL"""
    try:
        # Get Pinecone index (will fail if not available)
        pinecone_index = client_manager.get_pinecone_index()
        
        # Get embeddings (will fail if not available)
        query_embedding = await get_embeddings([query_text])
        
        logger.info(f"🔍 Querying Pinecone with: {query_text}")
        
        response = pinecone_index.query(
            vector=query_embedding[0],
            top_k=top_k,
            include_metadata=True
        )

        matches = response.get("matches", [])
        logger.info(f"📊 Pinecone returned {len(matches)} matches")

        # Filter by score threshold
        filtered_matches = []
        unique_conditions = {}
        
        for match in matches:
            score = match.get("score", 0)
            if score < config.PINECONE_SCORE_THRESHOLD:
                continue

            disease = match["metadata"].get("disease", "unknown").lower()
            if disease not in unique_conditions or score > unique_conditions[disease]["score"]:
                unique_conditions[disease] = {"match": match, "score": score}

        filtered_matches = [entry["match"] for entry in unique_conditions.values()]
        logger.info(f"✅ Selected {len(filtered_matches)} unique conditions after filtering")
        
        return filtered_matches

    except Exception as e:
        logger.error(f"❌ Error querying Pinecone: {e}")
        raise HTTPException(status_code=503, detail=f"Failed to query medical database: {str(e)}")

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
    """Rank and format condition information"""
    try:
        conditions = []
        for match in matches[:5]:  # Top 5 only
            disease_name = match["metadata"].get("disease", "Unknown").title()
            description = await generate_condition_description(disease_name, symptoms)

            conditions.append(ConditionInfo(
                name=disease_name,
                description=description,
                file_citation="medical_database.json"
            ))

        return conditions

    except Exception as e:
        logger.error(f"Error ranking conditions: {e}")
        return []

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
        should_query = await should_query_pinecone_database(context, conversation_depth)

        logger.info(f"🗣️ Generating conversational response: {symptom_count} symptoms, depth {conversation_depth}, query: {should_query}")

        possible_conditions = []
        if should_query:
            query_text = f"Symptoms: {', '.join(symptoms)}"
            matches = await query_index(query_text, symptoms, context)
            possible_conditions = await rank_conditions(matches, symptoms, context)

        # Generate empathetic, conversational response
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
            
        elif should_query and possible_conditions:
            # Assessment with conditions
            symptoms_text = ", ".join(symptoms)
            text_parts = [
                f"Thank you for sharing all that information with me. Based on your symptoms - {symptoms_text} - I can see why you're concerned.",
                "",
                "Here are some conditions that could potentially match what you're experiencing:"
            ]
            
            for i, condition in enumerate(possible_conditions[:3], 1):
                text_parts.append(f"**{i}. {condition.name}**")
                text_parts.append(f"   {condition.description}")
                text_parts.append("")

            text_parts.extend([
                "Now, I want to be clear that this isn't a diagnosis - only a healthcare provider who can examine you properly can determine that.",
                "",
                "My recommendation would be to see a healthcare provider for a proper evaluation. They can do the necessary tests and examinations to give you a definitive answer.",
                "",
                "In the meantime, please monitor your symptoms and seek immediate care if they worsen."
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
                "Do you have any questions about these possibilities?",
                "Is there anything else you'd like to tell me about your symptoms?"
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
    """Main triage endpoint with full Pinecone integration"""
    try:
        description = request.description.strip()
        thread_id = request.thread_id

        logger.info(f"🚀 Triage request: '{description[:50]}...', thread: {thread_id}")

        # Step 1: Get OpenAI client
        try:
            client = await client_manager.get_openai_client()
            logger.info("✅ OpenAI client obtained successfully")
        except Exception as e:
            logger.error(f"❌ Failed to get OpenAI client: {e}")
            raise HTTPException(status_code=503, detail=f"OpenAI service unavailable: {str(e)}")

        # Step 2: Handle thread validation and creation
        try:
            if thread_id and await validate_thread(thread_id, client):
                logger.info(f"✅ Using existing thread: {thread_id}")
            else:
                logger.info("🆕 Creating new OpenAI thread...")
                new_thread = await client.beta.threads.create()
                thread_id = new_thread.id
                logger.info(f"✅ Created new thread: {thread_id}")
        except Exception as e:
            logger.error(f"❌ Thread management failed: {e}")
            raise HTTPException(status_code=503, detail=f"Thread management failed: {str(e)}")

        # Step 3: Add user message to thread
        try:
            logger.info("📝 Adding user message to thread...")
            await client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content=description
            )
            logger.info("✅ User message added successfully")
        except Exception as e:
            logger.error(f"❌ Failed to add message to thread: {e}")
            raise HTTPException(status_code=503, detail=f"Failed to add message: {str(e)}")

        # Step 4: Classify intent
        try:
            logger.info("🎯 Classifying user intent...")
            intent_label = await classify_intent_with_gpt(description, client)
            logger.info(f"✅ Intent classified as: {intent_label}")
        except Exception as e:
            logger.error(f"❌ Intent classification failed: {e}")
            # Fallback to treating as medical
            intent_label = "OTHER"
            logger.info("⚠️ Using fallback intent: OTHER")

        # Step 5: Handle different intents
        try:
            if intent_label == "GREETING":
                logger.info("👋 Generating greeting response...")
                return await generate_greeting_response(thread_id, client)
            elif intent_label == "THANKS":
                logger.info("🙏 Generating thanks response...")
                return await generate_thanks_response(thread_id, client)
            elif intent_label == "INFO_REQUEST":
                logger.info("ℹ️ Generating info response...")
                return await generate_info_request_response(thread_id, client)
            else:
                logger.info("🏥 Processing medical request...")
                # Handle medical requests with full Pinecone integration
                context = await get_thread_context(thread_id, client)
                is_emergency = is_red_flag(" ".join(context["user_messages"]), context["max_severity"])
                logger.info(f"🩺 Medical context: {len(context['all_symptoms'])} symptoms, emergency: {is_emergency}")
                return await generate_conversational_response(context, is_emergency, thread_id, client)
        except Exception as e:
            logger.error(f"❌ Response generation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Response generation failed: {str(e)}")

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error in triage endpoint: {e}")
        logger.error(f"❌ Error type: {type(e).__name__}")
        # Return a safe emergency response
        return TriageResponse(
            text=f"I'm experiencing technical difficulties. If this is urgent, please call {config.NIGERIA_EMERGENCY_HOTLINE} immediately.",
            possible_conditions=[],
            safety_measures=[f"Call {config.NIGERIA_EMERGENCY_HOTLINE} if urgent", "Seek immediate medical attention"],
            triage=TriageInfo(type="hospital", location="Unknown"),
            send_sos=True,
            follow_up_questions=[],
            thread_id=thread_id or "error",
            symptoms_count=0,
            should_query_pinecone=False
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

@app.get("/test-openai")
async def test_openai():
    """Test OpenAI connection specifically"""
    try:
        logger.info("🧪 Testing OpenAI connection...")
        client = await client_manager.get_openai_client()
        
        # Test 1: List models
        models = await client.models.list()
        logger.info(f"✅ OpenAI models retrieved: {len(models.data)} models")
        
        # Test 2: Simple completion
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say 'test successful'"}],
            max_tokens=10
        )
        result = response.choices[0].message.content.strip()
        logger.info(f"✅ OpenAI completion test: {result}")
        
        return {
            "status": "success",
            "models_count": len(models.data),
            "test_completion": result,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ OpenAI test failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )

@app.post("/test-triage")
async def test_triage():
    """Simplified triage test"""
    try:
        logger.info("🧪 Testing simplified triage...")
        
        # Test data
        test_request = TriageRequest(
            description="Hello, how are you?",
            thread_id=None
        )
        
        # Process the request
        result = await triage(test_request)
        logger.info("✅ Simplified triage test successful")
        
        return {
            "status": "success",
            "result": result.dict(),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Simplified triage test failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "failed",
                "error": str(e),
                "error_type": type(e).__name__,
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
            await client.models.list()
        except Exception as e:
            openai_status = f"unhealthy: {str(e)}"

        # Test Pinecone (REQUIRED)
        pinecone_status = "healthy"
        try:
            index = client_manager.get_pinecone_index()
            stats = index.describe_index_stats()
            logger.info(f"Pinecone health check: {stats.total_vector_count} vectors")
        except Exception as e:
            pinecone_status = f"unhealthy: {str(e)}"

        overall_status = "healthy" if openai_status == "healthy" and pinecone_status == "healthy" else "unhealthy"

        return HealthResponse(
            status=overall_status,
            timestamp=datetime.utcnow().isoformat(),
            version="1.0.0",
            services={
                "openai": openai_status,
                "pinecone": pinecone_status
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
        "pinecone_required": True
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
