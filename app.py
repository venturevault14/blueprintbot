from openai import AsyncOpenAI
import uvicorn
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Request
import os
import json
import logging
import asyncio
import re
from datetime import datetime
from typing import List, Dict, Optional
from contextlib import asynccontextmanager

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded .env file successfully")
except ImportError:
    print("⚠️  python-dotenv not installed. Install with: pip install python-dotenv")
except Exception as e:
    print(f"⚠️  Could not load .env file: {e}")

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pinecone import (modern version)
try:
    from pinecone import Pinecone
    print("✅ Modern Pinecone client available")
except ImportError:
    print("❌ Modern Pinecone client not found. Please install: pip install pinecone")
    raise ImportError("Please install modern Pinecone: pip install pinecone")

# ========================================
# Configuration from Environment Variables
# ========================================

class Config:
    def __init__(self):
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        self.PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east1-gcp")
        self.ASSISTANT_ID = os.getenv(
            "ASSISTANT_ID", "asst_pAhSF6XJsj60efD9GEVdEG5n")
        self.EMBEDDING_MODEL = os.getenv(
            "EMBEDDING_MODEL", "text-embedding-ada-002")
        self.INDEX_NAME = os.getenv("INDEX_NAME", "triage-index")
        self.MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
        self.MIN_SYMPTOMS_FOR_PINECONE = int(
            os.getenv("MIN_SYMPTOMS_FOR_PINECONE", "3"))
        self.MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "3"))
        self.NIGERIA_EMERGENCY_HOTLINE = os.getenv("EMERGENCY_HOTLINE", "112")
        self.PINECONE_SCORE_THRESHOLD = float(
            os.getenv("PINECONE_SCORE_THRESHOLD", "0.8"))
        self.PORT = int(os.getenv("PORT", "8000"))

    def validate(self):
        """Validate required environment variables"""
        required_vars = ["OPENAI_API_KEY"]
        missing = [var for var in required_vars if not getattr(self, var)]
        if missing:
            raise ValueError(
                f"Missing required environment variables: {missing}")

config = Config()

# Validate configuration on startup
try:
    config.validate()
    logger.info("Configuration validated successfully")
except ValueError as e:
    logger.error(f"Configuration error: {e}")
    raise

# Debug environment variables
logger.info("=== Environment Variables Debug ===")
logger.info(f"OPENAI_API_KEY present: {'Yes' if config.OPENAI_API_KEY else 'No'}")
logger.info(f"OPENAI_API_KEY length: {len(config.OPENAI_API_KEY) if config.OPENAI_API_KEY else 0}")
logger.info(f"PINECONE_API_KEY present: {'Yes' if config.PINECONE_API_KEY else 'No'}")
logger.info(f"INDEX_NAME: {config.INDEX_NAME}")
logger.info(f"ASSISTANT_ID: {config.ASSISTANT_ID}")
logger.info("=== End Debug ===")

# ========================================
# Global Variables and Constants
# ========================================

# Red-flag symptoms for emergency detection
RED_FLAGS = [
    "bullet wound", "gunshot", "profuse bleeding", "crushing chest pain",
    "sudden shortness of breath", "loss of consciousness", "slurred speech", "seizure",
    "head trauma", "neck trauma", "high fever with stiff neck", "uncontrolled vomiting",
    "severe allergic reaction", "anaphylaxis", "difficulty breathing", "persistent cough with blood",
    "severe abdominal pain", "sudden vision loss", "chest tightness with sweating",
    "blood in urine", "inability to pass urine", "sharp abdominal pain", "intermenstrual bleeding"
]

# Global clients (initialized lazily)
openai_client = None
pinecone_client = None
index = None

# ========================================
# Lazy Initialization Functions
# ========================================

async def get_openai_client():
    """Get or initialize OpenAI client lazily"""
    global openai_client
    
    if openai_client is None:
        if not config.OPENAI_API_KEY:
            logger.error("OPENAI_API_KEY not configured")
            raise HTTPException(
                status_code=503,
                detail="OpenAI API key not configured"
            )
        
        try:
            logger.info("Lazy initializing OpenAI client...")
            openai_client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
            logger.info("✅ OpenAI client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise HTTPException(
                status_code=503,
                detail=f"Failed to initialize OpenAI client: {str(e)}"
            )
    
    return openai_client

def get_pinecone_index():
    """Get or initialize Pinecone index lazily"""
    global pinecone_client, index
    
    if index is None and config.PINECONE_API_KEY:
        try:
            logger.info("Lazy initializing Pinecone...")
            pinecone_client = Pinecone(api_key=config.PINECONE_API_KEY)
            index = pinecone_client.Index(name=config.INDEX_NAME)
            logger.info("✅ Pinecone initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            # Don't raise - just continue without Pinecone
    
    return index

# ========================================
# FastAPI App Configuration (Simple Startup)
# ========================================

app = FastAPI(
    title="Medical Triage Assistant API",
    description="AI-powered medical triage assistant for symptom assessment and condition suggestions",
    version="1.0.0"
)

# CORS middleware for web deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================================
# Pydantic Models
# ========================================

class TriageRequest(BaseModel):
    description: str = Field(..., min_length=1, max_length=1000,
                             description="User's symptom description")
    thread_id: Optional[str] = Field(
        None, description="Thread ID for conversation continuity")

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

class ErrorResponse(BaseModel):
    error: str
    message: str
    timestamp: str

# ========================================
# Utility Functions
# ========================================

async def validate_thread(thread_id: str) -> bool:
    """Check if an OpenAI thread ID is valid"""
    try:
        client = await get_openai_client()
        await client.beta.threads.retrieve(thread_id=thread_id)
        logger.info(f"Thread {thread_id} validated successfully")
        return True
    except Exception as e:
        logger.error(f"Thread validation failed for {thread_id}: {e}")
        return False

async def get_thread_context(thread_id: str) -> Dict:
    """Retrieve and analyze thread context"""
    try:
        client = await get_openai_client()
        messages = await client.beta.threads.messages.list(
            thread_id=thread_id,
            order='asc'
        )

        user_messages = []
        assistant_count = 0
        all_symptoms = []
        max_severity = 0

        for msg in messages.data:
            if msg.role == "user":
                content = ""
                if msg.content and hasattr(msg.content[0], "text"):
                    content = msg.content[0].text.value
                if content:
                    user_messages.append(content)
                    symptom_data = await extract_symptoms_comprehensive(content)
                    all_symptoms.extend(symptom_data["symptoms"])
                    max_severity = max(max_severity, symptom_data["severity"])
            elif msg.role == "assistant":
                assistant_count += 1

        # Deduplicate symptoms
        unique_symptoms = []
        seen = set()
        for symptom in all_symptoms:
            symptom_clean = symptom.lower().strip()
            if symptom_clean not in seen:
                seen.add(symptom_clean)
                unique_symptoms.append(symptom_clean)

        return {
            "user_messages": user_messages,
            "assistant_responses": assistant_count,
            "all_symptoms": unique_symptoms,
            "max_severity": max_severity
        }

    except Exception as e:
        logger.error(f"Error retrieving thread context for {thread_id}: {e}")
        return {
            "user_messages": [],
            "assistant_responses": 0,
            "all_symptoms": [],
            "max_severity": 0
        }

async def extract_symptoms_comprehensive(description: str) -> Dict:
    """Extract symptoms, duration, and severity from description"""
    try:
        client = await get_openai_client()
        description_lower = description.lower()

        # Extract duration using regex
        duration = None
        duration_patterns = [
            r"(since|started|began)\s*(yesterday|last night|today|[0-9]+\s*(day|hour|minute|week|month)s?\s*ago)",
            r"for\s*(about)?\s*([0-9]+\s*(minute|hour|day|week|month)s?)",
            r"(last|past)\s*([0-9]+\s*(minute|hour|day|week|month)s?)"
        ]
        for pattern in duration_patterns:
            match = re.search(pattern, description_lower)
            if match:
                duration = match.group(0)
                break

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
                break

        # Check for numeric pain scale
        pain_scale_match = re.search(r"pain\s+(\d+)/10", description_lower)
        if pain_scale_match:
            severity = int(pain_scale_match.group(1))

        # Extract symptoms using GPT
        prompt = (
            "Extract specific medical symptoms from this text. Return a JSON array of symptom strings. "
            "Be precise and use standard medical terminology. "
            "Example: [\"chest pain\", \"shortness of breath\", \"nausea\"]\n\n"
            f"Text: \"{description}\""
        )

        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Extract symptoms now."}
            ],
            temperature=0
        )

        try:
            symptoms = json.loads(response.choices[0].message.content.strip())
            if not isinstance(symptoms, list):
                symptoms = []
        except json.JSONDecodeError:
            logger.warning("GPT returned invalid JSON for symptoms, using fallback")
            symptoms = []

        # Normalize symptoms
        unique_symptoms = []
        seen = set()
        for symptom in symptoms:
            symptom_clean = symptom.lower().strip()
            if symptom_clean and symptom_clean not in seen:
                seen.add(symptom_clean)
                unique_symptoms.append(symptom_clean)

        logger.info(f"Extracted: symptoms={unique_symptoms}, duration={duration}, severity={severity}")
        return {"symptoms": unique_symptoms, "duration": duration, "severity": severity}

    except Exception as e:
        logger.error(f"Error extracting symptoms: {e}")
        return {"symptoms": [], "duration": None, "severity": 0}

def is_red_flag(text: str, severity: int = 0) -> bool:
    """Check for emergency red flags"""
    text_lower = text.lower()

    # Check explicit red flags
    for flag in RED_FLAGS:
        if flag in text_lower:
            logger.info(f"Red flag detected: {flag}")
            return True

    # Check high severity with critical symptoms
    if severity >= 8:
        critical_symptoms = ["chest pain", "abdominal pain", "breathing", "consciousness"]
        if any(symptom in text_lower for symptom in critical_symptoms):
            logger.info(f"High severity ({severity}) with critical symptom detected")
            return True

    return False

async def should_query_pinecone_database(context: Dict) -> bool:
    """Determine if we should query the medical database"""
    # Skip if Pinecone is not available
    pinecone_index = get_pinecone_index()
    if not pinecone_index:
        logger.info("Pinecone not available - skipping database query")
        return False
        
    all_symptoms = context.get("all_symptoms", [])
    symptom_count = len(all_symptoms)
    max_severity = context.get("max_severity", 0)
    full_text = " ".join(context.get("user_messages", [])).lower()

    # Check for emergency
    if is_red_flag(full_text, max_severity):
        logger.info("Emergency detected - querying Pinecone")
        return True

    # Check symptom threshold
    if symptom_count >= config.MIN_SYMPTOMS_FOR_PINECONE:
        logger.info(f"Sufficient symptoms ({symptom_count}) - querying Pinecone")
        return True

    # Check for explicit condition requests
    condition_keywords = [
        "what might be", "what could be", "what is", "infection", "condition",
        "disease", "diagnosis", "what's wrong", "what do i have"
    ]
    if any(keyword in full_text for keyword in condition_keywords):
        logger.info("Explicit condition request - querying Pinecone")
        return True

    logger.info(f"Not querying Pinecone: {symptom_count} symptoms, no explicit request")
    return False

async def get_embeddings(texts: List[str]) -> Optional[List[List[float]]]:
    """Get embeddings from OpenAI with retry logic"""
    client = await get_openai_client()
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

async def classify_intent_with_gpt(message: str) -> str:
    """Send a quick prompt to GPT to classify user intent"""
    prompt = (
        "You are a classifier. Label the user message with exactly one word: "
        "GREETING, THANKS, INFO_REQUEST, SYMPTOM_REPORT, or OTHER.\n\n"
        f"User: \"{message.strip()}\"\n"
        "Answer with exactly one label, and nothing else."
    )

    try:
        client = await get_openai_client()
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=4,
        )
        label = response.choices[0].message.content.strip().upper()

        # Guard against unexpected outputs
        valid_labels = {"GREETING", "THANKS", "INFO_REQUEST", "SYMPTOM_REPORT", "OTHER"}
        if label not in valid_labels:
            return "OTHER"
        return label

    except Exception as e:
        logger.error(f"Error in intent classification: {e}")
        return "OTHER"

async def generate_greeting_response(thread_id: str) -> TriageResponse:
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

    try:
        client = await get_openai_client()
        await client.beta.threads.messages.create(
            thread_id=thread_id,
            role="assistant",
            content=json.dumps(response_data)
        )
    except Exception as e:
        logger.error(f"Error adding greeting response to thread {thread_id}: {e}")

    return TriageResponse(**response_data)

async def generate_thanks_response(thread_id: str) -> TriageResponse:
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

    try:
        client = await get_openai_client()
        await client.beta.threads.messages.create(
            thread_id=thread_id,
            role="assistant",
            content=json.dumps(response_data)
        )
    except Exception as e:
        logger.error(f"Error adding thanks response to thread {thread_id}: {e}")

    return TriageResponse(**response_data)

async def generate_info_request_response(thread_id: str) -> TriageResponse:
    """Generate informative but friendly response about capabilities"""
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

    try:
        client = await get_openai_client()
        await client.beta.threads.messages.create(
            thread_id=thread_id,
            role="assistant",
            content=json.dumps(response_data)
        )
    except Exception as e:
        logger.error(f"Error adding info response to thread {thread_id}: {e}")

    return TriageResponse(**response_data)

# ========================================
# API Endpoints
# ========================================

@app.post("/triage", response_model=TriageResponse)
async def triage(request: TriageRequest):
    """Main triage endpoint with enhanced intent classification"""
    try:
        # Get OpenAI client (initialize if needed)
        client = await get_openai_client()
        
        description = request.description.strip()
        thread_id = request.thread_id

        logger.info(f"Triage request: '{description[:50]}...', thread: {thread_id}")

        # Validate or create thread
        if not thread_id or not await validate_thread(thread_id):
            try:
                new_thread = await client.beta.threads.create()
                thread_id = new_thread.id
                logger.info(f"Created new thread: {thread_id}")
            except Exception as e:
                logger.error(f"Failed to create OpenAI thread: {e}")
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
        except Exception as e:
            logger.error(f"Failed to add message to thread: {e}")
            # Continue anyway - we can still provide a response

        # Quick GPT-based intent classification
        intent_label = await classify_intent_with_gpt(description)
        logger.info(f"Intent classified as: {intent_label}")

        # Handle conversational intents with friendly responses
        if intent_label == "GREETING":
            return await generate_greeting_response(thread_id)
        elif intent_label == "THANKS":
            return await generate_thanks_response(thread_id)
        elif intent_label == "INFO_REQUEST":
            return await generate_info_request_response(thread_id)
        else:
            # For now, just return a simple medical response
            return TriageResponse(
                text="I understand you're looking for medical help. Please describe your symptoms and I'll do my best to assist you.",
                possible_conditions=[],
                safety_measures=["Stay hydrated", "Rest as needed"],
                triage=TriageInfo(type="clinic", location="Unknown"),
                send_sos=False,
                follow_up_questions=["What symptoms are you experiencing?"],
                thread_id=thread_id,
                symptoms_count=0,
                should_query_pinecone=False
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in triage endpoint: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Test OpenAI connection
        openai_status = "not_tested"
        try:
            client = await get_openai_client()
            openai_status = "healthy"
        except Exception:
            openai_status = "unhealthy"

        # Test Pinecone connection
        pinecone_status = "not_tested"
        try:
            pinecone_index = get_pinecone_index()
            if pinecone_index:
                pinecone_status = "healthy"
            else:
                pinecone_status = "unavailable"
        except Exception:
            pinecone_status = "unhealthy"

        return HealthResponse(
            status="healthy" if openai_status == "healthy" else "degraded",
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

@app.get("/debug-env")
async def debug_env():
    """Debug endpoint to check environment variables"""
    try:
        client = await get_openai_client()
        client_status = "initialized"
    except:
        client_status = "failed_to_initialize"
    
    return {
        "openai_key_present": bool(config.OPENAI_API_KEY),
        "openai_key_length": len(config.OPENAI_API_KEY) if config.OPENAI_API_KEY else 0,
        "pinecone_key_present": bool(config.PINECONE_API_KEY),
        "index_name": config.INDEX_NAME,
        "assistant_id": config.ASSISTANT_ID,
        "openai_client_status": client_status
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Medical Triage Assistant API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

# ========================================
# Error Handlers
# ========================================

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error="ValueError",
            message=str(exc),
            timestamp=datetime.utcnow().isoformat()
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="InternalServerError",
            message="An unexpected error occurred",
            timestamp=datetime.utcnow().isoformat()
        ).dict()
    )

# ========================================
# Local Development Only
# ========================================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=config.PORT)
