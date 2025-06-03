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
        """Get OpenAI client - initialize if needed"""
        if self._openai_client is None:
            if not config.OPENAI_API_KEY:
                raise HTTPException(status_code=503, detail="OpenAI API key not configured")
            self._openai_client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
            logger.info("✅ OpenAI client initialized")
        return self._openai_client

    def get_pinecone_index(self):
        """Get Pinecone index - initialize if needed (ALWAYS REQUIRED)"""
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

async def validate_thread(thread_id: str) -> bool:
    """Check if an OpenAI thread ID is valid"""
    if not thread_id or not thread_id.strip():
        return False
    
    try:
        client = await client_manager.get_openai_client()
        await client.beta.threads.retrieve(thread_id=thread_id.strip())
        logger.info(f"✅ Thread {thread_id} validated successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Thread validation failed for {thread_id}: {e}")
        return False

async def get_thread_context(thread_id: str) -> Dict:
    """Retrieve and analyze thread context"""
    try:
        client = await client_manager.get_openai_client()
        messages = await client.beta.threads.messages.list(
            thread_id=thread_id,
            order='asc',
            limit=100
        )

        user_messages = []
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

        # Deduplicate symptoms
        unique_symptoms = list(dict.fromkeys([s.lower().strip() for s in all_symptoms if s.strip()]))

        return {
            "user_messages": user_messages,
            "all_symptoms": unique_symptoms,
            "max_severity": max_severity
        }

    except Exception as e:
        logger.error(f"Error retrieving thread context for {thread_id}: {e}")
        return {"user_messages": [], "all_symptoms": [], "max_severity": 0}

async def extract_symptoms_comprehensive(description: str) -> Dict:
    """Extract symptoms, duration, and severity from description"""
    try:
        client = await client_manager.get_openai_client()
        description_lower = description.lower()

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
            logger.warning("GPT returned invalid JSON for symptoms")
            symptoms = []

        # Clean and deduplicate symptoms
        unique_symptoms = list(dict.fromkeys([s.lower().strip() for s in symptoms if s.strip()]))

        return {"symptoms": unique_symptoms, "severity": severity}

    except Exception as e:
        logger.error(f"Error extracting symptoms: {e}")
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

async def classify_intent_with_gpt(message: str) -> str:
    """Classify user intent"""
    prompt = (
        "You are a classifier. Label the user message with exactly one word: "
        "GREETING, THANKS, INFO_REQUEST, SYMPTOM_REPORT, or OTHER.\n\n"
        f"User: \"{message.strip()}\"\n"
        "Answer with exactly one label, and nothing else."
    )

    try:
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

async def should_query_pinecone_database(context: Dict) -> bool:
    """Determine if we should query the medical database (ALWAYS TRUE when symptoms present)"""
    all_symptoms = context.get("all_symptoms", [])
    symptom_count = len(all_symptoms)
    max_severity = context.get("max_severity", 0)
    full_text = " ".join(context.get("user_messages", [])).lower()

    # Always query for emergencies
    if is_red_flag(full_text, max_severity):
        logger.info("🚨 Emergency detected - querying Pinecone")
        return True

    # Always query when we have enough symptoms
    if symptom_count >= config.MIN_SYMPTOMS_FOR_PINECONE:
        logger.info(f"🔍 Sufficient symptoms ({symptom_count}) - querying Pinecone")
        return True

    # Always query for explicit condition requests
    condition_keywords = [
        "what might be", "what could be", "what is", "infection", "condition",
        "disease", "diagnosis", "what's wrong", "what do i have"
    ]
    if any(keyword in full_text for keyword in condition_keywords):
        logger.info("🔍 Explicit condition request - querying Pinecone")
        return True

    logger.info(f"ℹ️ Not querying Pinecone: {symptom_count} symptoms, no explicit request")
    return False

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

async def add_message_to_thread(thread_id: str, content: str, role: str = "assistant"):
    """Helper to add message to thread"""
    try:
        client = await client_manager.get_openai_client()
        await client.beta.threads.messages.create(
            thread_id=thread_id,
            role=role,
            content=content
        )
    except Exception as e:
        logger.error(f"Error adding message to thread {thread_id}: {e}")

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

    await add_message_to_thread(thread_id, json.dumps(response_data))
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

    await add_message_to_thread(thread_id, json.dumps(response_data))
    return TriageResponse(**response_data)

async def generate_info_request_response(thread_id: str) -> TriageResponse:
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

    await add_message_to_thread(thread_id, json.dumps(response_data))
    return TriageResponse(**response_data)

async def generate_medical_response(context: Dict, is_emergency: bool, thread_id: str) -> TriageResponse:
    """Generate comprehensive medical response with Pinecone integration"""
    try:
        symptoms = context["all_symptoms"]
        symptom_count = len(symptoms)
        should_query = await should_query_pinecone_database(context)

        possible_conditions = []
        if should_query:
            query_text = f"Symptoms: {', '.join(symptoms)}"
            matches = await query_index(query_text, symptoms, context)
            possible_conditions = await rank_conditions(matches, symptoms, context)

        symptoms_text = ", ".join(symptoms) if symptoms else "your symptoms"
        
        text_parts = [f"Based on your symptoms ({symptoms_text}), here's my assessment:\n"]

        if possible_conditions:
            text_parts.append("**Possible Conditions:**")
            for i, condition in enumerate(possible_conditions[:3], 1):
                text_parts.append(f"{i}. **{condition.name}**: {condition.description}")
            text_parts.append("")

        safety_measures = [
            "Stay hydrated by drinking plenty of water",
            "Get adequate rest to help your body heal",
            "Monitor your symptoms for any changes"
        ]

        if is_emergency:
            text_parts.append("🚨 **This appears to be an emergency!**")
            text_parts.append(f"Call {config.NIGERIA_EMERGENCY_HOTLINE} immediately or go to the nearest hospital.")
            safety_measures = [
                f"Call {config.NIGERIA_EMERGENCY_HOTLINE} now",
                "Do not drive yourself to the hospital",
                "Stay calm and follow emergency operator instructions"
            ]
        else:
            text_parts.append("**Recommended Action:**")
            text_parts.append("Please see a healthcare provider for proper evaluation and treatment.")

        text_parts.append("\n*This is not a medical diagnosis. Please consult a healthcare professional.*")

        response_data = {
            "text": "\n".join(text_parts),
            "possible_conditions": possible_conditions,
            "safety_measures": safety_measures,
            "triage": TriageInfo(
                type="hospital" if is_emergency else "clinic",
                location="Unknown"
            ),
            "send_sos": is_emergency,
            "follow_up_questions": ["Do you have any other symptoms?", "When did your symptoms start?"],
            "thread_id": thread_id,
            "symptoms_count": symptom_count,
            "should_query_pinecone": should_query
        }

        await add_message_to_thread(thread_id, json.dumps(response_data))
        return TriageResponse(**response_data)

    except Exception as e:
        logger.error(f"Error in generate_medical_response: {e}")
        # Emergency fallback
        return TriageResponse(
            text=f"I'm experiencing a technical issue. If this is urgent, please call {config.NIGERIA_EMERGENCY_HOTLINE}.",
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

        # Handle thread validation and creation
        client = await client_manager.get_openai_client()
        
        if thread_id and await validate_thread(thread_id):
            logger.info(f"✅ Using existing thread: {thread_id}")
        else:
            new_thread = await client.beta.threads.create()
            thread_id = new_thread.id
            logger.info(f"🆕 Created new thread: {thread_id}")

        # Add user message to thread
        await client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=description
        )

        # Classify intent
        intent_label = await classify_intent_with_gpt(description)
        logger.info(f"🎯 Intent classified as: {intent_label}")

        # Handle different intents
        if intent_label == "GREETING":
            return await generate_greeting_response(thread_id)
        elif intent_label == "THANKS":
            return await generate_thanks_response(thread_id)
        elif intent_label == "INFO_REQUEST":
            return await generate_info_request_response(thread_id)
        else:
            # Handle medical requests with full Pinecone integration
            context = await get_thread_context(thread_id)
            is_emergency = is_red_flag(" ".join(context["user_messages"]), context["max_severity"])
            return await generate_medical_response(context, is_emergency, thread_id)

    except Exception as e:
        logger.error(f"❌ Error in triage endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

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
