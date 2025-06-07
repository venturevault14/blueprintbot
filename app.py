import os
import json
import logging
import asyncio
from typing import List, Optional, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from openai import AsyncOpenAI
import httpx

# -------------------------
# Logging Configuration
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------
# Configuration
# -------------------------
class Config:
    def __init__(self):
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.ASSISTANT_ID = os.getenv("ASSISTANT_ID")  # Your Pierre assistant ID
        self.SUPABASE_URL = os.getenv("SUPABASE_URL")
        self.SUPABASE_KEY = os.getenv("SUPABASE_KEY")

    def validate(self):
        missing = [v for v in ["OPENAI_API_KEY","ASSISTANT_ID","SUPABASE_URL","SUPABASE_KEY"] if not getattr(self, v)]
        if missing:
            raise ValueError(f"Missing required env vars: {missing}")

config = Config()
config.validate()
logger.info("Configuration validated successfully")

# -------------------------
# OpenAI Client Manager
# -------------------------
class ClientManager:
    async def get_openai_client(self) -> AsyncOpenAI:
        try:
            client = AsyncOpenAI(api_key=config.OPENAI_API_KEY, timeout=60.0)
            try:
                await client.models.list()
            except Exception as e:
                logger.warning(f"Model list check failed: {e}")
            return client
        except Exception as e:
            logger.error(f"OpenAI client init failed: {e}")
            raise HTTPException(status_code=503, detail=str(e))

client_manager = ClientManager()

# -------------------------
# Pydantic Models
# -------------------------
class RecipeContext(BaseModel):
    raw:         Optional[str] = None
    title:       Optional[str] = None
    description: Optional[str] = None
    ingredients: Optional[str] = None
    preparation: Optional[str] = None

class UserProfile(BaseModel):
    dietary_preferences: List[str] = []
    allergies:           List[str] = []
    intolerances:        List[str] = []
    other_flags:         List[str] = []

class ChefRequest(BaseModel):
    user_id:     str
    recipe_id:   Optional[str] = None
    usermessage: str
    thread_id:   Optional[str] = None
    context:     Optional[RecipeContext] = None
    profile:     UserProfile

    @field_validator('recipe_id', 'thread_id', mode='before')
    @classmethod
    def strip_null(cls, v):
        return None if isinstance(v, str) and v.lower() in {"", "null", "none"} else v

class ChefResponse(BaseModel):
    text:                str
    recipe_id:           Optional[str] = None
    thread_id:           str
    follow_up_questions: List[str] = []

# -------------------------
# Supabase persistence
# -------------------------
async def insert_chat(user_id: str, system: str, content: str, thread_id: str, recipe_id: Optional[str] = None):
    url = f"{config.SUPABASE_URL}/rest/v1/chat"
    headers = {
        "apikey":        config.SUPABASE_KEY,
        "Authorization": f"Bearer {config.SUPABASE_KEY}",
        "Content-Type":  "application/json",
        "Prefer":        "return=minimal"
    }
    payload = {
        "user_id":   user_id,
        "system":    system,
        "content":   content,
        "thread_id": thread_id
    }
    if recipe_id:
        payload["recipe_id"] = recipe_id

    async with httpx.AsyncClient() as client:
        r = await client.post(url, json=payload, headers=headers)
        r.raise_for_status()

# -------------------------
# Assistant Instructions Builder
# -------------------------
def build_assistant_instructions(user_profile: UserProfile, context: Optional[RecipeContext] = None) -> str:
    base_instructions = """You are Pierre, a warm, knowledgeable, and helpful personal chef assistant. You specialize in:

🍳 **Recipe Creation & Modification**: Creating new recipes and modifying existing ones
📋 **Meal Planning**: Weekly meal plans, special occasions, dietary needs
🥗 **Dietary Guidance**: Accommodating restrictions, preferences, and health goals
👨‍🍳 **Cooking Tips**: Techniques, substitutions, and kitchen wisdom

**Your Personality:**
- Warm and enthusiastic about food
- Patient and encouraging, especially with beginners
- Knowledgeable but not overwhelming
- Practical and solution-oriented
- Use emojis occasionally to keep things friendly

**Response Format:**
- Always provide clear, actionable advice
- Use markdown formatting for recipes (headings, lists, etc.)
- Be concise but thorough
- Ask follow-up questions when helpful
- Suggest alternatives when needed"""

    # Add user profile information
    if user_profile.dietary_preferences or user_profile.allergies or user_profile.intolerances:
        profile_text = "\n\n**User Profile:**\n"
        if user_profile.dietary_preferences:
            profile_text += f"- Dietary preferences: {', '.join(user_profile.dietary_preferences)}\n"
        if user_profile.allergies:
            profile_text += f"- Allergies: {', '.join(user_profile.allergies)} (NEVER suggest these ingredients)\n"
        if user_profile.intolerances:
            profile_text += f"- Intolerances: {', '.join(user_profile.intolerances)} (avoid these ingredients)\n"
        if user_profile.other_flags:
            profile_text += f"- Other notes: {', '.join(user_profile.other_flags)}\n"
        base_instructions += profile_text

    # Add context if provided
    if context and (context.raw or (context.ingredients and context.preparation)):
        context_text = "\n\n**Current Recipe Context:**\n"
        if context.raw:
            context_text += f"{context.raw}\n"
        else:
            if context.title:
                context_text += f"**Title:** {context.title}\n"
            if context.description:
                context_text += f"**Description:** {context.description}\n"
            if context.ingredients:
                context_text += f"**Ingredients:** {context.ingredients}\n"
            if context.preparation:
                context_text += f"**Preparation:** {context.preparation}\n"
        base_instructions += context_text

    return base_instructions

# -------------------------
# Assistant Run Manager
# -------------------------
async def run_assistant_conversation(
    client: AsyncOpenAI,
    thread_id: str,
    message: str,
    instructions: str,
    max_wait_time: int = 30
) -> str:
    """Run the assistant and wait for completion"""
    
    try:
        # Add the user message to the thread
        await client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=message
        )

        # Create and run the assistant
        run = await client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=config.ASSISTANT_ID,
            instructions=instructions
        )

        # Wait for completion
        start_time = asyncio.get_event_loop().time()
        while True:
            current_time = asyncio.get_event_loop().time()
            if current_time - start_time > max_wait_time:
                logger.warning(f"Assistant run timed out after {max_wait_time}s")
                # Cancel the run
                try:
                    await client.beta.threads.runs.cancel(thread_id=thread_id, run_id=run.id)
                except:
                    pass
                return "I'm taking longer than usual to respond. Could you try rephrasing your request?"

            # Check run status
            run_status = await client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
            
            if run_status.status == "completed":
                break
            elif run_status.status in ["failed", "cancelled", "expired"]:
                logger.error(f"Assistant run failed with status: {run_status.status}")
                return "I encountered an issue processing your request. Please try again."
            elif run_status.status == "requires_action":
                # Handle function calls if needed (for future expansion)
                logger.info("Run requires action - not implemented yet")
                return "I need to perform an action that's not yet supported. Please try a different request."
            
            # Wait before checking again
            await asyncio.sleep(0.5)

        # Get the assistant's response
        messages = await client.beta.threads.messages.list(
            thread_id=thread_id,
            order="desc",
            limit=1
        )
        
        if messages.data and messages.data[0].role == "assistant":
            response_content = messages.data[0].content[0]
            if hasattr(response_content, 'text'):
                return response_content.text.value
        
        return "I couldn't generate a proper response. Please try again."

    except Exception as e:
        logger.error(f"Assistant conversation error: {e}")
        return f"I encountered an error: {str(e)}. Please try again."

# -------------------------
# FastAPI setup
# -------------------------
app = FastAPI(title="Pierre Chef Assistant", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/")
async def root():
    return {"status":"healthy", "message": "Pierre Chef Assistant is ready to help!"}

@app.get("/health")
async def health():
    return {"status":"healthy","service":"pierre-chef-api", "version": "2.0.0"}

@app.post("/chef", response_model=ChefResponse)
async def chef_endpoint(req: ChefRequest):
    try:
        client = await client_manager.get_openai_client()

        # Thread management
        use_existing = bool(req.thread_id and req.thread_id.startswith("thread_"))
        if use_existing:
            try:
                await client.beta.threads.retrieve(thread_id=req.thread_id)
                logger.info(f"Using existing thread: {req.thread_id}")
            except Exception as e:
                logger.warning(f"Failed to retrieve thread {req.thread_id}: {e}")
                use_existing = False
        
        thread_id = req.thread_id if use_existing else (await client.beta.threads.create()).id
        logger.info(f"Thread ID: {thread_id}")

        # Build personalized instructions
        instructions = build_assistant_instructions(req.profile, req.context)
        
        # Run the assistant conversation
        response_text = await run_assistant_conversation(
            client=client,
            thread_id=thread_id,
            message=req.usermessage,
            instructions=instructions,
            max_wait_time=30
        )

        # Generate follow-up questions based on the conversation
        follow_up_questions = []
        message_lower = req.usermessage.lower()
        
        # Context-aware follow-ups
        if any(word in message_lower for word in ["recipe", "cook", "make"]):
            follow_up_questions = [
                "Would you like any modifications to this recipe?",
                "Do you need substitutions for any ingredients?",
                "Would you like cooking tips for this dish?"
            ]
        elif "meal plan" in message_lower:
            follow_up_questions = [
                "Would you like me to adjust this for your schedule?",
                "Should I include prep times and shopping lists?",
                "Any specific cuisines you'd like to focus on?"
            ]
        elif any(word in message_lower for word in ["substitute", "replace", "change"]):
            follow_up_questions = [
                "Are there any other ingredients you'd like to modify?",
                "Would you like tips for the best substitution ratios?",
                "Should I suggest complementary flavor adjustments?"
            ]

        # Append follow-up questions to the response text
        if follow_up_questions:
            response_text += "\n\n---\n\n**Quick Questions for You:**\n"
            for i, question in enumerate(follow_up_questions, 1):
                response_text += f"{i}. {question}\n"

        # Persist bot response (with follow-ups included)
        try:
            await insert_chat(req.user_id, "bot", response_text, thread_id, req.recipe_id)
        except Exception as e:
            logger.warning(f"Failed to persist bot msg: {e}")

        return ChefResponse(
            text=response_text,
            recipe_id=req.recipe_id,
            thread_id=thread_id,
            follow_up_questions=[]  # Empty since they're now in the text
        )

    except Exception as e:
        logger.error(f"Chef endpoint error: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        # Return a helpful error response
        return ChefResponse(
            text="I'm having trouble processing your request right now. Please try again, or rephrase your question.",
            recipe_id=req.recipe_id,
            thread_id=req.thread_id or "error",
            follow_up_questions=["Could you try asking in a different way?"]
        )
