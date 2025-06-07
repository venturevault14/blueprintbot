import os
import json
import logging
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
        self.SUPABASE_URL    = os.getenv("SUPABASE_URL")
        self.SUPABASE_KEY    = os.getenv("SUPABASE_KEY")

    def validate(self):
        missing = []
        for var in ["OPENAI_API_KEY", "SUPABASE_URL", "SUPABASE_KEY"]:
            if not getattr(self, var):
                missing.append(var)
        if missing:
            raise ValueError(f"Missing required env vars: {missing}")

config = Config()
try:
    config.validate()
    logger.info("Configuration validated successfully")
except ValueError as e:
    logger.error(f"Configuration error: {e}")
    raise

# -------------------------
# OpenAI Client Manager
# -------------------------
class ClientManager:
    async def get_openai_client(self) -> AsyncOpenAI:
        try:
            client = AsyncOpenAI(api_key=config.OPENAI_API_KEY, timeout=30.0)
            try:
                await client.models.list()
            except Exception as model_error:
                logger.warning(f"Model list check failed: {model_error}")
            return client
        except Exception as e:
            logger.error(f"Failed to create OpenAI client: {e}")
            raise HTTPException(status_code=503, detail=f"OpenAI connection failed: {e}")

client_manager = ClientManager()

# -------------------------
# Pydantic Models
# -------------------------
class RecipeContext(BaseModel):
    title:       Optional[str] = None
    description: Optional[str] = None
    raw:         Optional[str] = None  # full original recipe as text

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
        if isinstance(v, str) and v.lower() in {'', 'null', 'none'}:
            return None
        return v

class ChefResponse(BaseModel):
    text:                str
    recipe_id:           Optional[str] = None
    thread_id:           str
    follow_up_questions: Optional[List[str]] = []

# -------------------------
# Supabase persistence
# -------------------------
async def insert_chat(
    user_id: str,
    system_role: str,
    content: str,
    thread_id: str,
    recipe_id: Optional[str] = None
):
    url = f"{config.SUPABASE_URL}/rest/v1/chat"
    headers = {
        "apikey":        config.SUPABASE_KEY,
        "Authorization": f"Bearer {config.SUPABASE_KEY}",
        "Content-Type":  "application/json",
        "Prefer":        "return=minimal"
    }
    payload = {
        "user_id":   user_id,
        "system":    system_role,
        "content":   content,
        "thread_id": thread_id,
    }
    if recipe_id:
        payload["recipe_id"] = recipe_id

    async with httpx.AsyncClient() as client:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()

# -------------------------
# Helper to generate raw markdown
# -------------------------
async def generate_markdown(system_prompt: str, client: AsyncOpenAI) -> str:
    resp = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system_prompt}],
        temperature=0.7,
        max_tokens=1000
    )
    return resp.choices[0].message.content.strip()

# -------------------------
# FastAPI setup
# -------------------------
app = FastAPI(title="Pierre: Personal Chef Assistant API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/")
async def root():
    return {"message": "API is running!", "status": "healthy"}

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "pierre-chef-api"}

@app.post("/chef", response_model=ChefResponse)
async def chef_endpoint(req: ChefRequest):
    try:
        client = await client_manager.get_openai_client()

        # Thread handling
        use_existing = False
        if req.thread_id and req.thread_id.startswith("thread_"):
            try:
                await client.beta.threads.retrieve(thread_id=req.thread_id)
                use_existing = True
            except:
                pass
        thread_id = req.thread_id if use_existing else (await client.beta.threads.create()).id

        # Persist user message
        try:
            await insert_chat(req.user_id, "user", req.usermessage, thread_id, req.recipe_id)
        except Exception as e:
            logger.warning(f"Failed to persist user message: {e}")

        # Determine intent (force EDIT_RECIPE for replace/substitute)
        low = req.usermessage.lower()
        if any(k in low for k in ["replace", "substitute", "change", "modify"]):
            intent = "EDIT_RECIPE"
        else:
            if any(w in low for w in ["meal plan", "plan for"]):
                intent = "MEAL_PLAN_REQUEST"
            elif any(w in low for w in ["recipe", "cook", "make"]):
                intent = "RECIPE_REQUEST"
            elif any(w in low for w in ["hello", "hi", "hey"]):
                intent = "GREETING"
            elif any(w in low for w in ["thank", "thanks"]):
                intent = "THANKS"
            else:
                intent = "OTHER"

        follow: List[str] = []
        # Build response
        if intent == "GREETING":
            text = "Hello! 👋 I'm Pierre, your personal chef assistant. What can I help you with today?"
        elif intent == "THANKS":
            text = "You're very welcome! 😊"
        elif intent == "MEAL_PLAN_REQUEST":
            system = f"You are Pierre, generate a weekly meal plan in markdown based on: {req.usermessage}"
            text = await generate_markdown(system, client)
        elif intent == "RECIPE_REQUEST":
            system = f"You are Pierre, generate a recipe in markdown using: {req.usermessage}"
            text = await generate_markdown(system, client)
        elif intent == "EDIT_RECIPE":
            if not req.context or not req.context.raw:
                text = (
                    "To edit a recipe, please include the original recipe text in `context.raw`, "
                    "e.g. `{ \"context\": { \"raw\": \"Full recipe here\" } }`"
                )
            else:
                system = (
                    f"You are Pierre. Here is the original recipe:\n\n{req.context.raw}\n\n"
                    f"Apply these modifications: {req.usermessage}\n\n"
                    "Return the full updated recipe in markdown."
                )
                text = await generate_markdown(system, client)
        else:
            text = "I'm not sure I understand—could you clarify?"
            follow = [
                "Could you clarify what you'd like—meal plan, recipe, or edit?",
                "What ingredients or contexts should I know about?"
            ]

        # Persist bot response
        try:
            await insert_chat(req.user_id, "bot", text, thread_id, req.recipe_id)
        except Exception as e:
            logger.warning(f"Failed to persist bot message: {e}")

        return ChefResponse(
            text=text,
            recipe_id=req.recipe_id,
            thread_id=thread_id,
            follow_up_questions=follow
        )

    except Exception as e:
        logger.error(f"Chef endpoint error: {e}", exc_info=True)
        return ChefResponse(
            text=f"I encountered an error: {e}. Please try again.",
            recipe_id=req.recipe_id,
            thread_id=req.thread_id or "",
            follow_up_questions=[]
        )
