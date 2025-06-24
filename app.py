import os
import json
import logging
import asyncio
from typing import List, Optional, Dict, Any

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
        self.ASSISTANT_ID = os.getenv("ASSISTANT_ID")  # Your Blueprint Lab assistant ID
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
# Pydantic Models for Business Context
# -------------------------
class BusinessBlueprint(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    industry: Optional[str] = None
    target_market: Optional[str] = None
    revenue_model: Optional[str] = None
    startup_costs: Optional[str] = None
    monthly_costs: Optional[str] = None
    profit_margins: Optional[str] = None
    equipment_needed: Optional[List[str]] = []
    skills_required: Optional[List[str]] = []
    marketing_strategy: Optional[str] = None
    step_by_step_plan: Optional[List[str]] = []
    success_metrics: Optional[str] = None
    challenges: Optional[List[str]] = []
    resources: Optional[List[str]] = []
    affiliate_links: Optional[List[str]] = []
    raw_blueprint: Optional[str] = None  # Full raw text of the business plan

class UserBusinessProfile(BaseModel):
    experience_level: Optional[str] = None  # beginner, intermediate, advanced
    available_capital: Optional[str] = None
    time_commitment: Optional[str] = None  # part-time, full-time
    skills: List[str] = []
    interests: List[str] = []
    location: Optional[str] = None
    goals: List[str] = []
    risk_tolerance: Optional[str] = None  # low, medium, high

class BusinessRequest(BaseModel):
    user_id: str
    business_id: Optional[str] = None
    usermessage: str
    thread_id: Optional[str] = None
    blueprint: Optional[BusinessBlueprint] = None
    user_profile: UserBusinessProfile

    @field_validator('business_id', 'thread_id', mode='before')
    @classmethod
    def strip_null(cls, v):
        return None if isinstance(v, str) and v.lower() in {"", "null", "none"} else v

class BusinessResponse(BaseModel):
    text: str
    business_id: Optional[str] = None
    thread_id: str
    follow_up_questions: List[str] = []
    recommended_actions: List[str] = []

# -------------------------
# Supabase persistence
# -------------------------
async def insert_business_chat(user_id: str, system: str, content: str, thread_id: str, business_id: Optional[str] = None):
    url = f"{config.SUPABASE_URL}/rest/v1/chat"
    headers = {
        "apikey": config.SUPABASE_KEY,
        "Authorization": f"Bearer {config.SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal"
    }
    payload = {
        "user_id": user_id,
        "system": system,
        "content": content,
        "thread_id": thread_id
    }
    if business_id:
        payload["business_id"] = business_id

    async with httpx.AsyncClient() as client:
        r = await client.post(url, json=payload, headers=headers)
        r.raise_for_status()

# -------------------------
# Assistant Instructions Builder
# -------------------------
def build_business_assistant_instructions(user_profile: UserBusinessProfile, blueprint: Optional[BusinessBlueprint] = None) -> str:
    base_instructions = """You are the Blueprint Lab Business Assistant, an expert entrepreneurial mentor inspired by Chris Koerner's approach to launching successful side hustles. You specialize in:

🚀 **Business Plan Analysis**: Breaking down and optimizing business blueprints
💡 **Side Hustle Strategy**: Turning skills and interests into profitable ventures  
📊 **Market Validation**: Helping validate business ideas before launch
💰 **Revenue Optimization**: Maximizing profit margins and scaling strategies
⚡ **Execution Guidance**: Step-by-step implementation roadmaps
🛠️ **Resource Matching**: Connecting users with the right tools and affiliate resources

**Your Personality:**
- Direct and action-oriented (like Chris Koerner's "just do the thing" approach)
- Encouraging but realistic about challenges
- Focused on practical, executable advice
- Data-driven when discussing financials and metrics
- Emphasize speed to market and MVP concepts

**Your Expertise Areas:**
- Converting skills into profitable business models
- Low-cost startup strategies and bootstrapping
- Digital marketing and organic growth tactics
- Leveraging Chris Koerner's 75+ business experiences
- Affiliate marketing and resource monetization
- Risk assessment and mitigation strategies

**Response Format:**
- Always provide actionable next steps
- Use markdown formatting for plans and lists
- Include specific numbers when discussing costs/revenue
- Reference relevant tools from the Blueprint Lab toolkit when applicable
- Be concise but comprehensive
- Focus on rapid execution and testing"""

    # Add user profile information
    if any([user_profile.experience_level, user_profile.available_capital, user_profile.skills, user_profile.interests]):
        profile_text = "\n\n**User Profile:**\n"
        if user_profile.experience_level:
            profile_text += f"- Experience Level: {user_profile.experience_level}\n"
        if user_profile.available_capital:
            profile_text += f"- Available Capital: {user_profile.available_capital}\n"
        if user_profile.time_commitment:
            profile_text += f"- Time Commitment: {user_profile.time_commitment}\n"
        if user_profile.skills:
            profile_text += f"- Current Skills: {', '.join(user_profile.skills)}\n"
        if user_profile.interests:
            profile_text += f"- Interests: {', '.join(user_profile.interests)}\n"
        if user_profile.location:
            profile_text += f"- Location: {user_profile.location}\n"
        if user_profile.goals:
            profile_text += f"- Goals: {', '.join(user_profile.goals)}\n"
        if user_profile.risk_tolerance:
            profile_text += f"- Risk Tolerance: {user_profile.risk_tolerance}\n"
        base_instructions += profile_text

    # Add blueprint context if provided
    if blueprint:
        context_text = "\n\n**Current Business Blueprint:**\n"
        if blueprint.raw_blueprint:
            context_text += f"{blueprint.raw_blueprint}\n"
        else:
            if blueprint.title:
                context_text += f"**Title:** {blueprint.title}\n"
            if blueprint.description:
                context_text += f"**Description:** {blueprint.description}\n"
            if blueprint.industry:
                context_text += f"**Industry:** {blueprint.industry}\n"
            if blueprint.target_market:
                context_text += f"**Target Market:** {blueprint.target_market}\n"
            if blueprint.revenue_model:
                context_text += f"**Revenue Model:** {blueprint.revenue_model}\n"
            if blueprint.startup_costs:
                context_text += f"**Startup Costs:** {blueprint.startup_costs}\n"
            if blueprint.monthly_costs:
                context_text += f"**Monthly Operating Costs:** {blueprint.monthly_costs}\n"
            if blueprint.profit_margins:
                context_text += f"**Profit Margins:** {blueprint.profit_margins}\n"
            if blueprint.equipment_needed:
                context_text += f"**Equipment Needed:** {', '.join(blueprint.equipment_needed)}\n"
            if blueprint.skills_required:
                context_text += f"**Skills Required:** {', '.join(blueprint.skills_required)}\n"
            if blueprint.marketing_strategy:
                context_text += f"**Marketing Strategy:** {blueprint.marketing_strategy}\n"
            if blueprint.step_by_step_plan:
                context_text += f"**Implementation Steps:**\n"
                for i, step in enumerate(blueprint.step_by_step_plan, 1):
                    context_text += f"  {i}. {step}\n"
            if blueprint.success_metrics:
                context_text += f"**Success Metrics:** {blueprint.success_metrics}\n"
            if blueprint.challenges:
                context_text += f"**Potential Challenges:** {', '.join(blueprint.challenges)}\n"
            if blueprint.resources:
                context_text += f"**Recommended Resources:** {', '.join(blueprint.resources)}\n"
        base_instructions += context_text

    base_instructions += "\n\n**Always remember:** Focus on helping the user take immediate action toward launching their business, just like Chris Koerner's approach of 'doing the thing' rather than overthinking it."

    return base_instructions

# -------------------------
# Assistant Run Manager (same as before)
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
                logger.info("Run requires action - not implemented yet")
                return "I need to perform an action that's not yet supported. Please try a different request."
            
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
app = FastAPI(title="Blueprint Lab Business Assistant", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/")
async def root():
    return {"status": "healthy", "message": "Blueprint Lab Business Assistant is ready to help you launch your side hustle!"}

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "blueprint-lab-api", "version": "2.0.0"}

@app.post("/business", response_model=BusinessResponse)
async def business_endpoint(req: BusinessRequest):
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
        instructions = build_business_assistant_instructions(req.user_profile, req.blueprint)
        
        # Run the assistant conversation
        response_text = await run_assistant_conversation(
            client=client,
            thread_id=thread_id,
            message=req.usermessage,
            instructions=instructions,
            max_wait_time=30
        )

        # Generate context-aware follow-up questions and recommended actions
        follow_up_questions = []
        recommended_actions = []
        message_lower = req.usermessage.lower()
        
        # Context-aware follow-ups based on business topics
        if any(word in message_lower for word in ["start", "launch", "begin"]):
            follow_up_questions = [
                "What's your target launch timeline?",
                "Do you need help with market validation?",
                "Would you like me to break down the first 30 days?"
            ]
            recommended_actions = [
                "Create an MVP version to test the market",
                "Set up basic tracking for key metrics",
                "Join Chris Koerner's community for support"
            ]
        elif any(word in message_lower for word in ["marketing", "customers", "sales"]):
            follow_up_questions = [
                "What's your current marketing budget?",
                "Which social media platforms does your audience use?",
                "Do you need help with content creation?"
            ]
            recommended_actions = [
                "Start with organic social media marketing",
                "Test different pricing strategies",
                "Build an email list from day one"
            ]
        elif any(word in message_lower for word in ["costs", "money", "budget", "profit"]):
            follow_up_questions = [
                "Would you like help optimizing your cost structure?",
                "Should we explore additional revenue streams?",
                "Do you need guidance on pricing strategy?"
            ]
            recommended_actions = [
                "Track all expenses from the beginning",
                "Focus on high-margin activities first",
                "Consider affiliate opportunities from the toolkit"
            ]
        elif any(word in message_lower for word in ["scale", "grow", "expand"]):
            follow_up_questions = [
                "What growth metric are you focusing on?",
                "Are you ready to invest in paid advertising?",
                "Would you like help with hiring strategies?"
            ]
            recommended_actions = [
                "Automate repetitive processes",
                "Explore micro-influencer partnerships",
                "Document your processes for delegation"
            ]

        # Default questions if no specific context
        if not follow_up_questions:
            follow_up_questions = [
                "What specific aspect would you like to dive deeper into?",
                "Do you need help with the next immediate step?",
                "Would you like me to analyze any risks or challenges?"
            ]

        # Append follow-up questions to the response text
        if follow_up_questions:
            response_text += "\n\n---\n\n**Quick Questions for You:**\n"
            for i, question in enumerate(follow_up_questions, 1):
                response_text += f"{i}. {question}\n"

        if recommended_actions:
            response_text += "\n**Recommended Next Actions:**\n"
            for i, action in enumerate(recommended_actions, 1):
                response_text += f"• {action}\n"

        # Persist bot response
        try:
            await insert_business_chat(req.user_id, "bot", response_text, thread_id, req.business_id)
        except Exception as e:
            logger.warning(f"Failed to persist bot msg: {e}")

        return BusinessResponse(
            text=response_text,
            business_id=req.business_id,
            thread_id=thread_id,
            follow_up_questions=[],  # Empty since they're now in the text
            recommended_actions=[]   # Empty since they're now in the text
        )

    except Exception as e:
        logger.error(f"Business endpoint error: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        return BusinessResponse(
            text="I'm having trouble processing your request right now. Let's get back to building your business - could you try rephrasing your question?",
            business_id=req.business_id,
            thread_id=req.thread_id or "error",
            follow_up_questions=["Could you try asking in a different way?"],
            recommended_actions=["Check the Blueprint Lab toolkit for immediate resources"]
        )
