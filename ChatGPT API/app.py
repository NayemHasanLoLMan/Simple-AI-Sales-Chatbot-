from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.middleware.sessions import SessionMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from langchain_ollama import OllamaLLM
from starlette.responses import FileResponse
import logging
from fastapi.middleware.cors import CORSMiddleware
import json
from datetime import datetime
from pathlib import Path
import asyncio

# Set up logging with more detailed configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("sales_assistant")

# Initialize FastAPI app
app = FastAPI(title="Automotive Sales Assistant")

# CORS and middleware configuration
app.add_middleware(
    SessionMiddleware, 
    secret_key="your-secret-key-here",
    max_age=3600  # Session timeout in seconds
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Create data directory if it doesn't exist
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Improved system context with specific constraints for shorter responses
SYSTEM_CONTEXT = """
You are a helpful and knowledgeable car dealership virtual assistant. Your primary goal is to assist customers with their automotive needs while providing excellent customer service. Follow these guidelines:

1. INITIAL GREETING:
- Introduce yourself as Alex a virtual assistant
- Ask how you can help the customer today
- Offer main categories: New Cars, Used Cars, Service Department, or General Inquiries

2. RESPONSE HANDLING:
For New Car Inquiries:
- Ask about preferred vehicle type (SUV, Sedan, Truck, etc.)
- Inquire about key features they're looking for
- Provide current inventory information
- Offer to schedule test drives
- Share information about current promotions and financing options

For Used Car Inquiries:
- Ask about budget range
- Inquire about preferred vehicle type
- Offer to show available inventory within specifications
- Provide vehicle history report information
- Discuss financing options for used vehicles

For Service Department:
- Allow scheduling of service appointments
- Provide maintenance package information
- Offer service status updates
- Share basic maintenance tips
- Connect to service advisor if needed

For General Inquiries:
- Provide dealership hours and location
- Share contact information
- Discuss financing options
- Answer FAQs about dealership policies

3. COMMUNICATION STYLE:
- Maintain a professional yet friendly tone
- Use clear, concise language
- Avoid technical jargon unless specifically discussing mechanical details
- Show empathy and understanding for customer concerns
- Provide specific next steps and clear calls to action

4. APPOINTMENT HANDLING:
When scheduling appointments:
- Collect customer name and contact information
- Verify preferred date and time
- Confirm vehicle details
- Send confirmation details
- Provide dealership location and preparation instructions

5. HANDOFF PROTOCOL:
Transfer to human staff when:
- Customer specifically requests human assistance
- Complex financing questions arise
- Detailed technical issues need discussion
- Complaint resolution is needed

6. FOLLOW-UP:
- Summarize key points discussed
- Provide relevant contact information
- Share next steps
- Offer additional assistance if needed

Remember to:
- Keep responses concise but informative
- Ask clarifying questions when needed
- Provide specific model information and pricing when available
- Always verify customer understanding before moving to next steps"""

# Initialize LLaMA model with specific parameters
model = OllamaLLM(
    model="llama3",
    temperature=0.7,
    top_p=0.95,
    max_tokens = 1024  # Limit response length
)

class CustomerDataManager:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.current_interactions_file = data_dir / "current_interactions.json"
        self.completed_interactions_file = data_dir / "completed_interactions.json"
        self._initialize_files()

    def _initialize_files(self):
        for file in [self.current_interactions_file, self.completed_interactions_file]:
            if not file.exists():
                file.write_text('{}')

    def save_interaction(self, session_id: str, user_info: dict, conversation: list):
        current_data = self._load_json(self.current_interactions_file)
        current_data[session_id] = {
            "user_info": user_info,
            "conversation": conversation,
            "last_updated": datetime.now().isoformat()
        }
        self._save_json(self.current_interactions_file, current_data)

    def complete_interaction(self, session_id: str):
        current_data = self._load_json(self.current_interactions_file)
        if session_id in current_data:
            completed_data = self._load_json(self.completed_interactions_file)
            completed_data[session_id] = current_data[session_id]
            completed_data[session_id]["completed_at"] = datetime.now().isoformat()
            self._save_json(self.completed_interactions_file, completed_data)
            del current_data[session_id]
            self._save_json(self.current_interactions_file, current_data)

    def _load_json(self, file_path: Path) -> dict:
        try:
            return json.loads(file_path.read_text())
        except json.JSONDecodeError:
            return {}

    def _save_json(self, file_path: Path, data: dict):
        file_path.write_text(json.dumps(data, indent=2))

# Initialize customer data manager
data_manager = CustomerDataManager(DATA_DIR)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    session = request.session
    if 'conversation_history' not in session:
        session['conversation_history'] = []
        session['user_info'] = {}
        first_message = {
            "role": "assistant",
            "content": "Welcome! I'm Alex. What's your name?"
        }
        session['conversation_history'].append(first_message)
    return templates.TemplateResponse(
        "index.html", 
        {"request": request, "initial_message": session['conversation_history'][0]['content']}
    )

@app.get('/favicon.ico')
async def favicon():
    return FileResponse('static/favicon.ico', media_type='image/vnd.microsoft.icon')

# Enhanced message processing function
async def process_message(message: str, user_info: dict, conversation_history: list) -> str:
    # Extract key information using simple pattern matching
    if len(conversation_history) == 1:  # First user message - likely name
        user_info['name'] = message.strip()
    elif 'budget' not in user_info and any(word in message.lower() for word in ['$', 'dollar', 'budget', 'spend']):
        # Extract budget information
        user_info['budget'] = message
    elif 'vehicle_type' not in user_info and any(word in message.lower() for word in ['car', 'suv', 'truck', 'van']):
        # Extract vehicle type preference
        user_info['vehicle_type'] = message

    # Prepare context for the model
    messages = [
        {"role": "system", "content": SYSTEM_CONTEXT},
        {"role": "system", "content": f"Current user info: {json.dumps(user_info)}"}
    ]
    messages.extend(conversation_history[-3:])  # Keep context window small
    
    input_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
    
    try:
        response = await asyncio.to_thread(model.invoke, input_text)
        return response
    except Exception as e:
        logger.error(f"Model error: {e}")
        return "I apologize, but I'm having trouble processing that. Could you please rephrase?"

@app.post("/api")
async def api(request: Request):
    try:
        data = await request.json()
        message = data.get("message")
        if not message:
            raise HTTPException(status_code=400, detail="Message is required")

        session = request.session
        session.setdefault('conversation_history', [])
        session.setdefault('user_info', {})

        # Update conversation history
        session['conversation_history'].append({"role": "user", "content": message})

        # Process message and get response
        response = await process_message(
            message,
            session['user_info'],
            session['conversation_history']
        )

        # Update conversation history with assistant's response
        session['conversation_history'].append({
            "role": "assistant",
            "content": response
        })

        # Save interaction data
        data_manager.save_interaction(
            str(id(session)),
            session['user_info'],
            session['conversation_history']
        )

        return JSONResponse({"content": response})

    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        return JSONResponse(
            {"error": "An error occurred processing your request"}, 
            status_code=500
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)