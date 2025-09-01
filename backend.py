from pydantic import BaseModel
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from ai_agent import get_response_from_ai_agent
from dotenv import load_dotenv
load_dotenv()

# Define your request and response models first
class RequestState(BaseModel):
    model_name: str
    model_provider: str
    system_prompt: Optional[str] = "Act as an AI chatbot who is smart and friendly"
    messages: List[str]
    allow_search: bool = False

class ChatResponse(BaseModel):
    response: str

# Then create FastAPI app and your endpoint
app = FastAPI(title="LangGraph AI Agent")

ALLOWED_MODEL_NAMES = [
    # Groq
    "llama3-70b-8192",
    "mixtral-8x7b-32768",
    "llama-3.3-70b-versatile",

    # OpenAI
    "gpt-4o-mini",

    # Gemini
    "gemini-1.5-flash",
    "gemini-1.5-pro"
]

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: RequestState):
    if request.model_name not in ALLOWED_MODEL_NAMES:
        raise HTTPException(status_code=400, detail="Invalid model name. Kindly select a valid AI model.")

    try:
        response = get_response_from_ai_agent(
            request.model_name,
            request.messages[-1],   # currently only last message
            request.allow_search,
            request.system_prompt,
            request.model_provider
        )

        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="127.0.0.1", port=8000, reload=True)
