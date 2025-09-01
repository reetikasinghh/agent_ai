import os
import traceback
from dotenv import load_dotenv
load_dotenv()
from fastapi import HTTPException
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


# Load API keys from environment variables
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Default system prompt
system_prompt = "Act as an AI chatbot who is smart and friendly"

def get_response_from_ai_agent(
    llm_id: str,
    query: str,
    allow_search: bool,
    system_prompt: str,
    provider: str
) -> str:
    try:
        # Select provider
        if provider.lower() == "groq":
            if not GROQ_API_KEY:
                raise ValueError("GROQ_API_KEY is missing in environment variables.")
            llm = ChatGroq(model=llm_id, api_key=GROQ_API_KEY)

        elif provider.lower() == "openai":
            if not OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is missing in environment variables.")
            llm = ChatOpenAI(model=llm_id, api_key=OPENAI_API_KEY)

        elif provider.lower() == "gemini":
            if not GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY is missing in environment variables.")
            llm = ChatGoogleGenerativeAI(model=llm_id, google_api_key=GEMINI_API_KEY)

        else:
            raise ValueError(f"Unsupported provider: {provider}")

        # Tools
        tools = []
        if allow_search:
            if not TAVILY_API_KEY:
                raise ValueError("TAVILY_API_KEY is missing in environment variables.")
            tools.append(TavilySearchResults(max_results=2, api_key=TAVILY_API_KEY))

        # Agent
        agent = create_react_agent(
            model=llm,
            tools=tools
        )

        # Conversation State
        state = {
            "messages": [SystemMessage(content=system_prompt), HumanMessage(content=query)]
        }

        # Call agent
        response = agent.invoke(state)

        # Extract AI messages
        messages_out = response.get("messages", [])
        ai_messages = [msg.content for msg in messages_out if isinstance(msg, AIMessage)]

        # Return the last AI message or a default message
        return ai_messages[-1] if ai_messages else "No response from AI agent."

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"AI Agent Error: {str(e)}")
