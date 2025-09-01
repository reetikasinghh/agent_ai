import streamlit as st
import requests

st.set_page_config(page_title="LangGraph Agent UI", layout="centered")
st.title("vistron AI agent")
st.write(
    "Create and Interact with the AI Agent to! "
    "E.g. Act as a teacher, astrologer, or fitness coach..."
)

# System prompt input
system_prompt = st.text_area(
    "Define your AI Agent: ",
    height=70,
    placeholder="Type your system prompt here..."
)

# Model lists
MODEL_NAMES_GROQ = ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"]
MODEL_NAMES_OPENAI = ["gpt-4o-mini"]
MODEL_NAMES_GEMINI = ["gemini-1.5-flash", "gemini-1.5-pro"]

# Use lowercase provider names to match backend checks
provider = st.radio("Select Provider:", ("groq", "openai", "gemini"))

# Provider → Model mapping
if provider == "groq":
    selected_model = st.selectbox("Select Groq Model:", MODEL_NAMES_GROQ)
elif provider == "openai":
    selected_model = st.selectbox("Select OpenAI Model:", MODEL_NAMES_OPENAI)
elif provider == "gemini":
    selected_model = st.selectbox("Select Gemini Model:", MODEL_NAMES_GEMINI)

# Optional search toggle
allow_web_search = st.checkbox("Allow Web Search")

# User query
user_query = st.text_area(
    "Enter your query: ",
    height=150,
    placeholder="Ask Anything!"
)

API_URL = "http://127.0.0.1:8000/chat"

# Button to send request
if st.button("Ask Agent!"):
    if user_query.strip():
        payload = {
            "model_name": selected_model,
            "model_provider": provider,  # lowercase to match backend
            "system_prompt": system_prompt,
            "messages": [user_query],
            "allow_search": allow_web_search,
        }

        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            response_data = response.json()
            if "error" in response_data:
                st.error(response_data["error"])
            else:
                st.subheader("Agent Response")
                st.markdown(f"**Final Response:** {response_data['response']}")
        else:
            st.error(f"API Error: {response.status_code}")
    else:
        st.warning("Please enter a query before submitting.")
