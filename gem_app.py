import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

CHROMA_PERSIST_DIR = "chroma_db_pydanticai"
CHROMA_COLLECTION_NAME = "pydantic_ai_rag"
HF_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-2.5-pro"

#API key config
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except (FileNotFoundError, KeyError):
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("Google API Key not found!")
    st.info("Please set your GOOGLE_API_KEY in a .env file or in Streamlit's secrets manager.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)


@st.cache_resource(show_spinner="Loading Pydantic AI Vector Store...")
def load_vector_store():
    if not os.path.exists(CHROMA_PERSIST_DIR):
        st.error(f"Vector store not found at '{CHROMA_PERSIST_DIR}'")
        st.info("Please run the data ingestion scripts first.")
        st.stop()
    
    embeddings = HuggingFaceEmbeddings(model_name=HF_EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=CHROMA_COLLECTION_NAME
    )
    return vectorstore

#Save chat history as user/assistant
def format_chat_history(messages):
    """
    Formats the chat history from Streamlit's session state into a single string.
    """
    history = []
    for message in messages:
        role = "User" if message["role"] == "user" else "Assistant"
        history.append(f"{role}: {message['content']}")
    return "\n".join(history)


def get_gemini_response(current_prompt: str, context: str, chat_history: str):
    formatted_prompt = f"""
You are an expert-level AI Assistant for the Pydantic AI Python framework. Your knowledge comes ONLY from the documentation provided in the CONTEXT section. You must stick to these rules:

1.  Base All Answers on Context: Your answers MUST be derived exclusively from the information within the `CONTEXT` block.
2.  Use Chat History: Use the `CHAT HISTORY` to understand follow-up questions and maintain conversational flow.
3.  Honesty is Critical: If the answer is not in the `CONTEXT`, state that clearly. Do not guess.
4.  Provide Details and Code: When the context has the answer, explain it thoroughly and provide code examples if they are available in the context.
5.  Cite Sources: If the context includes source URLs, list them at the end of your response under a "Sources:" heading.

---
CONTEXT:
{context}
---
CHAT HISTORY:
{chat_history}
---

CURRENT QUESTION:
{current_prompt}

ASSISTANT'S ANSWER:
"""
    model = genai.GenerativeModel(GEMINI_MODEL)
    response_stream = model.generate_content(formatted_prompt, stream=True)
    
    for chunk in response_stream:
        if hasattr(chunk, 'text'):
            yield chunk.text

#App config
st.set_page_config(page_title="Pydantic AI Chatbot", page_icon="🤖")
st.title("🤖 Chat with the Pydantic AI Docs")
st.caption(f"YOur personal tutor for Pydantic AI")

try:
    vectorstore = load_vector_store()
    retriever = vectorstore.as_retriever(
    search_type="mmr", #Maximal Marginal Relevance (MMR)
    search_kwargs={"k": 6,"lambda_mult": 0.3, "fetch_k": 20} 
    # Fetch 20 docs, then select the best 6 diverse ones
)
except Exception as e:
    st.error("An error occurred while loading the application.")
    st.exception(e)
    st.stop()

# Initialize session state for messages if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Ask me anything about Pydantic AI."}]

# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about Pydantic AI..."):

    st.session_state.messages.append({"role": "user", "content": prompt}) #Add user's new message to the history
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching documentation..."):
            relevant_docs = retriever.invoke(prompt)
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            chat_history_str = format_chat_history(st.session_state.messages[:-1])

            stream = get_gemini_response(prompt, context, chat_history_str)

            response = st.write_stream(stream)
            
    st.session_state.messages.append({"role": "assistant", "content": response}) #Add the assistant's response to the history