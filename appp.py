import streamlit as st
import ollama
import os

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_PERSIST_DIR = "chroma_db_pydanticai_final"
CHROMA_COLLECTION_NAME = "pydantic_ai_rag_final"
LLM_MODEL = "llama3"
HF_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


@st.cache_resource(show_spinner="Loading Pydantic AI Vector Store...")
def load_vector_store():
    if not os.path.exists(CHROMA_PERSIST_DIR):
        st.error(f"Vector store not found at '{CHROMA_PERSIST_DIR}'") #If it doesn't exist,stop the process
        st.info("Please run the data ingestion scripts first (`1_crawl.py`, then `2_embed.py`).")
        st.stop()
    
    embeddings = HuggingFaceEmbeddings(model_name=HF_EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=CHROMA_COLLECTION_NAME
    )
    return vectorstore

def get_ollama_response(current_prompt: str, context: str, chat_history: list):
    formatted_prompt = f"""
You are an expert at Pydantic AI. Your answers MUST be based *only* on the provided CONTEXT.
Use the CHAT HISTORY to understand follow-up questions.
If the context does not contain the answer, state that you could not find the answer in the documentation. Do not use outside knowledge.

CONTEXT:
{context}

QUESTION:
{current_prompt}

ANSWER:
"""
    messages_for_api = list(chat_history)
    messages_for_api.append({'role': 'user', 'content': formatted_prompt})

    response_stream = ollama.chat(
        model=LLM_MODEL, 
        messages=messages_for_api, 
        stream=True
    )

    for chunk in response_stream:
        if 'content' in chunk['message']:
            yield chunk['message']['content']

#App config
st.set_page_config(page_title="Pydantic AI Chatbot", page_icon="🤖")
st.title("🤖 Chat with the Pydantic AI Docs")
st.caption(f"Your personal tutor for Pydantic AI")

try:
    vectorstore = load_vector_store()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
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
    st.session_state.messages.append({"role": "user", "content": prompt}) # Add user's new message to the history
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching documentation..."):
            relevant_docs = retriever.invoke(prompt)
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            chat_history_list = st.session_state.messages[:-1]

            stream = get_ollama_response(prompt, context, chat_history_list)
            response = st.write_stream(stream)
            
    st.session_state.messages.append({"role": "assistant", "content": response}) #Add the chatbot's reply to the history