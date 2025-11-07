import streamlit as st
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from denario import Denario

st.title("Denario")

# Initialize session state
if 'denario' not in st.session_state:
    st.session_state.denario = Denario()
if 'connected' not in st.session_state:
    st.session_state.connected = False

# Connection UI
st.header("Connect to Local LLMs")
vllm_base_url = st.text_input("vLLM Base URL", "http://localhost:8000/v1")
ollama_host = st.text_input("Ollama Host", "http://localhost:11434")

if st.button("Connect"):
    st.session_state.denario.connect_local_llm(vllm_base_url=vllm_base_url, ollama_host=ollama_host)
    st.session_state.local_models = st.session_state.denario.get_local_models()
    st.session_state.connected = True
    st.success("Connected to local LLMs!")

# Model Selection UI
if st.session_state.connected:
    st.header("Select a Model")
    provider = st.selectbox("Select LLM Provider", options=list(st.session_state.local_models.keys()))

    if provider:
        model = st.selectbox("Select Model", options=st.session_state.local_models[provider])
        st.write(f"You selected: {provider} - {model}")
