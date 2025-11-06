import streamlit as st
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from denario import Denario

st.title("Denario")

denario = Denario()
local_models = denario.get_local_models()

provider = st.selectbox("Select LLM Provider", options=list(local_models.keys()))

if provider:
    model = st.selectbox("Select Model", options=local_models[provider])
    st.write(f"You selected: {provider} - {model}")
