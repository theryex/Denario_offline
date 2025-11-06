import streamlit as st
from denario import Denario

st.title("Denario")

denario = Denario()
local_models = denario.get_local_models()

provider = st.selectbox("Select LLM Provider", options=list(local_models.keys()))

if provider:
    model = st.selectbox("Select Model", options=local_models[provider])
    st.write(f"You selected: {provider} - {model}")
