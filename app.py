# Save this file as app.py in your project's root folder (e.g., Denario_offline/app.py)

import streamlit as st
import requests

# This is now an ABSOLUTE import from the 'denario' package
from denario.denario import Denario 
from denario.llm import models as predefined_models

# --- API Communication Functions ---
@st.cache_data(ttl=300)
def get_vllm_models(api_url):
    try:
        response = requests.get(f"{api_url}/v1/models")
        response.raise_for_status()
        return [model["id"] for model in response.json().get("data", [])]
    except requests.exceptions.RequestException:
        st.sidebar.warning(f"vLLM server not found at {api_url}")
        return []

@st.cache_data(ttl=300)
def get_ollama_models(api_url):
    try:
        response = requests.get(f"{api_url}/api/tags")
        response.raise_for_status()
        return [model["name"] for model in response.json().get("models", [])]
    except requests.exceptions.RequestException:
        st.sidebar.warning(f"Ollama server not found at {api_url}")
        return []

# --- Main Application ---
st.set_page_config(layout="wide", page_title="Denario")
st.title("Denario")
# (Your other introductory text and UI elements can go here)

if 'denario' not in st.session_state:
    st.session_state.denario = Denario(project_dir="denario_project", clear_project_dir=True)
    st.toast("Denario instance created!")

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("Denario")
    with st.expander("Local Model Endpoints"):
        vllm_base_url = st.text_input("vLLM Base URL", "http://localhost:8000")
        ollama_base_url = st.text_input("Ollama Base URL", "http://localhost:11434")
    
    # (Your other sidebar items like API Keys, Upload Data, etc.)
    # ...

# --- Create the merged dictionary of all models ---
all_available_models = {}
all_available_models["[Cloud] gpt-4o"] = "gpt-4o" # Example, add more as needed
for model_name in predefined_models.keys():
    all_available_models[f"[Cloud] {model_name}"] = model_name

for model_name in get_vllm_models(vllm_base_url):
    display_name = f"[vLLM] {model_name.split('/')[-1]}"
    all_available_models[display_name] = {"provider": "vLLM", "api_base": vllm_base_url, "model": model_name}

for model_name in get_ollama_models(ollama_base_url):
    display_name = f"[Ollama] {model_name.split(':')[0]}"
    all_available_models[display_name] = {"provider": "Ollama", "api_base": ollama_base_url, "model": model_name}

# --- Main UI with Tabs ---
tab_names = ["Input prompt", "Idea", "Methods", "Analysis", "Paper", "Literature review", "Referee report", "Keywords"]
input_tab, idea_tab, methods_tab, analysis_tab, paper_tab, lit_tab, referee_tab, keywords_tab = st.tabs(tab_names)

with input_tab:
    st.header("Input prompt")
    data_desc = st.text_area("Describe the data and tools...", height=200)
    if st.button("Set Data Description"):
        st.session_state.denario.set_data_description(data_desc)
        st.success("Data description set!")
    # ... your file upload logic ...

with idea_tab:
    st.header("Generate an Idea")
    st.subheader("Idea Generation Models")
    
    idea_maker_display_name = st.selectbox("Select Idea Maker Model", options=list(all_available_models.keys()), key="idea_maker_select")
    idea_hater_display_name = st.selectbox("Select Idea Hater Model", options=list(all_available_models.keys()), key="idea_hater_select", index=1 if len(all_available_models) > 1 else 0)

    idea_maker_config = all_available_models[idea_maker_display_name]
    idea_hater_config = all_available_models[idea_hater_display_name]
    
    if st.button("Generate Idea"):
        with st.spinner("Agents are brainstorming..."):
            st.session_state.denario.get_idea_cmagent(
                idea_maker_model=idea_maker_config,
                idea_hater_model=idea_hater_config
            )
            st.success("Idea generated!")
            if hasattr(st.session_state.denario.research, 'idea'):
                st.session_state.idea = st.session_state.denario.research.idea
            st.rerun()

    if 'idea' in st.session_state:
        st.markdown("### Generated Idea:")
        st.markdown(st.session_state.idea)

# ... (Continue building out your other tabs in the same way) ...
