# Save this file as app.py in your project's root folder (e.g., Denario_offline/app.py)

import streamlit as st
import requests

# These are now absolute imports from your 'denario' package
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

# --- Main Application Setup ---
st.set_page_config(layout="wide", page_title="Denario")
st.title("Denario")
st.write("AI agents to assist the development of a scientific research process.")
st.warning("⚠️ This is a demo deployment. Your session will expire if you close the tab or refresh the page. Recall to download your project files from the sidebar before leaving! ⚠️")

if 'denario' not in st.session_state:
    # Initialize the backend Denario class
    st.session_state.denario = Denario(project_dir="denario_project", clear_project_dir=True)
    st.toast("Denario instance created!")

# --- Sidebar UI ---
with st.sidebar:
    st.header("API keys")
    # (Your existing API key logic here)
    st.button("Set API Keys")

    st.header("Upload data")
    st.file_uploader("Upload the data files here", accept_multiple_files=True)

    # --- NEW: Local Model Endpoint Configuration ---
    with st.expander("Local Model Endpoints"):
        vllm_base_url = st.text_input("vLLM Base URL", "http://localhost:8000")
        ollama_base_url = st.text_input("Ollama Base URL", "http://localhost:11434")

    st.header("Download project")
    st.download_button("Download all project files", data="", file_name="project.zip")


# --- Centralized Model Dictionary Creation ---
# This dictionary combines all available models from all sources.
all_available_models = {}

# 1. Add your original, hardcoded "cloud" models
for model_name in predefined_models.keys():
    all_available_models[f"[Cloud] {model_name}"] = model_name

# 2. Add models found on the vLLM server
for model_name in get_vllm_models(vllm_base_url):
    display_name = f"[vLLM] {model_name.split('/')[-1]}"
    all_available_models[display_name] = {"provider": "vLLM", "api_base": vllm_base_url, "model": model_name}

# 3. Add models found on the Ollama server
for model_name in get_ollama_models(ollama_base_url):
    display_name = f"[Ollama] {model_name.split(':')[0]}"
    all_available_models[display_name] = {"provider": "Ollama", "api_base": ollama_base_url, "model": model_name}


# --- Main UI with Tabs (Mirrors your original app) ---
tab_names = ["Input prompt", "Idea", "Methods", "Analysis", "Paper", "Literature review", "Referee report", "Keywords"]
tabs = st.tabs(tab_names)

with tabs[0]: # Input prompt
    st.header("Input prompt")
    st.write("Describe the data and tools to be used in the project. You may also include information about the computing resources required.")
    data_desc = st.text_area("E.g. Analyze the experimental data stored in ./path/to/data.csv using sklearn and pandas...", height=150, label_visibility="collapsed")
    
    if st.button("Set Data Description"):
        if data_desc:
            st.session_state.denario.set_data_description(data_desc)
            st.success("Data description has been set!")
            st.rerun()
        else:
            st.warning("Please provide a data description.")
    
    st.subheader("Current data description")
    st.markdown(st.session_state.denario.research.data_description or "*Data description not set.*")

with tabs[1]: # Idea
    st.header("Research idea")
    st.write("Generate a research idea provided the data description.")
    
    # In your UI this is under a radio button, we'll just show the main one for now
    st.write("Choose a LLM model for the first generation")
    
    idea_model_display_name = st.selectbox("LLM Model", options=list(all_available_models.keys()), key="idea_model_select")
    idea_model_config = all_available_models[idea_model_display_name]
    
    if st.button("Generate", key="generate_idea"):
        with st.spinner("Generating idea..."):
            # Note: We call get_idea which has a 'fast' mode default and uses the 'llm' param
            st.session_state.denario.get_idea(llm=idea_model_config)
            st.success("Idea generated!")
            st.session_state.idea_text = st.session_state.denario.research.idea
            st.rerun()

    if 'idea_text' in st.session_state:
        st.subheader("Generated idea")
        st.markdown(st.session_state.idea_text)

with tabs[2]: # Methods
    st.header("Methods")
    st.write("Generate the methods to be employed in the computation of the results, provided the data and idea description.")
    
    st.write("Choose a LLM model for the first generation")
    method_model_display_name = st.selectbox("LLM Model", options=list(all_available_models.keys()), key="method_model_select")
    method_model_config = all_available_models[method_model_display_name]
    
    if st.button("Generate", key="generate_method"):
        with st.spinner("Generating methods..."):
            st.session_state.denario.get_method(llm=method_model_config)
            st.success("Methods generated!")
            st.session_state.method_text = st.session_state.denario.research.methodology
            st.rerun()

    if 'method_text' in st.session_state:
        st.subheader("Generated methods")
        st.markdown(st.session_state.method_text)

with tabs[3]: # Analysis
    st.header("Analysis")
    st.write("Compute the results, given the methods, idea and data description.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("Engineer: Generates the code to compute the results")
        engineer_model_display_name = st.selectbox("Engineer Model", options=list(all_available_models.keys()), key="engineer_model_select")
        engineer_model_config = all_available_models[engineer_model_display_name]
    with col2:
        st.write("Researcher: processes the results and writes the results report")
        researcher_model_display_name = st.selectbox("Researcher Model", options=list(all_available_models.keys()), key="researcher_model_select")
        researcher_model_config = all_available_models[researcher_model_display_name]

    if st.button("Generate", key="generate_analysis"):
        with st.spinner("Running analysis..."):
            st.session_state.denario.get_results(
                engineer_model=engineer_model_config,
                researcher_model=researcher_model_config
            )
            st.success("Analysis complete!")
            st.session_state.results_text = st.session_state.denario.research.results
            st.rerun()
            
    if 'results_text' in st.session_state:
        st.subheader("Results")
        st.markdown(st.session_state.results_text)


with tabs[4]: # Paper
    st.header("Article")
    st.write("Write the article using the computed results of the research.")
    with st.expander("Options for the paper writing agents"):
        paper_model_display_name = st.selectbox("LLM Model", options=list(all_available_models.keys()), key="paper_model_select")
        paper_model_config = all_available_models[paper_model_display_name]

    if st.button("Generate", key="generate_paper"):
        with st.spinner("Writing paper..."):
            st.session_state.denario.get_paper(llm=paper_model_config)
            st.success("Paper generated!")
            # Add logic to display links to latex/pdf files here
            st.rerun()

    st.write("Latex not generated yet.")
    st.write("Pdf not generated yet.")

with tabs[5]: # Literature review
    st.header("Literature review")
    st.write("Check if the research idea has been investigated in previous literature.")
    
    st.selectbox("Choose mode for literature search:", ["semantic_scholar", "futurehouse"])
    
    lit_review_model_display_name = st.selectbox("LLM Model", options=list(all_available_models.keys()), key="lit_review_model_select")
    lit_review_model_config = all_available_models[lit_review_model_display_name]
    
    # Add button and call to denario.check_idea here

with tabs[6]: # Referee report
    st.header("Referee report")
    st.write("Review a paper, producing a report providing feedback on the quality of the article and aspects to be improved.")

    referee_model_display_name = st.selectbox("Choose a LLM model for the referee", options=list(all_available_models.keys()), key="referee_model_select")
    referee_model_config = all_available_models[referee_model_display_name]
    
    if st.button("Review", key="generate_review"):
        with st.spinner("Reviewing paper..."):
            st.session_state.denario.referee(llm=referee_model_config)
            st.success("Review complete!")
            # Add logic to display review text here
            st.rerun()
            
    st.write("Referee report not created yet.")

with tabs[7]: # Keywords
    st.header("Keywords")
    st.write("Generate keywords from your research text.")
    st.text_area("Enter your research text to extract keywords:", height=200)
    st.slider("Number of keywords to generate:", 1, 10, 5)
    st.selectbox("Keyword Type:", ["unesco", "aas"])
    st.button("Generate Keywords")
