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
    st.button("Set API Keys")
    st.header("Upload data")
    st.file_uploader("Upload the data files here", accept_multiple_files=True)
    with st.expander("Local Model Endpoints"):
        vllm_base_url = st.text_input("vLLM Base URL", "http://localhost:8000")
        ollama_base_url = st.text_input("Ollama Base URL", "http://localhost:11434")
    st.header("Download project")
    st.download_button("Download all project files", data=b"", file_name="project.zip")

# --- Centralized Model Dictionary Creation ---
all_available_models = {}

# 1. Add your original, hardcoded "cloud" models
for model_name in predefined_models.keys():
    all_available_models[f"[Cloud] {model_name}"] = model_name

# 2. Add models found on the vLLM server
for model_name in get_vllm_models(vllm_base_url):
    display_name = f"[vLLM] {model_name.split('/')[-1]}"
    # --- FIX: Added default values for required fields ---
    all_available_models[display_name] = {
        "provider": "vLLM", "api_base": vllm_base_url, "model": model_name,
        "max_tokens": 4096, "temperature": 0.7, "repetition_penalty": 1.1
    }

# 3. Add models found on the Ollama server
for model_name in get_ollama_models(ollama_base_url):
    display_name = f"[Ollama] {model_name.split(':')[0]}"
    # --- FIX: Added default values for required fields ---
    all_available_models[display_name] = {
        "provider": "Ollama", "api_base": ollama_base_url, "model": model_name,
        "max_tokens": 4096, "temperature": 0.7, "repetition_penalty": 1.1
    }

# --- Main UI with Tabs ---
tab_names = ["Input prompt", "Idea", "Methods", "Analysis", "Paper", "Literature review", "Referee report", "Keywords"]
tabs = st.tabs(tab_names)

with tabs[0]: # Input prompt
    st.header("Input prompt")
    data_desc = st.text_area("Describe the data and tools...", height=150, label_visibility="collapsed")
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
    idea_model_display_name = st.selectbox("LLM Model", options=list(all_available_models.keys()), key="idea_model_select")
    idea_model_config = all_available_models[idea_model_display_name]
    generate_clicked = st.button("Generate", key="generate_idea")
    if generate_clicked:
        if not st.session_state.denario.research.data_description:
            st.error("Please set a data description first using the Input prompt tab.")
        else:
            with st.spinner("Generating idea..."):
                try:
                    st.session_state.denario.get_idea(llm=idea_model_config)
                    st.success("Idea generated!")
                    st.session_state.idea_text = st.session_state.denario.research.idea
                    st.rerun()
                except FileNotFoundError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

    if 'idea_text' in st.session_state:
        st.subheader("Generated idea")
        st.markdown(st.session_state.idea_text)

with tabs[2]: # Methods
    st.header("Methods")
    st.write("Generate the methods to be employed in the computation of the results.")
    method_model_display_name = st.selectbox("LLM Model", options=list(all_available_models.keys()), key="method_model_select")
    method_model_config = all_available_models[method_model_display_name]
    generate_clicked = st.button("Generate", key="generate_method")
    if generate_clicked:
        if not st.session_state.denario.research.data_description:
            st.error("Please set a data description first using the Input prompt tab.")
        elif not st.session_state.denario.research.idea:
            st.error("Please generate a research idea first using the Idea tab.")
        else:
            with st.spinner("Generating methods..."):
                try:
                    st.session_state.denario.get_method(llm=method_model_config)
                    st.success("Methods generated!")
                    st.session_state.method_text = st.session_state.denario.research.methodology
                    st.rerun()
                except FileNotFoundError as e:
                    st.error(str(e))
                except ValueError as e:
                    st.error(f"Invalid model configuration: {str(e)}")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
    if 'method_text' in st.session_state:
        st.subheader("Generated methods")
        st.markdown(st.session_state.method_text)

with tabs[3]: # Analysis
    st.header("Analysis")
    col1, col2 = st.columns(2)
    with col1:
        engineer_model_display_name = st.selectbox("Engineer Model", options=list(all_available_models.keys()), key="engineer_model_select")
        engineer_model_config = all_available_models[engineer_model_display_name]
    with col2:
        researcher_model_display_name = st.selectbox("Researcher Model", options=list(all_available_models.keys()), key="researcher_model_select")
        researcher_model_config = all_available_models[researcher_model_display_name]
    if st.button("Generate", key="generate_analysis"):
        with st.spinner("Running analysis..."):
            st.session_state.denario.get_results(engineer_model=engineer_model_config, researcher_model=researcher_model_config)
            st.success("Analysis complete!")
            st.session_state.results_text = st.session_state.denario.research.results
            st.rerun()
    if 'results_text' in st.session_state:
        st.subheader("Results")
        st.markdown(st.session_state.results_text)

with tabs[4]: # Paper
    st.header("Article")
    paper_model_display_name = st.selectbox("LLM Model", options=list(all_available_models.keys()), key="paper_model_select")
    paper_model_config = all_available_models[paper_model_display_name]
    if st.button("Generate", key="generate_paper"):
        with st.spinner("Writing paper..."):
            st.session_state.denario.get_paper(llm=paper_model_config)
            st.success("Paper generated!")
            st.rerun()
    st.write("Latex not generated yet.")
    st.write("Pdf not generated yet.")

with tabs[5]: # Literature review
    st.header("Literature review")
    st.selectbox("Choose mode for literature search:", ["semantic_scholar", "futurehouse"])
    lit_review_model_display_name = st.selectbox("LLM Model", options=list(all_available_models.keys()), key="lit_review_model_select")
    # Add button and call to denario.check_idea here

with tabs[6]: # Referee report
    st.header("Referee report")
    referee_model_display_name = st.selectbox("Choose a LLM model for the referee", options=list(all_available_models.keys()), key="referee_model_select")
    referee_model_config = all_available_models[referee_model_display_name]
    if st.button("Review", key="generate_review"):
        with st.spinner("Reviewing paper..."):
            st.session_state.denario.referee(llm=referee_model_config)
            st.success("Review complete!")
            st.rerun()
    st.write("Referee report not created yet.")

with tabs[7]: # Keywords
    st.header("Keywords")
    st.text_area("Enter your research text to extract keywords:", height=200)
    st.slider("Number of keywords to generate:", 1, 10, 5)
    st.selectbox("Keyword Type:", ["unesco", "aas"])
    st.button("Generate Keywords")
