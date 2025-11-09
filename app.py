import streamlit as st
import requests
from denario import Denario # Assuming your class is in denario/__init__.py or denario/denario.py

# --- API Communication Functions (from our previous script) ---

@st.cache_data(ttl=300)
def get_vllm_models(api_url):
    try:
        response = requests.get(f"{api_url}/v1/models")
        response.raise_for_status()
        return [model["id"] for model in response.json().get("data", [])]
    except requests.exceptions.RequestException:
        return []

@st.cache_data(ttl=300)
def get_ollama_models(api_url):
    try:
        response = requests.get(f"{api_url}/api/tags")
        response.raise_for_status()
        return [model["name"] for model in response.json().get("models", [])]
    except requests.exceptions.RequestException:
        return []

# --- UI Helper Function ---

def agent_model_selector(agent_role: str, defaults: dict, vllm_url: str, ollama_url: str):
    """Creates a set of widgets in the sidebar for a specific agent role."""
    st.subheader(agent_role)
    provider = st.selectbox("Provider", ["vLLM", "Ollama"], key=f"{agent_role}_provider")
    
    api_base = vllm_url if provider == 'vLLM' else ollama_url
    models = get_vllm_models(vllm_url) if provider == 'vLLM' else get_ollama_models(ollama_url)
    
    model = st.selectbox("Model", models, key=f"{agent_role}_model")
    
    st.slider("Temperature", 0.0, 1.0, defaults['temp'], 0.05, key=f"{agent_role}_temp")
    st.slider("Max Tokens", 50, 8000, defaults['tokens'], key=f"{agent_role}_max")
    st.slider("Repetition Penalty", 0.0, 2.0, defaults['penalty'], 0.05, key=f"{agent_role}_penalty")

    # Return a dictionary that can be passed to llm_parser
    return {
        "provider": provider,
        "api_base": api_base,
        "model": model,
        "temperature": st.session_state[f"{agent_role}_temp"],
        "max_tokens": st.session_state[f"{agent_role}_max"],
        "repetition_penalty": st.session_state[f"{agent_role}_penalty"]
    }

# --- Streamlit App ---

st.set_page_config(layout="wide", page_title="Denario AI Research Pilot")
st.title("🔬 Denario AI Research Pilot")

# --- Initialize Denario in Session State ---
if 'denario' not in st.session_state:
    # Initialize Denario with a project directory. Clear it for fresh runs.
    st.session_state.denario = Denario(project_dir="denario_project", clear_project_dir=True)
    st.toast("Denario instance created!")

# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("API Endpoints")
    vllm_base_url = st.text_input("vLLM Base URL", "http://localhost:8000")
    ollama_base_url = st.text_input("Ollama Base URL", "http://localhost:11434")
    st.markdown("---")
    
    st.header("Agent Model Configuration")
    
    with st.expander("Idea Generation Agents", expanded=True):
        idea_maker_config = agent_model_selector("Idea Maker", {'temp': 0.7, 'tokens': 2048, 'penalty': 1.1}, vllm_base_url, ollama_base_url)
        idea_hater_config = agent_model_selector("Idea Hater", {'temp': 0.8, 'tokens': 2048, 'penalty': 1.2}, vllm_base_url, ollama_base_url)
    
    with st.expander("Methodology Agents"):
        method_gen_config = agent_model_selector("Method Generator", {'temp': 0.6, 'tokens': 3000, 'penalty': 1.1}, vllm_base_url, ollama_base_url)
        
    with st.expander("Results Agents (Engineer)"):
        engineer_config = agent_model_selector("Engineer", {'temp': 0.3, 'tokens': 4096, 'penalty': 1.0}, vllm_base_url, ollama_base_url)

    with st.expander("Paper Writing Agent"):
        paper_writer_config = agent_model_selector("Paper Writer", {'temp': 0.7, 'tokens': 4096, 'penalty': 1.15}, vllm_base_url, ollama_base_url)


# --- Main Application Flow ---

st.header("1. Data and Tools Description")
data_desc = st.text_area("Provide a detailed description of the data, tools, and libraries to be used for the research.", height=200, key="data_description")
if st.button("Set Data Description"):
    if data_desc:
        st.session_state.denario.set_data_description(data_desc)
        st.success("Data description has been set and saved.")
    else:
        st.warning("Please provide a data description.")

st.markdown("---")

st.header("2. Generate Research Idea")
if st.button("Generate Idea"):
    with st.spinner("Agents are brainstorming..."):
        try:
            # Here we pass the configuration dictionaries directly
            st.session_state.denario.get_idea_cmagent(
                idea_maker_model=idea_maker_config,
                idea_hater_model=idea_hater_config
                # Add other planner/orchestrator models if you want to configure them too
            )
            st.success("Idea generated successfully!")
            st.session_state.idea = st.session_state.denario.research.idea
        except Exception as e:
            st.error(f"An error occurred: {e}")

if 'idea' in st.session_state:
    with st.expander("View Generated Idea", expanded=True):
        st.markdown(st.session_state.idea)

st.markdown("---")

st.header("3. Generate Methodology")
if st.button("Generate Method"):
    with st.spinner("Methodology agent at work..."):
        try:
            st.session_state.denario.get_method_cmbagent(
                method_generator_model=method_gen_config
            )
            st.success("Methodology generated!")
            st.session_state.method = st.session_state.denario.research.methodology
        except Exception as e:
            st.error(f"An error occurred: {e}")

if 'method' in st.session_state:
    with st.expander("View Generated Method"):
        st.markdown(st.session_state.method)
