import streamlit as st
from openai import OpenAI

# App title
st.set_page_config(page_title="🤖 vLLM Chat App")

# Function to get the model name from the vLLM server
@st.cache_resource
def get_model_name():
    """Fetches the model name from the vLLM server."""
    try:
        client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
        models = client.models.list()
        if models.data:
            return models.data[0].id
        else:
            return None
    except Exception as e:
        st.error(f"Could not connect to vLLM server at http://localhost:8000. Please ensure the server is running. Error: {e}")
        return None

# Get the model name
model_name = get_model_name()

# Only proceed if a model name was successfully retrieved
if model_name:
    # Display the fetched model name
    st.title("🤖 vLLM Chat App")
    st.info(f"Interacting with model: **{model_name}**")

    # OpenAI client pointing to the local vLLM server
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

    # Sidebar for model parameters
    with st.sidebar:
        st.title("Settings")
        st.subheader("Model Parameters")
        temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.01)
        top_p = st.sidebar.slider("Top P", min_value=0.0, max_value=1.0, value=1.0, step=0.01)
        max_tokens = st.sidebar.slider("Max Tokens", min_value=64, max_value=4096, value=512, step=64)

        # Button to clear chat history
        if st.button("Clear Chat History"):
            st.session_state.messages = [
                {"role": "assistant", "content": "How may I assist you today?"}
            ]

    # Store LLM-generated responses
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

    # Display or clear chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User-provided prompt
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Generate a new response if the last message is not from the assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = client.chat.completions.create(
                    model=model_name,  # Use the dynamically fetched model name
                    messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    stream=True
                )
                placeholder = st.empty()
                full_response = ''
                for chunk in response:
                    if chunk.choices[0].delta.content is not None:
                        full_response += chunk.choices[0].delta.content
                        placeholder.markdown(full_response)
                placeholder.markdown(full_response)
        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message)
else:
    st.error("Could not retrieve model name. Please ensure your vLLM server is running and accessible.")
