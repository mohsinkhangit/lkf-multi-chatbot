# streamlit_app.py
import os
import streamlit as st
from streamlit_option_menu import option_menu
from modules import gemini_module
from dotenv import load_dotenv

load_dotenv()

# --- SECURITY ---
secret_password = os.environ.get('STREAMLIT_APP_PASSWORD')  # Safer retrieval
if not secret_password:
    st.error("STREAMLIT_APP_PASSWORD environment variable not set.")
    st.stop()  # Prevent the app from running without the password

# --- MODEL DEFINITIONS ---
gemini_models = {
    "gemini-2.0-flash-001": "Gemini 2.0 Flash",
    "gemini-2.0-pro-exp-02-05": "Gemini 2.0 Pro",
    "gemini-2.0-flash-exp": "Gemini 2.0 Flash",
    "gemini-2.0-flash-lite-preview-02-05": "Gemini 2.0 Flash-Lite",
    "gemini-2.0-flash-thinking-exp-01-21": "Gemini 2.0 Flash Thinking",
    "gemini-1.5-flash": "Gemini 1.5 Flash",
    "gemini-1.5-pro": "Gemini 1.5 Pro",
}

openai_models = {
    "GPT-4o": "GPT-4o",
    "GTP-4o-mini": "GPT-4o-mini",
}

llama_models = {
    "LLAMA": "LLAMA",
}

deepseek_models = {
    "DeepSeek": "DeepSeek",
}


all_models = {
    "Gemini": gemini_models,
    "OpenAI": openai_models,
    # "LLAMA": llama_models,
    "DeepSeek": deepseek_models,
}


# --- ERROR HANDLING ---
def handle_error(e, model_category):
    """Generic error handling function."""
    st.error(f"An error occurred while querying {model_category}: {e}")


# --- STREAMLIT APP ---
st.title("Multi-Page Model Selection App")

# Password protection
if 'password_correct' not in st.session_state:
    st.session_state['password_correct'] = False

if not st.session_state['password_correct']:  # Only show password input if not correct
    entered_password = st.text_input("Password", type="password")
    if entered_password:
        if entered_password == secret_password:
            st.session_state['password_correct'] = True
            st.rerun()  # Refresh the app to show the content
        else:
            st.error("Incorrect password")
    st.stop() #stops the app so the rest of the code is not run

# Initialize chat history - Moved outside password check to only happen once
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Left panel menu
with st.sidebar:
    selected_category = option_menu(
        menu_title="Choose Model Category",  # Added menu title
        options=list(all_models.keys()),
        icons=["rocket", "robot", "search"],
        orientation="vertical",
    )

# Main panel
if selected_category:
    model_names = list(all_models[selected_category].keys())
    selected_model_name = st.selectbox("Select Model", model_names)

    # Category-specific options
    if selected_category == "Gemini":
        grounding_source = st.checkbox("Enable Grounding Source: Google Search")
        if grounding_source and selected_model_name not in ("gemini-2.0-flash-001", \
                                                            "gemini-2.0-pro-exp-02-05",
                                                            "gemini-2.0-flash-exp"):
            st.warning("Grounding Source is not available for this model.")
            grounding_source = False
    else:
        grounding_source = False #Setting grounding_source to false to avoid errors

    # Get the model ID
    selected_model_id = selected_model_name # Corrected: Model ID is the key itself

    st.write(f"You selected: {selected_model_name} (Model ID: `{selected_model_id}`)")  # More informative

    # Chat input
    prompt = st.chat_input("Say something")

    if prompt:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response from Gemini
        response = gemini_module.generate_response(selected_model_name, st.session_state.messages,
                                                   grounding_source=grounding_source)

        if response:
            # Add Gemini's response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response[-1]['content']})
            with st.chat_message("assistant"):
                st.markdown(response[-1]['content'])
        else:
            st.error("No response received from the model.")
else:
    st.info("Select a model category from the sidebar to start.")