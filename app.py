import os
import streamlit as st
from streamlit_option_menu import option_menu
from langchain_core.messages import HumanMessage, AIMessage

# Import our refactored and new modules
from modules import gemini_module
from modules import session_manager_module as sm  # Use a short alias
from modules import lc_memory_module
from dotenv import load_dotenv

load_dotenv()


# --- STREAMLIT APP ---
st.set_page_config(layout="wide")
st.title("LKF GenAI Chat")

sm.init_db()  # Initializes the 'sessions' table
lc_memory_module.create_history_table_if_not_exists()

# --- SECURITY ---
secret_password = os.environ.get('STREAMLIT_APP_PASSWORD')
if not secret_password:
    st.error("STREAMLIT_APP_PASSWORD environment variable not set.")
    st.stop()

# --- MODEL DEFINITIONS ---
# (Model definitions remain the same as before)
gemini_models = {
    "gemini-2.0-flash-lite-001": "gemini-2.0-flash-lite-001",
    "gemini-2.0-flash-001": "gemini-2.0-flash-001",
    "gemini-2.5-pro": "gemini-2.5-pro",
    "gemini-1.5-flash": "Gemini 1.5 Flash"
}
# ... other models

openai_models = {
    "GPT-4o": {"name": "gpt-4o", "deployment_name": "gpt-4o-2", "api_version": "2024-08-01-preview"},
    "GTP-4o-mini": {"name": "gpt-4o-mini", "deployment_name": "gpt-4o-mini", "api_version": "2024-08-01-preview"}
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
    # "LLAMA": llama_models, # Uncomment if LLAMA models are to be included
    "DeepSeek": deepseek_models,
}

def _convert_messages_to_dict(messages) -> list[dict]:
    """Helper to convert LangChain messages to a dict format for display."""
    return [
        {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
        for m in messages
    ]

def handle_error(e, model_category):
    """Generic error handling function."""
    st.error(f"An error occurred while querying {model_category}: {e}")

def new_chat_session():
    """Initializes a new chat session."""
    # (The logic to summarize the previous chat before switching remains the same)
    if (st.session_state.get('current_session_id') and
            st.session_state.messages and  # Check if there are messages in the active session
            (st.session_state.get(
                'current_session_topic') is None or st.session_state.current_session_topic == "New Chat")):

        first_user_message_content = next((m["content"] for m in st.session_state.messages if m["role"] == "user"),
                                          None)

        if first_user_message_content:
            # Determine appropriate model_id for topic generation: use currently selected or a default
            # Assuming gemini_module.generate_topic_from_text can work without a specific model_id for summarization
            # or uses a default if not provided/suitable.
            # If a model_id is strictly required here, consider passing a most common one like "gemini-1.5-flash"
            # or handling its absence within gemini_module.

            # This 'selected_model_id' would be from the _current_ state before clearing
            current_model_id_for_topic = st.session_state.get('selected_model_id_main_panel',
                                                              'gemini-2.0-flash-lite-001')  # Provide a fallback

            with st.spinner("Summarizing previous chat..."):
                topic = gemini_module.generate_topic_from_text(
                    current_model_id_for_topic,  # Passed for potential use by gemini_module
                    first_user_message_content
                )
                if topic and topic != "Untitled Conversation":
                    sm.update_session_topic_db(st.session_state.current_session_id, topic)

    session_id = sm.create_new_session_db()
    st.session_state.current_session_id = session_id
    st.session_state.current_session_topic = "New Chat"
    st.session_state.messages = []
    st.session_state.processing_prompt = None  # Clear any pending prompt
    update_all_sessions()
    # st.rerun()

def load_chat_session(session_data):
    """Loads a chat session using LangChain history."""
    st.session_state.current_session_id = session_data["session_id"]
    st.session_state.current_session_topic = session_data["topic"]

    # Use LangChain to get message history
    history = lc_memory_module.get_chat_history(session_data["session_id"])
    st.session_state.messages = _convert_messages_to_dict(history.messages)

    st.session_state.processing_prompt = None  # Clear any pending prompt
    st.rerun()


def delete_chat_session(session_id: str):
    """Deletes a session record and its associated message history."""
    if sm.delete_session_db(session_id):
        st.success("Chat session deleted.")
        st.session_state.current_session_id = None  # Force reload to default
        update_all_sessions()
        st.rerun()
    else:
        st.error("Failed to delete chat session.")

def update_all_sessions():
    """Fetches and updates the list of all chat sessions."""
    st.session_state.all_sessions = sm.get_all_sessions_db()

# Password protection - Start
if 'password_correct' not in st.session_state:
    st.session_state['password_correct'] = False
if not st.session_state['password_correct']:
    entered_password = st.text_input("Password", type="password")
    if entered_password:
        if entered_password == secret_password:
            st.session_state['password_correct'] = True
            st.rerun()
        else:
            st.error("Incorrect password")
    st.stop()
# Password protection - End

# Initialize session/DBs (runs once per app startup)
sm.init_db()

# Initialize session state for chat management
if 'current_session_id' not in st.session_state or st.session_state.current_session_id is None:
    update_all_sessions()
    if st.session_state.all_sessions:
        load_chat_session(st.session_state.all_sessions[0])  # Load most recent
    else:  # No sessions exist, create a new one
        new_chat_session()
        st.rerun()

# Display current topic and chat messages from history
st.subheader(f"Current Chat: {st.session_state.current_session_topic}")
for message in st.session_state.get("messages", []):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- SIDEBAR ---
with st.sidebar:
    st.button("➕ New Chat", on_click=new_chat_session, use_container_width=True)
    st.markdown("---")

    st.subheader("Model Category")

    selected_category = option_menu(
        menu_title=None,
        options=list(all_models.keys()),
        icons=["✨", "🤖", "🔍"],  # Using emojis for broad compatibility
        orientation="vertical",
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": "orange", "font-size": "12px"},
            "nav-link": {"font-size": "10px", "text-align": "left", "margin": "0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#0288d1"},
        }
    )
    st.markdown("---")
    # (The loop for displaying past chats remains the same, as it uses st.session_state.all_sessions)
    st.subheader("Past Chats")
    if st.session_state.all_sessions:
        for session in st.session_state.all_sessions:
            is_current = (session["session_id"] == st.session_state.current_session_id)
            display_topic = session["topic"] if session["topic"] else "Untitled Chat"

            # Using st.columns for horizontal layout of session name and delete button
            col_topic, col_delete = st.columns([0.8, 0.2])
            with col_topic:
                if st.button(display_topic, key=f"load_{session['session_id']}", use_container_width=True,
                             type="primary" if is_current else "secondary"):
                    load_chat_session(session)
            with col_delete:
                if st.button("🗑️", key=f"delete_{session['session_id']}", help=f"Delete '{display_topic}'",
                             use_container_width=False
                             ):
                    st.session_state[f"confirm_delete_{session['session_id']}"] = True

            # Display confirmation dialog if triggered for this session
            if st.session_state.get(f"confirm_delete_{session['session_id']}", False):
                st.warning(f"Are you sure you want to delete '{display_topic}'? This cannot be undone.")
                col_confirm_yes, col_confirm_no = st.columns(2)
                with col_confirm_yes:
                    if st.button("Yes, Delete", key=f"confirm_yes_{session['session_id']}"):
                        delete_chat_session(session['session_id'])  # This will trigger a rerun
                with col_confirm_no:
                    if st.button("No, Cancel", key=f"confirm_no_{session['session_id']}"):
                        st.session_state[f"confirm_delete_{session['session_id']}"] = False  # Reset flag
                        st.rerun()  # Rerun to hide the dialog immediately
            else:
                # Add a small visual separation for sessions when no confirmation is shown
                st.markdown("")

    else:
        st.info("No past chats yet. Start a new conversation!")
    # ...

# --- MAIN CHAT LOGIC ---
if selected_category:
    # (Model selection UI remains the same)
    model_names = list(all_models[selected_category].keys())
    selected_model_name = st.selectbox("Select Model", model_names)
    selected_model_id = selected_model_name  # Or get specific ID for OpenAI/others
    # ...

    # Chat input
    prompt = st.chat_input("Say something")

    if "processing_prompt" not in st.session_state:
        st.session_state.processing_prompt = None

    # Step 1: Handle user input from the chat box
    if prompt:
        st.session_state.processing_prompt = prompt
        is_first_message = not st.session_state.get("messages", [])
        if is_first_message:
            with st.spinner("Generating chat topic..."):
                topic = gemini_module.generate_topic_from_text(selected_model_id, prompt)
                if topic and topic != "Untitled Conversation":
                    sm.update_session_topic_db(st.session_state.current_session_id, topic)
                    st.session_state.current_session_topic = topic
                    update_all_sessions()
                    st.rerun()

    # Step 2: Process the prompt for a chat response.
    if st.session_state.processing_prompt:
        current_prompt = st.session_state.processing_prompt
        history = lc_memory_module.get_chat_history(st.session_state.current_session_id)
        history.add_user_message(current_prompt)

        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(current_prompt)

        # Generate and display response
        with st.spinner(f"Getting response from {selected_model_name}..."):
            # Pass LangChain message objects directly to the model if it supports them,
            # otherwise convert them back to dicts if your gemini_module expects that.
            # 1. Get the current list of LangChain message objects
            messages_for_api = history.messages

            # 2. Call the function, which now returns a string
            assistant_content = gemini_module.generate_response(
                selected_model_id,
                messages_for_api
            )

            # 3. Handle the response and add it to history here
            if assistant_content:
                history.add_ai_message(assistant_content)
                with st.chat_message("assistant"):
                    st.markdown(assistant_content)
            else:
                # Handle cases where the model returns no content
                st.error("The model did not return a response.")

        # IMPORTANT: Clear the prompt and refresh messages from source of truth
        st.session_state.processing_prompt = None
        st.session_state.messages = _convert_messages_to_dict(history.messages)
        st.rerun()