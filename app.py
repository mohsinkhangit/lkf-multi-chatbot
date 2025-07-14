import os
import streamlit as st
from streamlit_option_menu import option_menu
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

# --- PAGE CONFIGURATION (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(
    page_title="LKF GenAI Chat",
    page_icon="🤖",
    layout="wide"
)

# --- LOAD ENVIRONMENT VARIABLES ---
load_dotenv()

# --- MODULE IMPORTS ---
from modules import gemini_module
from modules import session_manager_module as sm
from modules import lc_memory_module

# --- MODEL AND CONSTANT DEFINITIONS ---
GEMINI_MODELS = {
    "gemini-2.0-flash-lite-001": "gemini-2.0-flash-lite-001",
    "gemini-2.0-flash-001": "gemini-2.0-flash-001",
    "gemini-2.5-pro": "gemini-2.5-pro",
    "gemini-1.5-flash": "Gemini 1.5 Flash"
}
OPENAI_MODELS = {
    "GPT-4o": {"name": "gpt-4o", "deployment_name": "gpt-4o-2", "api_version": "2024-08-01-preview"},
    "GTP-4o-mini": {"name": "gpt-4o-mini", "deployment_name": "gpt-4o-mini", "api_version": "2024-08-01-preview"}
}
ALL_MODELS = {
    "Gemini": GEMINI_MODELS,
    "OpenAI": OPENAI_MODELS,
    "DeepSeek": {"DeepSeek": "DeepSeek"},
    "LLAMA": {"LLAMA": "LLAMA"},
}


# --- HELPER & CALLBACK FUNCTIONS ---

def login_screen():
    """Displays a centered login screen with a logo and login button."""
    _, col2, _ = st.columns([1, 2, 1])
    with col2:
        try:
            col2.image("assets/header-logo.svg", use_container_width=True)
        except Exception:
            st.warning("Logo image not found. Place 'header-logo.svg' in the 'assets' folder.")

        st.markdown("<h1 style='text-align: center;'>Welcome to LKF AI Chat Portal</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Please log in with your Microsoft account to continue.</p>",
                    unsafe_allow_html=True)
        st.button("Log in with Microsoft", on_click=st.login, type="primary", use_container_width=True)


def _convert_messages_to_dict(messages) -> list[dict]:
    """Converts LangChain message objects to a list of dicts for display."""
    return [{"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content} for m in messages]


def new_chat_session():
    """Initializes a new chat session for the currently logged-in user."""
    # --- CHANGE ---
    # Get the user's email from the st.user object.
    user_email = st.user.email
    session_id = sm.create_new_session_db(user_id=user_email)
    st.session_state.current_session_id = session_id
    st.session_state.current_session_topic = "New Chat"
    st.session_state.messages = []

    # We need to refresh the session list to show the "New Chat" session
    update_all_sessions()

    st.rerun()


def load_chat_session(session_data):
    """Loads a specific chat session from the database and refreshes the UI."""
    st.session_state.current_session_id = session_data["session_id"]
    st.session_state.current_session_topic = session_data["topic"]
    history = lc_memory_module.get_chat_history(session_data["session_id"])
    st.session_state.messages = _convert_messages_to_dict(history.messages)
    st.rerun()


def delete_chat_session(session_id: str):
    """Deletes a chat session and its history, then reloads the UI."""
    if sm.delete_session_db(session_id):
        st.success("Chat session deleted.")
        st.session_state.current_session_id = None
        update_all_sessions()
        st.rerun()
    else:
        st.error("Failed to delete chat session.")


def update_all_sessions():
    """Fetches and updates the list of all chat sessions in session state."""
    user_email = st.user.email
    st.session_state.all_sessions = sm.get_sessions_for_user_db(user_id=user_email)


# ----------------------------------
# --- MAIN APPLICATION LOGIC ---
# ----------------------------------

# Use the built-in Streamlit authentication to gate the app
if not st.user.is_logged_in:
    login_screen()
    st.stop()

# --- AUTHENTICATED APPLICATION ---
# This code runs only after a user has successfully logged in.

# Initialize databases once per session
if 'db_initialized' not in st.session_state:
    with st.spinner("Initializing resources..."):
        sm.init_db()
        lc_memory_module.create_history_table_if_not_exists()
    st.session_state.db_initialized = True

# Initialize chat session state if it doesn't exist
if 'current_session_id' not in st.session_state:
    update_all_sessions()
    if st.session_state.all_sessions:
        load_chat_session(st.session_state.all_sessions[0])
    else:
        new_chat_session()

# --- HEADER AND UI LAYOUT ---
col_title, col_logout = st.columns([4, 1])
with col_title:
    st.title("LKF GenAI Chat")
with col_logout:
    st.write("")  # Spacer
    st.button("Log out", on_click=st.logout, use_container_width=True)

st.subheader(f"Current Chat: {st.session_state.current_session_topic}")

# Display chat messages from history
for message in st.session_state.get("messages", []):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- SIDEBAR ---
with st.sidebar:
    st.subheader(f"Welcome, {st.user.name}!")
    st.markdown("---")
    st.button("➕ New Chat", on_click=new_chat_session, use_container_width=True)
    st.markdown("---")
    st.subheader("Model Category")
    selected_category = option_menu(menu_title=None, options=list(ALL_MODELS.keys()), icons=["✨", "🤖", "🔍", "🦙"],
                                    styles={"nav-link": {"font-size": "14px"}})
    st.markdown("---")
    st.subheader("Past Chats")

    if 'all_sessions' in st.session_state and st.session_state.all_sessions:
        for session in st.session_state.all_sessions:
            is_current = (session["session_id"] == st.session_state.current_session_id)
            display_topic = session["topic"] if session["topic"] else "Untitled Chat"
            col_topic, col_delete = st.columns([0.8, 0.2])
            with col_topic:
                if st.button(display_topic, key=f"load_{session['session_id']}",
                             type="primary" if is_current else "secondary", use_container_width=True):
                    load_chat_session(session)
            with col_delete:
                if st.button("🗑️", key=f"delete_{session['session_id']}", help=f"Delete '{display_topic}'"):
                    st.session_state[f"confirm_delete_{session['session_id']}"] = True

            if st.session_state.get(f"confirm_delete_{session['session_id']}", False):
                st.warning(f"Delete '{display_topic}'?")
                col_yes, col_no = st.columns(2)
                if col_yes.button("Yes", key=f"confirm_yes_{session['session_id']}", use_container_width=True):
                    delete_chat_session(session['session_id'])
                if col_no.button("No", key=f"confirm_no_{session['session_id']}", use_container_width=True):
                    st.session_state[f"confirm_delete_{session['session_id']}"] = False
                    st.rerun()
    else:
        st.info("No past chats yet.")

# --- MAIN CHAT INTERACTION LOGIC ---
if selected_category:
    model_names = list(ALL_MODELS[selected_category].keys())
    selected_model_name = st.selectbox("Select Model", model_names)
    selected_model_id = selected_model_name  # Defaulting to name, adjust if needed for specific APIs

    prompt = st.chat_input("Say something")

    if prompt and 'processing_prompt' not in st.session_state:
        st.session_state.processing_prompt = prompt

        if not st.session_state.get("messages"):
            with st.spinner("Generating chat topic..."):
                topic = gemini_module.generate_topic_from_text(selected_model_id, prompt)
                if topic and topic != "Untitled Conversation":
                    sm.update_session_topic_db(st.session_state.current_session_id, topic)
                    st.session_state.current_session_topic = topic
                    update_all_sessions()
                    st.rerun()

    if st.session_state.get('processing_prompt'):
        current_prompt = st.session_state.processing_prompt
        history = lc_memory_module.get_chat_history(st.session_state.current_session_id)
        history.add_user_message(current_prompt)

        with st.chat_message("user"):
            st.markdown(current_prompt)

        with st.spinner(f"Getting response from {selected_model_name}..."):
            assistant_content = gemini_module.generate_response(selected_model_id, history.messages)
            if assistant_content:
                history.add_ai_message(assistant_content)
                with st.chat_message("assistant"):
                    st.markdown(assistant_content)
            else:
                st.error("The model did not return a response.")

        del st.session_state['processing_prompt']
        st.session_state.messages = _convert_messages_to_dict(history.messages)
        st.rerun()