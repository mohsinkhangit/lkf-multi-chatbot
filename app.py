import os
import streamlit as st
from streamlit_option_menu import option_menu
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

import base64
from pypdf import PdfReader
import io
# --- MODULE IMPORTS (ensure all are present) ---
from modules import gemini_module, openai_module
from modules import session_manager_module as sm
from modules import lc_memory_module
from modules.db_connection_manager import get_db_connection

# --- PAGE CONFIGURATION (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(
    page_title="LKF GenAI Chat",
    page_icon="🤖",
    layout="wide"
)

# Hide Streamlit's default UI elements for a cleaner look
st.markdown(
    r"""
    <style>
    .stAppDeployButton { visibility: hidden; }
    .stAppToolbar { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True
)

# --- LOAD ENVIRONMENT VARIABLES ---
load_dotenv()

# --- MODEL AND CONSTANT DEFINITIONS ---
GEMINI_MODELS = {
    "gemini-2.0-flash-lite-001": "gemini-2.0-flash-lite-001",
    "gemini-2.0-flash-001": "gemini-2.0-flash-001",
    "gemini-2.5-pro": "gemini-2.5-pro",
    "gemini-2.5-flash": "gemini-2.5-flash"
}
OPENAI_MODELS = {
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini"
}
ALL_MODELS = {
    "Gemini": GEMINI_MODELS,
    "OpenAI": OPENAI_MODELS,
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
    """
    Converts LangChain message BaseMessage objects to a list of dicts for display.
    Handles multimodal content by focusing on its string representation.
    NOTE: With the new strategy, st.session_state.messages will primarily hold text,
    but this function is robust enough to handle lists if they appear.
    """
    display_messages = []
    for msg in messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        content = msg.content

        if isinstance(content, list):
            # If content is a list (e.g., from a past multimodal message if one was ever stored directly)
            # Join parts for display, or show specific placeholders.
            display_parts = []
            for part in content:
                if isinstance(part, str):
                    display_parts.append(part)
                elif isinstance(part, dict):
                    if part.get("mime_type", '').startswith("image/"):
                        display_parts.append(f"*(Image: {part.get('source_file', 'uploaded')})*")
                    elif part.get("mime_type", '') == "text/plain":
                        display_parts.append(f"*(Document text: {part.get('source_file', 'uploaded')})*")
                    else:
                        display_parts.append(f"*(Complex content)*")
                else:
                    display_parts.append(f"*(Unknown content type)*")
            display_content_str = " ".join(display_parts)
        else:
            # Simple string content
            display_content_str = content

        display_messages.append({"role": role, "content": display_content_str})

    return display_messages


def new_chat_session():
    """Initializes a new chat session for the currently logged-in user."""
    user_email = st.user.email
    session_id = sm.create_new_session_db(user_id=user_email)
    st.session_state.current_session_id = session_id
    st.session_state.current_session_topic = "New Chat"
    st.session_state.messages = []
    st.session_state.processing_prompt_and_file = {"prompt": None, "files": []}
    st.session_state.uploaded_files_queue = []  # Ensure queue is empty
    update_all_sessions()


def load_chat_session(session_data):
    """Loads a specific chat session from the database and refreshes the UI."""
    st.session_state.current_session_id = session_data["session_id"]
    st.session_state.current_session_topic = session_data["topic"]
    history = lc_memory_module.get_chat_history(session_data["session_id"])
    st.session_state.messages = _convert_messages_to_dict(history.messages)
    # Clear any pending processing state from previous interaction that might be left
    st.session_state.processing_prompt_and_file = {"prompt": None, "files": []}
    st.session_state.uploaded_files_queue = []  # Ensure queue is empty
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


def _process_uploaded_file(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile):
    """
    Process an uploaded file and return its content in a format suitable for LLM APIs.
    """
    file_bytes = uploaded_file.getvalue()
    file_name = uploaded_file.name

    if uploaded_file.type.startswith("image/"):
        return {
            "file_type": "image_url",  # Indicate this is an image URL type
            "mime_type": uploaded_file.type,
            "content": base64.b64encode(file_bytes).decode('utf-8'),
            "source_file": file_name
        }
    elif uploaded_file.type == "application/pdf":
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return {
                "file_type": "document_text",  # Indicate this is PDF text content
                "mime_type": "application/pdf",
                "content": text,
                "source_file": file_name
            }
        except Exception as e:
            st.error(f"Error extracting text from PDF '{file_name}': {e}")
            return {"type": "document_error", "content": f"Could not process PDF: {file_name}", "error": str(e)}
    elif uploaded_file.type == "text/plain":
        return {
            "file_type": "document_text",  # Indicate this is plain text content
            "mime_type": "text/plain",
            "content": file_bytes.decode('utf-8'),
            "source_file": file_name
        }
    else:
        st.warning(f"Unsupported file type: {uploaded_file.type} for file '{file_name}'")
        return {"type": "unsupported_file", "content": f"File {file_name} is of unsupported type {uploaded_file.type}"}


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
        try:
            sm.init_db()
            lc_memory_module.create_history_table_if_not_exists()
        except Exception as e:
            st.error(f"Error initializing database")
            st.stop()
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
    st.button("➕ New Chat", on_click=new_chat_session, type="primary", use_container_width=True)
    st.markdown("---")
    st.subheader("Model Category")
    # Added "LLAMA" and "DeepSeek" back to options, mapping to generic icons.
    selected_category = option_menu(menu_title=None, options=list(ALL_MODELS.keys()), icons=["✨", "🤖", "🦙", "🔍"],
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
    selected_model_id = selected_model_name

    # Model-specific options
    grounding_source = False
    if selected_category == "Gemini":
        grounding_source = st.checkbox("Enable Grounding Source: Google Search")
    # For OpenAI or other models, add their specific options here if needed:
    # elif selected_category == "OpenAI":
    #     some_openai_option = st.checkbox("Enable some OpenAI feature")

    # --- File Upload Section ---
    st.markdown("---")  # Visual separator
    with st.expander("Upload Documents/Images for Context", expanded=False):
        # Allow multiple files to be uploaded
        uploaded_files = st.file_uploader(
            "Choose files...",
            type=["jpg", "jpeg", "png", "webp", "pdf", "txt"],
            accept_multiple_files=True,  # Allow multiple files
            key="multi_file_uploader"  # Use a unique key
        )

        # Add new uploaded files to the queue
        if uploaded_files:
            for file in uploaded_files:
                # Only add if not already in queue (based on name and size as a simple check)
                if not any(f.name == file.name and f.size == file.size for f in st.session_state.uploaded_files_queue):
                    st.session_state.uploaded_files_queue.append(file)

        # Display files currently in the queue
        if st.session_state.uploaded_files_queue:
            st.write("--- Files ready for submission ---")
            # for idx, file in enumerate(st.session_state.uploaded_files_queue):
            #     col_file_name, col_remove = st.columns([0.9, 0.1])
            #     with col_file_name:
            #         st.markdown(f"- {file.name} ({file.type})")
            #     with col_remove:
            #         if st.button("x", key=f"remove_file_{idx}", help="Remove this file"):
            #             st.session_state.uploaded_files_queue.pop(idx)
            #             st.rerun()  # Refresh to update the list
            # if st.button("Clear all selected files", key="clear_all_files"):
            #     st.session_state.uploaded_files_queue = []
            #     st.rerun()

    # --- Text Chat Input (Submission Trigger) ---
    user_text_prompt = st.chat_input("Say something")

    # This 'if prompt:' statement now acts as the sole trigger for processing.
    if user_text_prompt:  # The entire prompt submission logic starts here

        # Store the current input (text and files) into a dedicated processing state
        st.session_state.processing_prompt_and_file = {
            "prompt": user_text_prompt,
            "files": st.session_state.uploaded_files_queue  # Get all files currently in queue
        }
        # Immediately clear the queue since they are now attached to the prompt
        st.session_state.uploaded_files_queue = []

        # --- EXECUTE PROCESSING ---
        # This part processes both text and files stored in 'processing_prompt_and_file'

        current_input_data_for_llm = st.session_state.processing_prompt_and_file
        current_text_prompt_for_llm = current_input_data_for_llm["prompt"]
        current_files_for_llm = current_input_data_for_llm["files"]

        # Prepare processed file content for LLM calls (will be passed separately)
        processed_llm_file_data = []

        # Process and add file content only to 'processed_llm_file_data'
        if current_files_for_llm:
            for file_obj in current_files_for_llm:
                processed_content = _process_uploaded_file(file_obj)
                if processed_content and "error" not in processed_content:
                    processed_llm_file_data.append(processed_content)
                elif "error" in processed_content:
                    st.error(f"Skipping file '{file_obj.name}' due to processing error: {processed_content['error']}")

        # Display user's input in chat history BEFORE LLM call
        with st.chat_message("user"):
            if current_text_prompt_for_llm:
                st.markdown(current_text_prompt_for_llm)
            if current_files_for_llm:
                for file_obj in current_files_for_llm:
                    st.info(f"Attached: {file_obj.name} ({file_obj.type})")
                    # Optional: display raw image / PDF preview here based on type

        # Handle topic generation for new chats
        history_manager = lc_memory_module.get_chat_history(st.session_state.current_session_id)
        is_first_message_in_new_chat = not history_manager.messages  # Check if history is genuinely empty in DB

        if is_first_message_in_new_chat:
            with st.spinner("Generating chat topic..."):
                topic_text_for_generation = current_text_prompt_for_llm if current_text_prompt_for_llm else f"New chat with file(s)"
                if selected_category == "Gemini":
                    topic = gemini_module.generate_topic_from_text(selected_model_id, topic_text_for_generation)
                elif selected_category == "OpenAI":
                    topic = openai_module.generate_topic_from_text(selected_model_id,
                                                                   topic_text_for_generation)  # Assuming openai_module has this
                else:
                    topic = "Untitled Chat"  # Fallback for unsupported categories

                if topic and topic != "Untitled Conversation":
                    sm.update_session_topic_db(st.session_state.current_session_id, topic)
                    st.session_state.current_session_topic = topic
                    update_all_sessions()
                    # CRITICAL: This rerun is what causes the loop if not handled correctly.
                    # It should NOT be here if we want to proceed to LLM call in the same turn.
                    # Instead, allow the script to continue to the LLM call below.
                    # st.rerun() # REMOVE THIS LINE

        # Add the current user's text message ONLY to History. Files are separate.
        # Only add if there's actual text content to add to history.
        if current_text_prompt_for_llm:
            history_manager.add_user_message(HumanMessage(content=current_text_prompt_for_llm))

        # Check if there's *anything* to send to the LLM (text XOR files).
        if not (current_text_prompt_for_llm or processed_llm_file_data):
            st.warning("No text or supported file content to send to LLM. Please provide a prompt or valid file.")
            del st.session_state['processing_prompt_and_file']  # Clear state
            st.rerun()  # Clean up UI

        # Generate response from selected model
        with st.spinner(f"Getting response from {selected_model_name}..."):
            assistant_content = ""
            # IMPORTANT: Now pass processed_llm_file_data as a separate argument.
            # Your gemini_module.generate_response and openai_module.generate_response
            # functions MUST be updated to accept and process this new argument.
            if selected_category == "Gemini":
                assistant_content = gemini_module.generate_response(
                    selected_model_id,
                    history_manager.messages,  # Text-only history
                    processed_files=processed_llm_file_data,  # New argument for files
                    grounding_source=grounding_source
                )
            elif selected_category == "OpenAI":
                assistant_content = openai_module.generate_response(
                    selected_model_id,
                    history_manager.messages,  # Text-only history
                    processed_files=processed_llm_file_data  # New argument for files
                )
            else:
                st.warning(f"Response generation not implemented for {selected_category} models.")
                assistant_content = "Sorry, this model category is not yet supported for responses."

            if assistant_content:
                history_manager.add_ai_message(assistant_content)
                with st.chat_message("assistant"):
                    st.markdown(assistant_content)
            else:
                st.error("The model did not return a response.")

        # Clear processing state and refresh UI
        del st.session_state['processing_prompt_and_file']
        st.session_state.messages = _convert_messages_to_dict(history_manager.messages)
        st.rerun()  # Final re