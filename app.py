import os
import uuid
import streamlit as st
from streamlit_option_menu import option_menu
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

# PIL for local image previews (used in _display_queued_file_preview)
from PIL import Image

# --- MODULE IMPORTS (ensure all are present) ---
from modules import gemini_module, openai_module
from modules import session_manager_module as sm
from modules import lc_memory_module
from modules.gc_storage_manager import upload_file_to_gcs

# --- PAGE CONFIGURATION (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(
    page_title="LKF GenAI Chat",
    page_icon="🤖",
    layout="wide"
)

# --- CUSTOM CSS FOR UI/UX IMPROVEMENTS ---
st.markdown(
    """
    <style>
    /* Hide Streamlit's deploy button and toolbar for a cleaner look */
    .stAppDeployButton { visibility: hidden; }
    .stAppToolbar { visibility: hidden; }

    </style>
    """, unsafe_allow_html=True
)

# --- LOAD ENVIRONMENT VARIABLES ---
load_dotenv()

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

# Get GCS bucket name and signed URL duration from environment variables
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME")
SIGNED_URL_DURATION_SECONDS = int(os.environ.get("SIGNED_URL_DURATION_SECONDS", 300))


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
    Converts LangChain BaseMessage objects to a list of dicts for display.
    Handles multimodal content by generating HTML for files/images for display within markdown.
    """
    display_messages = []
    for msg in messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        content = msg.content
        formatted_content_parts = []

        if isinstance(content, list):
            # This is a multimodal message (text + potentially file metadata dictionaries)
            for part in content:
                if isinstance(part, str):
                    formatted_content_parts.append(part) # Add text directly
                elif isinstance(part, dict):
                    # This is assumed to be a processed_file_metadata dict (e.g., {"name":..., "mime_type":..., "display_url":...})
                    if part.get("mime_type", '').startswith("image/") and part.get("display_url"):
                        formatted_content_parts.append(
                            f'<div style="text-align: left; margin-top: 10px; display: flex; align-items: center; gap: 10px;">'
                            f'<img src="{part["display_url"]}" style="max-width: 100px; border-radius: 8px; object-fit: contain;"/>'
                            f'<p style="font-size: 0.9em; margin: 0; word-wrap: break-word;">{part.get("name", "Image")}</p>'
                            f'</div>'
                        )
                    elif part.get("mime_type", '') == "application/pdf" and part.get("display_url"):
                        formatted_content_parts.append(
                            f'<div style="text-align: left; margin-top: 10px; display: flex; align-items: center; gap: 10px;">'
                            f'<a href="{part["display_url"]}" target="_blank" style="text-decoration: none;">'
                            f'<img src="https://upload.wikimedia.org/wikipedia/commons/8/87/PDF_file_icon.svg" style="width: 60px; height: 60px; object-fit: contain;"/>'
                            f'</a>'
                            f'<p style="font-size: 0.9em; margin: 0; word-wrap: break-word;">PDF: <a href="{part["display_url"]}" target="_blank">{part.get("name", "Document")}</a></p>'
                            f'</div>'
                        )
                    elif part.get("display_url"): # For other file types with a display URL
                        formatted_content_parts.append(
                            f'<div style="text-align: left; margin-top: 10px; display: flex; align-items: center; gap: 10px;">'
                            f'<a href="{part["display_url"]}" target="_blank" style="text-decoration: none;">'
                            f'<img src="https://cdn-icons-png.flaticon.com/512/3796/3796062.png" style="width: 60px; height: 60px; object-fit: contain;"/>' # Generic file icon
                            f'</a>'
                            f'<p style="font-size: 0.9em; margin: 0; word-wrap: break-word;">File: <a href="{part["display_url"]}" target="_blank">{part.get("name", "Document")}</a></p>'
                            f'</div>'
                        )
                    else:
                        formatted_content_parts.append(f"*(File: {part.get('name', 'Unknown file')}, no display link)*")
                else:
                    formatted_content_parts.append(f"*(Unrecognized content part: {type(part)})*")
            # Join all parts. st.markdown will render the HTML.
            display_content_str = "".join(formatted_content_parts)
        else:
            # Simple string content (typically AI messages or old plain user messages)
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
    st.session_state["uploader_key"] = str(uuid.uuid4())  # Unique key for file uploader to reset state
    update_all_sessions()


def load_chat_session(session_data):
    """Loads a specific chat session from the database and refreshes the UI."""
    st.session_state.current_session_id = session_data["session_id"]
    st.session_state.current_session_topic = session_data["topic"]
    # Assuming lc_memory_module's get_chat_history returns LangChain messages
    history = lc_memory_module.get_chat_history(session_data["session_id"])
    st.session_state.messages = _convert_messages_to_dict(history.messages)  # Convert for display
    # Clear any pending processing state from previous interaction that might be left
    st.session_state["uploader_key"] = str(uuid.uuid4())  # Reset uploader key to a new unique value
    st.rerun()


def delete_chat_session(session_id: str):
    """Deletes a chat session and its history, then reloads the UI."""
    if sm.delete_session_db(session_id):
        st.success("Chat session deleted.")
        # Reset current session if it was the one deleted
        if st.session_state.current_session_id == session_id:
            st.session_state.current_session_id = None
            # Attempt to load first session, or create a new one
            update_all_sessions()
            if st.session_state.all_sessions:
                load_chat_session(st.session_state.all_sessions[0])
            else:
                new_chat_session()
        else:
            update_all_sessions() # Update list without changing current session
            st.rerun() # Rerun to update sidebar
    else:
        st.error("Failed to delete chat session.")


def update_all_sessions():
    """Fetches and updates the list of all chat sessions in session state."""
    user_email = st.user.email
    st.session_state.all_sessions = sm.get_sessions_for_user_db(user_id=user_email)


def _process_uploaded_file_for_gcs(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile, user_name: str):
    """
    Upload file to GCS and return its processed metadata (GCS URI, display URL etc.).
    """
    gcs_uri, display_url = upload_file_to_gcs(uploaded_file, user_name)  # Pass bucket name
    if gcs_uri and display_url:
        return {
            "name": uploaded_file.name,
            "mime_type": uploaded_file.type,
            "gcs_uri": gcs_uri,
            "display_url": display_url,  # The signed URL for direct browser display
            "size": uploaded_file.size
        }
    else:
        st.warning(f"Failed to upload '{uploaded_file.name}'.")
        return {"error": f"Failed upload for {uploaded_file.name}"}


def _display_queued_file_preview(file_obj: st.runtime.uploaded_file_manager.UploadedFile, idx: int):
    """Displays a small preview/thumbnail for a file in the queue."""
    col_icon, col_name, col_remove = st.columns([0.1, 0.8, 0.1])
    with col_icon:
        if file_obj.type and file_obj.type.startswith('image/'):
            file_obj.seek(0)
            image = Image.open(file_obj)
            st.image(image, width=50)
            file_obj.seek(0)
        elif file_obj.type == "application/pdf":
            st.image("https://upload.wikimedia.org/wikipedia/commons/8/87/PDF_file_icon.svg", width=50, caption="PDF")
        elif file_obj.type == "text/plain":
            st.image(
                "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/Text-x-generic.svg/200px-Text-x-generic.svg.png",
                width=50, caption="TXT")
        else:
            st.image("https://cdn-icons-png.flaticon.com/512/3796/3796062.png", width=50) # Generic file icon
    with col_name:
        st.markdown(f"**{file_obj.name}**")
        st.caption(f"{file_obj.type} - {file_obj.size / 1024:.1f} KB")
    with col_remove:
        if st.button("x", key=f"remove_file_{idx}", help="Remove this file"):
            st.session_state.uploaded_files_queue.pop(idx)
            st.rerun()
# ----------------------------------
# --- MAIN APPLICATION LOGIC ---
# ----------------------------------

# Use the built-in Streamlit authentication to gate the app.
# Note: For production, replace st.user with actual SSO integration.
if not st.user.is_logged_in:
    login_screen()
    st.stop()

# --- AUTHENTICATED APPLICATION ---

# Initialize databases once per session
if 'db_initialized' not in st.session_state:
    with st.spinner("Initializing resources..."):
        try:
            sm.init_db()
            lc_memory_module.create_history_table_if_not_exists()
        except Exception as e:
            st.error(f"Error initializing database: {e}")
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
    st.write("") # Spacer
    st.button("Log out", on_click=st.logout, use_container_width=True)

st.subheader(f"Current Chat: {st.session_state.current_session_topic}")

# Display chat messages from history
for message in st.session_state.get("messages", []):
    with st.chat_message(message["role"]):
        # Crucial: Allow HTML for rendering thumbnails and links in the history
        st.markdown(message["content"], unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.subheader(f"Welcome, {st.user.name.split('@')[0].capitalize()}!")  # Display name without domain
    st.markdown("---")
    st.button("➕ New Chat", on_click=new_chat_session, type="primary", use_container_width=True)
    st.markdown("---")
    st.subheader("Model Category")

    selected_category = option_menu(menu_title=None, options=list(ALL_MODELS.keys()), icons=["✨", "🤖"],
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
                st.warning(f"Are you sure you want to delete '{display_topic}'?")
                col_yes, col_no = st.columns(2)
                if col_yes.button("Yes", key=f"confirm_yes_{session['session_id']}", use_container_width=True):
                    delete_chat_session(session['session_id'])  # Deletes and reruns
                if col_no.button("No", key=f"confirm_no_{session['session_id']}", use_container_width=True):
                    st.session_state[f"confirm_delete_{session['session_id']}"] = False
                    st.rerun() # Rerun to hide confirmation
    else:
        st.info("No past chats yet.")

# --- MAIN CHAT INTERACTION LOGIC ---
if selected_category:
    model_names = list(ALL_MODELS[selected_category].keys())
    selected_model_name = st.selectbox("Select Model", model_names)
    selected_model_id = selected_model_name  # Model ID is same as selected name here

    # Model-specific options
    grounding_source = False
    if selected_category == "Gemini":
        grounding_source = st.checkbox("Enable Grounding Source: Google Search")

    # --- File Upload Section ---
    st.markdown("---") # Visual separator
    with st.expander("Upload Documents/Images for Context"):
        # Allow multiple files to be uploaded
        uploaded_files = st.file_uploader(
            "Choose files...",
            type=["jpg", "jpeg", "png", "webp", "pdf", "txt"],
            accept_multiple_files=True,  # Allow multiple files
            key=st.session_state['uploader_key']
        )

    # --- Text Chat Input (Submission Trigger) ---
    user_text_prompt = st.chat_input("Say something")

    if user_text_prompt: # The entire prompt submission logic starts here
        processed_file_metadata = [] # To store GCS URIs, display URLs, etc.
        # Process and upload files to GCS at submission time
        if uploaded_files:
            with st.spinner("Uploading files to Cloud Storage..."):
                for file_obj in uploaded_files:
                    processed_content = _process_uploaded_file_for_gcs(file_obj, st.user.name)
                    if processed_content and "error" not in processed_content:
                        processed_file_metadata.append(processed_content)
                    else:
                        st.error(f"Skipping file '{file_obj.name}' due to processing/upload error.")

        # --- Display the user's current input and files in current chat bubble ---
        with st.chat_message("user"):
            if user_text_prompt:
                st.markdown(user_text_prompt)

            if processed_file_metadata:
                st.markdown("---")
                st.write("Attached Files:")
                cols_per_row = 4
                cols = st.columns(cols_per_row)
                for idx, file_meta in enumerate(processed_file_metadata):
                    with cols[idx % cols_per_row]:
                        if file_meta['mime_type'].startswith('image/') and file_meta.get('display_url'):
                            st.image(file_meta['display_url'], caption=file_meta['name'], width=100)
                        elif file_meta.get('display_url'):
                            # For non-images, provide a clickable link
                            st.markdown(f"📄 [{file_meta['name']}]({file_meta['display_url']})")
                        else:
                            st.markdown(f"🔗 {file_meta['name']}")

        # --- Prepare the full user message content for storage in history_manager ---
        final_human_message_content_parts = []
        if user_text_prompt:
            final_human_message_content_parts.append(user_text_prompt)

        # Add processed file metadata directly to the content list of the HumanMessage
        for file_meta in processed_file_metadata:
            final_human_message_content_parts.append(file_meta)

        # Get the current history manager to add the new message
        history_manager = lc_memory_module.get_chat_history(st.session_state.current_session_id)

        # Check if there's *anything* meaningful to send to the LLM (text XOR files).
        if not final_human_message_content_parts:
            st.warning("No text or supported file content to send to LLM. Please provide a prompt or valid file.")
            # Clear state and rerun for a clean UI
            del st.session_state['processing_prompt_and_file']
            # Re-fetch messages to ensure the display reflects only previous state
            st.session_state.messages = _convert_messages_to_dict(history_manager.messages)
            st.rerun()

        # Add the full user message (text + file metadata) to the chat history
        history_manager.add_user_message(HumanMessage(content=final_human_message_content_parts))

        # Handle topic generation for new chats
        # Check if this is the very first message after creating a new chat session
        # A simple check: if history contains only 1 message (the one just added by user)
        if len(history_manager.messages) == 1:
            with st.spinner("Generating chat topic..."):
                topic_text_for_generation = user_text_prompt if user_text_prompt else f"New chat with file(s)"
                if selected_category == "Gemini":
                    topic = gemini_module.generate_topic_from_text(selected_model_id, topic_text_for_generation)
                elif selected_category == "OpenAI":
                    # Assume openai_module also has generate_topic_from_text
                    topic = openai_module.generate_topic_from_text(selected_model_id, topic_text_for_generation)
                else:
                    topic = "Untitled Chat" # Fallback

                if topic and topic != "Untitled Conversation": # "Untitled Conversation" is often a default fallback
                    sm.update_session_topic_db(st.session_state.current_session_id, topic)
                    st.session_state.current_session_topic = topic
                    update_all_sessions() # Refresh sidebar topics

        with st.spinner(f"Getting response from {selected_model_name}..."):
            assistant_content_generator = None
            # Pass processed_file_metadata containing GCS URIs, display URLs etc.
            # The `generate_response` functions in modules should be able to handle
            # LangChain BaseMessage objects (which might have list content now) for history
            # and the separate `processed_files` for new file inputs for the LLM.
            if selected_category == "Gemini":
                assistant_content_generator = gemini_module.generate_response(
                    selected_model_id,
                    history_manager.messages,  # History now contains text + file metadata
                    grounding_source=grounding_source
                )
            elif selected_category == "OpenAI":
                assistant_content_generator = openai_module.generate_response(
                    selected_model_id,
                    history_manager.messages, # History now contains text + file metadata
                )
            else:
                st.warning(f"Response generation not implemented for {selected_category} models.")
                # Fallback display for unsupported models
                with st.chat_message("assistant"):
                    st.write("Sorry, this model category is not yet supported for responses.")
                assistant_content_generator = None # Ensure it's None to avoid subsequent error

            if assistant_content_generator:
                with st.chat_message("assistant"):
                    # st.write_stream displays the content incrementally and returns the full string
                    full_response_text = st.write_stream(assistant_content_generator)
                history_manager.add_ai_message(full_response_text)
            else:
                st.error("The model did not return a response or an error occurred during generation.")

        # del st.session_state['processing_prompt_and_file']
        st.session_state["uploader_key"] = str(uuid.uuid4())  # Reset uploader key to a new unique value
        st.session_state.messages = _convert_messages_to_dict(
            lc_memory_module.get_chat_history(st.session_state.current_session_id).messages
        )
        st.session_state.clear()
        st.rerun() # Final rerun to update chat UI with AI response and clear inputx/