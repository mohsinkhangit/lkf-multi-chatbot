# modules/lc_memory_module.py
from langchain_postgres import PostgresChatMessageHistory
from  .message_encryption import EncryptedPostgresChatMessageHistory

# --- CHANGE: Import from the new connection manager ---
from modules.db_connection_manager import get_db_connection

# Define the table name as a constant for clarity and consistency
TABLE_NAME = "chat_message_history"

def get_chat_history(session_id: str) -> EncryptedPostgresChatMessageHistory:
    """
    Returns a LangChain chat history object for a given session_id.
    This object uses the app's single, cached database connection.
    """
    # Get the shared database connection from the new module
    sync_conn = get_db_connection()

    # The constructor requires table_name and session_id as positional arguments,
    # and the connection as a keyword argument.
    return EncryptedPostgresChatMessageHistory(
        TABLE_NAME,
        session_id,
        sync_connection=sync_conn
    )

def create_history_table_if_not_exists():
    """
    A helper function to create the message history table on app startup.
    This prevents errors if the table doesn't exist yet.
    """
    conn = get_db_connection()
    EncryptedPostgresChatMessageHistory.create_tables(conn, TABLE_NAME)