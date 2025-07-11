# modules/session_manager_module.py
import uuid
import psycopg
from typing import List, Dict

# --- CHANGE: Import from the new connection manager ---
from modules.db_connection_manager import get_db_connection

# --- This import is now safe as the dependency is one-way ---
from modules import lc_memory_module


def init_db():
    """
    Initializes the 'sessions' table using the shared connection.
    """
    sql = """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id UUID PRIMARY KEY,
            topic VARCHAR(255),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
    """
    # Get the shared connection
    conn = get_db_connection()
    try:
        # Cursor is a context manager, which is fine
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()  # Manually commit the transaction
    except psycopg.Error as e:
        print(f"Database initialization error: {e}")
        conn.rollback()  # Rollback on error
        raise


def create_new_session_db() -> str:
    """
    Creates a new session record and returns its UUID string.
    """
    sql = "INSERT INTO sessions (session_id, topic) VALUES (%s, %s) RETURNING session_id"
    new_uuid = uuid.uuid4()
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (new_uuid, "New Chat"))
            session_id = cur.fetchone()[0]
        conn.commit()
        return str(session_id)
    except psycopg.Error as e:
        print(f"Error creating new session: {e}")
        conn.rollback()
        raise


def get_all_sessions_db() -> List[Dict[str, str]]:
    """
    Retrieves all chat sessions. Read operations don't need commit/rollback.
    """
    sql = "SELECT session_id, topic FROM sessions ORDER BY created_at DESC"
    conn = get_db_connection()
    with conn.cursor() as cur:
        cur.execute(sql)
        sessions = [
            {"session_id": str(row[0]), "topic": row[1]}
            for row in cur.fetchall()
        ]
    return sessions


def update_session_topic_db(session_id: str, topic: str):
    """
    Updates the topic for a given session.
    """
    sql = "UPDATE sessions SET topic = %s WHERE session_id = %s"
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (topic, uuid.UUID(session_id)))
        conn.commit()
    except psycopg.Error as e:
        print(f"Error updating session topic: {e}")
        conn.rollback()
        raise


def delete_session_db(session_id: str) -> bool:
    """
    Deletes a session record and clears its associated messages via LangChain.
    """
    sql = "DELETE FROM sessions WHERE session_id = %s"
    conn = get_db_connection()
    try:
        # Step 1: Delete from our sessions table
        with conn.cursor() as cur:
            cur.execute(sql, (uuid.UUID(session_id),))

        # Step 2: Use lc_memory_module to clear its messages.
        history = lc_memory_module.get_chat_history(session_id)
        history.clear()  # This method handles its own commit.

        # Step 3: Commit the deletion for our sessions table.
        conn.commit()

        print(f"Successfully deleted session {session_id}")
        return True
    except (Exception, psycopg.Error) as e:
        print(f"Error during deletion of session {session_id}: {e}")
        conn.rollback()
        return False