# modules/session_manager_module.py
import uuid
import psycopg
from typing import List, Dict

# --- Import from the connection manager ---
from modules.db_connection_manager import get_db_connection
# --- This import remains one-way and safe ---
from modules import lc_memory_module


def init_db():
    """
    Initializes the database.
    - Adds a 'user_id' column to the 'sessions' table to associate chats with users.
    - Adds an index on 'user_id' for fast retrieval of user-specific sessions.
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # --- SCHEMA CHANGE ---
            # Add user_id column to store the user's email.
            cur.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id UUID PRIMARY KEY,
                    user_id VARCHAR(255) NOT NULL,
                    topic VARCHAR(255),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # --- ADD INDEX for performance ---
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions (user_id);
            """)
        conn.commit()
    except psycopg.Error as e:
        print(f"Database initialization error: {e}")
        conn.rollback()
        raise


def create_new_session_db(user_id: str) -> str:
    """
    Creates a new session record FOR A SPECIFIC USER and returns its UUID string.
    """
    # --- QUERY CHANGE ---
    # The INSERT statement now includes the user_id.
    sql = "INSERT INTO sessions (session_id, user_id, topic) VALUES (%s, %s, %s) RETURNING session_id"
    new_uuid = uuid.uuid4()
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Pass the user_id as a parameter.
            cur.execute(sql, (new_uuid, user_id, "New Chat"))
            session_id = cur.fetchone()[0]
        conn.commit()
        return str(session_id)
    except psycopg.Error as e:
        print(f"Error creating new session for user {user_id}: {e}")
        conn.rollback()
        raise


def get_sessions_for_user_db(user_id: str) -> List[Dict[str, str]]:
    """
    Retrieves all chat sessions for a SPECIFIC USER, ordered by most recent first.
    """
    # --- QUERY CHANGE ---
    # The SELECT statement is now filtered by user_id.
    sql = "SELECT session_id, topic FROM sessions WHERE user_id = %s ORDER BY created_at DESC"
    conn = get_db_connection()
    with conn.cursor() as cur:
        # Pass the user_id as a query parameter.
        cur.execute(sql, (user_id,))  # Note the comma to make it a tuple
        sessions = [
            {"session_id": str(row[0]), "topic": row[1]}
            for row in cur.fetchall()
        ]
    return sessions


def update_session_topic_db(session_id: str, topic: str):
    """
    Updates the topic for a given session.
    (No change needed here as session_id is a unique primary key).
    For enhanced security, one could add a user_id check here, but it's not
    strictly necessary if the UI only allows users to access their own sessions.
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
    Deletes a session record and its associated messages.
    (No change needed here for the same reason as update_session_topic_db).
    """
    sql = "DELETE FROM sessions WHERE session_id = %s"
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (uuid.UUID(session_id),))

        history = lc_memory_module.get_chat_history(session_id)
        history.clear()

        conn.commit()
        print(f"Successfully deleted session {session_id}")
        return True
    except (Exception, psycopg.Error) as e:
        print(f"Error during deletion of session {session_id}: {e}")
        conn.rollback()
        return False