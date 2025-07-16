# modules/db_connection_manager.py
import os
import psycopg
import streamlit as st
from psycopg import OperationalError

# This is the single source of truth for the database connection string.
CONNECTION_STRING = os.environ.get("POSTGRES_CONNECTION_STRING")
if not CONNECTION_STRING:
    raise ValueError("POSTGRES_CONNECTION_STRING environment variable not set.")

@st.cache_resource
def init_db_connection():
    """
    Initializes and caches a single database connection for the entire app session.
    This function is decorated with st.cache_resource to ensure it runs only once.
    """
    try:
        print("--- Initializing Database Connection (runs once) ---")

        # Adapt the SQLAlchemy-style DSN for direct use with psycopg
        psycopg_conn_string = CONNECTION_STRING.replace("+psycopg2", "")

        # Establish the connection
        conn = psycopg.connect(psycopg_conn_string)

        # It's good practice to set autocommit False to manage transactions manually.
        conn.autocommit = False
    except OperationalError as e:
        print(f"Error connecting to the database: {e}")
        raise

    return conn


def get_db_connection():
    """
    Retrieves the cached database connection created by init_db_connection.
    """
    return init_db_connection()