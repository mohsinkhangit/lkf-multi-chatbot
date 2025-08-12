import base64
import json
import os
from typing import Dict, List, Tuple, Sequence


from cryptography.fernet import Fernet
from google.cloud import kms_v1

from langchain_core.messages import BaseMessage, messages_from_dict, message_to_dict
from langchain_postgres.chat_message_histories import (
    PostgresChatMessageHistory,
    _get_messages_query,
    _insert_message_query,
)

from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# --- Configuration for Google Cloud KMS ---
# REPLACE THESE WITH YOUR ACTUAL KMS DETAILS
GCP_PROJECT_ID = os.environ["GOOGLE_CLOUD_PROJECT"]  # e.g., "my-gcp-project-12345"
KMS_KEY_RING_LOCATION = "asia-east2"  # e.g., "us-central1", "europe-west1"
KMS_KEY_RING_ID = "message_encryption_key"  # e.g., "my-application-keys"
KMS_CRYPTO_KEY_ID = "chat_history_key"  # e.g., "message-history-kek"

# Construct the KEK name
KMS_KEY_NAME = f"projects/{GCP_PROJECT_ID}/locations/{KMS_KEY_RING_LOCATION}/keyRings/{KMS_KEY_RING_ID}/cryptoKeys/{KMS_CRYPTO_KEY_ID}"

# --- KMS Client Initialization ---
kms_client = kms_v1.KeyManagementServiceClient()


# --- Envelope Encryption Functions ---
def _generate_dek() -> bytes:
    """Generates a random 32-byte Data Encryption Key (DEK)."""
    return os.urandom(32)


def encrypt_with_envelope(plaintext: bytes) -> Tuple[bytes, bytes]:
    """
    Encrypts plaintext using a locally-generated DEK, then encrypts the DEK
    using Google Cloud KMS (KEK).

    Returns:
        A tuple of (ciphertext, encrypted_dek).
    """
    # 1. Generate a local DEK
    dek = _generate_dek()
    fernet = Fernet(base64.urlsafe_b64encode(dek))  # Fernet requires URL-safe base64 key

    # 2. Encrypt plaintext locally with DEK
    ciphertext = fernet.encrypt(plaintext)

    # 3. Encrypt the DEK using KMS (KEK)
    try:
        encrypt_dek_response = kms_client.encrypt(
            request={
                "name": KMS_KEY_NAME,
                "plaintext": dek,
            }
        )
        encrypted_dek = encrypt_dek_response.ciphertext
        return ciphertext, encrypted_dek
    except Exception as e:
        print(f"Error encrypting DEK with KMS: {e}")
        raise

def decrypt_with_envelope(ciphertext: bytes, encrypted_dek: bytes) -> bytes:
    """
    Decrypts encrypted_dek using KMS (KEK), then decrypts ciphertext
    using the decrypted DEK.

    Returns:
        The original plaintext.
    """
    # 1. Decrypt the DEK using KMS (KEK)
    try:
        decrypt_dek_response = kms_client.decrypt(
            request={
                "name": KMS_KEY_NAME,  # Note: KMS uses the KEK to decrypt its own ciphertext (the DEK)
                "ciphertext": encrypted_dek,
            }
        )
        decrypted_dek = decrypt_dek_response.plaintext
    except Exception as e:
        print(f"Error decrypting DEK with KMS: {e}")
        raise

    # 2. Decrypt ciphertext locally with decrypted DEK
    fernet = Fernet(base64.urlsafe_b64encode(decrypted_dek))
    plaintext = fernet.decrypt(ciphertext)
    return plaintext


# --- Custom Encrypted Chat Message History ---

class EncryptedPostgresChatMessageHistory(PostgresChatMessageHistory):
    """
    Chat message history that encrypts and decrypts messages using Google Cloud KMS
    with envelope encryption.

    Messages are stored in an underlying BaseChatMessageHistory (e.g., InMemoryChatMessageHistory)
    as encrypted blobs.
    """

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """
        Encrypts messages and then adds them to the Postgres database.
        Each message is stored as a JSON blob containing its ciphertext
        and the encrypted data encryption key (DEK).
        """
        if self._connection is None:
            raise ValueError(
                "Please initialize with a sync connection or use aadd_messages."
            )

        encrypted_payloads = []
        for message in messages:
            # 1. Convert the message to JSON bytes
            message_json_bytes = json.dumps(message_to_dict(message)).encode('utf-8')

            # 2. Encrypt the bytes using the envelope encryption pattern
            ciphertext, encrypted_dek = encrypt_with_envelope(message_json_bytes)

            # 3. Prepare the payload for database storage (Base64 encode bytes)
            db_payload = {
                "ciphertext": base64.b64encode(ciphertext).decode('utf-8'),
                "encrypted_dek": base64.b64encode(encrypted_dek).decode('utf-8'),
            }

            # The value to be inserted into the DB for this message
            encrypted_payloads.append((self._session_id, json.dumps(db_payload)))

        # 4. Use the parent's internal method to get the query and insert
        query = _insert_message_query(self._table_name)
        with self._connection.cursor() as cursor:
            cursor.executemany(query, encrypted_payloads)
        self._connection.commit()

    async def aadd_messages(self, messages: Sequence[BaseMessage]) -> None:
        """
        Asynchronously encrypts and adds messages to the Postgres database.
        """
        if self._aconnection is None:
            raise ValueError(
                "Please initialize with an async connection or use add_messages."
            )

        encrypted_payloads = []
        for message in messages:
            message_json_bytes = json.dumps(message_to_dict(message)).encode('utf-8')
            ciphertext, encrypted_dek = encrypt_with_envelope(message_json_bytes)
            db_payload = {
                "ciphertext": base64.b64encode(ciphertext).decode('utf-8'),
                "encrypted_dek": base64.b64encode(encrypted_dek).decode('utf-8'),
            }
            encrypted_payloads.append((self._session_id, json.dumps(db_payload)))

        query = _insert_message_query(self._table_name)
        async with self._aconnection.cursor() as cursor:
            await cursor.executemany(query, encrypted_payloads)
        await self._aconnection.commit()

    def get_messages(self) -> List[BaseMessage]:
        """
        Retrieves encrypted messages from Postgres and decrypts them.
        """
        if self._connection is None:
            raise ValueError(
                "Please initialize with a sync connection or use aget_messages."
            )

        # 1. Fetch the raw encrypted data from the database
        query = _get_messages_query(self._table_name)
        with self._connection.cursor() as cursor:
            cursor.execute(query, {"session_id": self._session_id})
            # Each record 0  is a JSON string of our encrypted payload
            items = [record[0] for record in cursor.fetchall()]

        decrypted_dicts = []
        for item in items:
            # 2. Parse the payload and decode from Base64
            # db_payload = json.loads(item)
            ciphertext = base64.b64decode(item["ciphertext"])
            encrypted_dek = base64.b64decode(item["encrypted_dek"])

            # 3. Decrypt the message
            decrypted_json_bytes = decrypt_with_envelope(ciphertext, encrypted_dek)
            message_dict = json.loads(decrypted_json_bytes.decode('utf-8'))
            decrypted_dicts.append(message_dict)

        # 4. Convert the decrypted dictionaries back into BaseMessage objects
        return messages_from_dict(decrypted_dicts)

    async def aget_messages(self) -> List[BaseMessage]:
        """
        Asynchronously retrieves and decrypts messages from Postgres.
        """
        if self._aconnection is None:
            raise ValueError(
                "Please initialize with an async connection or use get_messages."
            )

        query = _get_messages_query(self._table_name)
        async with self._aconnection.cursor() as cursor:
            await cursor.execute(query, {"session_id": self._session_id})
            items = [record [0] for record in await cursor.fetchall()]

        decrypted_dicts = []
        for item in items:
            db_payload = json.loads(item)
            ciphertext = base64.b64decode(db_payload["ciphertext"])
            encrypted_dek = base64.b64decode(db_payload["encrypted_dek"])
            decrypted_json_bytes = decrypt_with_envelope(ciphertext, encrypted_dek)
            message_dict = json.loads(decrypted_json_bytes.decode('utf-8'))
            decrypted_dicts.append(message_dict)

        return messages_from_dict(decrypted_dicts)
