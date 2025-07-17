# gc_store_manager.py

import os
import streamlit as st
from google.cloud import storage
import datetime
import pytz

# --- Configuration ---
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME")
if not GCS_BUCKET_NAME:
    st.error("FATAL ERROR: GCS_BUCKET_NAME environment variable is not set.")
    st.stop()

SIGNED_URL_DURATION_SECONDS = 3000  # 50 minutes

def get_service_account_email():
    """
    Retrieves the service account email from the GCP metadata server.
    This is the standard way to get it when running on Cloud Run/Functions/GCE.
    """
    import requests
    try:
        url = "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/email"
        headers = {"Metadata-Flavor": "Google"}
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.text
    except Exception:
        # This will fail when running locally without the metadata server.
        # It's intended to work inside the Cloud Run environment.
        return None


# Attempt to get the service account email once
SERVICE_ACCOUNT_EMAIL = get_service_account_email()


def upload_file_to_gcs(uploaded_file, username):
    """
    Uploads a file to GCS and returns its URI and a signed URL.
    This version is optimized for Cloud Run environments.
    """
    if not SERVICE_ACCOUNT_EMAIL:
        st.error("Could not determine service account email. Cannot generate signed URL.")
        print("ERROR: Service account email could not be retrieved from metadata server.")
        return None, None

    try:
        # 1. Initialize client. It will use Application Default Credentials automatically.
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET_NAME)

        blob_name = f"{username}/{uploaded_file.name}"
        blob = bucket.blob(blob_name)

        # 2. Upload the file
        uploaded_file.seek(0)
        blob.upload_from_file(uploaded_file, content_type=uploaded_file.type)

        gcs_uri = f"gs://{GCS_BUCKET_NAME}/{blob_name}"

        # 3. Generate the signed URL using the IAM API
        # The library now handles this correctly when the service account email is provided.
        expiration_time = datetime.datetime.now(pytz.utc) + datetime.timedelta(seconds=SIGNED_URL_DURATION_SECONDS)

        display_url = blob.generate_signed_url(
            version="v4",
            expiration=expiration_time,
            method='GET',
            service_account_email=SERVICE_ACCOUNT_EMAIL  # This is the key
        )

        st.success(f"✅ Successfully uploaded `{uploaded_file.name}`.")
        return gcs_uri, display_url

    except Exception as e:
        st.error(f"❌ Error during GCS operation: {e}")
        print(f"Error uploading file {uploaded_file.name}: {e}")
        return None, None