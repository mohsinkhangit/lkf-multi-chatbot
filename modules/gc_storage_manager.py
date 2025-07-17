# gc_store_manager.py

import os
import streamlit as st
from google.cloud import storage
from dotenv import load_dotenv
import datetime
import pytz
import google.auth  # Import the google.auth library

load_dotenv()

GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "")
SIGNED_URL_DURATION_SECONDS = 3000  # 50 minutes


def upload_file_to_gcs(uploaded_file, username):
    """
    Uploads a file-like object to a Google Cloud Storage bucket
    and returns its GCS URI and a temporary signed URL for display.
    """
    try:
        # Get the default credentials and the service account email
        # This will automatically work on Cloud Run and other GCP environments
        credentials, project_id = google.auth.default()

        # Initialize the client with the discovered credentials.
        # This makes it explicit but is often not required if GOOGLE_APPLICATION_CREDENTIALS is not set.
        client = storage.Client(credentials=credentials)
        bucket = client.bucket(GCS_BUCKET_NAME)

        blob_name = f"{username}/{uploaded_file.name}"
        blob = bucket.blob(blob_name)

        # Rewind the file stream before uploading
        uploaded_file.seek(0)

        # Upload the file's content from the stream
        blob.upload_from_file(uploaded_file, content_type=uploaded_file.type)

        # Construct the Google Cloud Storage URI
        gcs_uri = f"gs://{GCS_BUCKET_NAME}/{blob_name}"

        # --- THIS IS THE CRITICAL FIX ---
        # Generate a signed URL using the IAM API by providing the service account email.
        # This requires the service account to have the 'Service Account Token Creator' role.
        expiration_time = datetime.datetime.now(pytz.utc) + datetime.timedelta(seconds=SIGNED_URL_DURATION_SECONDS)

        file_display_url = blob.generate_signed_url(
            version="v4",
            expiration=expiration_time,
            method='GET',
            service_account_email=credentials.service_account_email,  # Use the discovered service account
            access_token=credentials.token  # Pass the token for authentication
        )

        st.success(f"✅ Successfully uploaded `{uploaded_file.name}`.")
        return gcs_uri, file_display_url

    except Exception as e:
        st.error(f"❌ Error uploading `{uploaded_file.name}`: {e}")
        print(f"Error uploading file {uploaded_file.name}: {e}")
        # Make sure to return a tuple of Nones to prevent the TypeError
        return None, None