import os
import streamlit as st
from google.cloud import storage
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
import datetime # Import datetime module
import pytz     # Recommended for timezone-aware datetimes (pip install pytz)
# --- Configuration ---
# IMPORTANT: Replace with the name of your GCS bucket

GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "")  # Replace with your GCS bucket name

# Configure how long signed URLs are valid (e.g., 5 minutes for demonstration)
# For production, adjust based on security needs.
SIGNED_URL_DURATION_SECONDS = 3000  # 50 minutes

# --- Function to Upload File to GCS ---
def upload_file_to_gcs(uploaded_file, username):
    """
    Uploads a file-like object to a Google Cloud Storage bucket
    and returns its GCS URI and a temporary signed URL for display.
    """
    try:
        # Initialize a client (automatically uses Application Default Credentials)
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET_NAME)

        # Create a blob (object) name. Using the original filename for simplicity.
        # For real-world applications, consider adding a unique prefix (e.g., timestamp, user ID)
        # to prevent overwrites or organize files better.
        # upload to a subdirectory if needed, e.g., "uploads/"
        # For example, to upload to a subdirectory:
        # Pass the datetime object
        blob = bucket.blob(f"{username}/{uploaded_file.name}")
        # blob = bucket.blob(uploaded_file.name)

        # Upload the file's content
        # Streamlit's uploaded_file object behaves like a file-like object,
        # so blob.upload_from_file() can read directly from it.
        blob.upload_from_file(uploaded_file)

        # Construct the Google Cloud Storage URI
        gcs_uri = f"gs://{GCS_BUCKET_NAME}/{uploaded_file.name}"

        # Generate a temporary, time-limited signed URL for viewing the object.
        # This is recommended for private buckets, allowing temporary access.
        # The service account or user credentials must have 'Storage Object Viewer' role.
        expiration_time = datetime.datetime.now(pytz.utc) + datetime.timedelta(seconds=SIGNED_URL_DURATION_SECONDS)
        file_display_url = blob.generate_signed_url(expiration=expiration_time,
                                                    method='GET')
    # file_display_url = blob.generate_signed_url(expiration=SIGNED_URL_DURATION_SECONDS, method='GET')

        st.success(f"✅ Successfully uploaded `{uploaded_file.name}`.")
        return gcs_uri, file_display_url
    except Exception as e:
        st.error(f"❌ Error uploading `{uploaded_file.name}`: {e}")
        return None, None