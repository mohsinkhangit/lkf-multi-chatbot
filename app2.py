import streamlit as st
from google.cloud import storage
from modules.gc_storage_manager import upload_file_to_gcs
import os
import mimetypes  # Handy for robust MIME type checking, though st.file_uploader usually provides it

# --- Configuration ---
# IMPORTANT: Replace with the name of your GCS bucket


# Configure how long signed URLs are valid (e.g., 5 minutes for demonstration)
# For production, adjust based on security needs.
SIGNED_URL_DURATION_SECONDS = 300  # 5 minutes


# --- Streamlit UI ---
st.set_page_config(page_title="GCS Multi-File Uploader with Thumbnails", layout="wide")

st.title("🚀 Multi-File Uploader to Google Cloud Storage")

st.info(
    f"This application uploads files to your GCS bucket: **`{GCS_BUCKET_NAME}`**. "
    "For displaying file previews, temporary **signed URLs** are generated, "
    f"valid for {SIGNED_URL_DURATION_SECONDS / 60:.0f} minutes.", icon="ℹ️"
)
st.warning(
    "**Important Permissions:** Your service account (Cloud Run) or local credentials "
    "**must have `Storage Object Creator` AND `Storage Object Viewer` roles** on the "
    "GCS bucket for both uploading and generating signed URLs. "
    "For local testing, ensure 'GOOGLE_APPLICATION_CREDENTIALS' is set correctly.",
    icon="⚠️"
)

# File uploader widget, allowing multiple files
uploaded_files = st.file_uploader(
    "Choose files to upload (PDF, TXT, common image formats)",
    type=["pdf", "txt", "png", "jpg", "jpeg", "gif", "webp"],  # Specify accepted file types
    accept_multiple_files=True
)

if uploaded_files:
    st.subheader("Selected Files Overview:")
    for file in uploaded_files:
        st.write(f"- {file.name} (Type: {file.type if file.type else 'unknown'}, Size: {file.size} bytes)")

    if st.button("Start Upload to GCS", type="primary", use_container_width=True):
        st.subheader("Upload Results & Previews:")
        all_uploads_successful = True
        uploaded_file_details = []

        for file in uploaded_files:
            gcs_uri, display_url = upload_file_to_gcs(file, )
            if gcs_uri and display_url:
                uploaded_file_details.append({
                    "name": file.name,
                    "type": file.type,  # Streamlit provides MIME type here
                    "gcs_uri": gcs_uri,
                    "display_url": display_url
                })
            else:
                all_uploads_successful = False

        if uploaded_file_details:
            st.markdown("---")
            # Display uploaded files in a structured way
            for detail in uploaded_file_details:
                col1, col2 = st.columns([1, 2])  # Adjust column ratio for layout
                with col1:
                    st.write(f"**File Name:** `{detail['name']}`")
                    st.write(f"**GCS URI:** `{detail['gcs_uri']}`")
                    st.markdown(
                        f"**View/Download:** [Click Here]({detail['display_url']}) (valid for {SIGNED_URL_DURATION_SECONDS / 60:.0f} min)")
                with col2:
                    # Check if it's an image based on MIME type provided by Streamlit
                    if detail['type'] and detail['type'].startswith('image/'):
                        st.image(detail['display_url'], caption=f"Thumbnail for {detail['name']}", width=250)
                    else:
                        st.info("No visual preview available for this file type. Click the link to view/download.")
                st.markdown("---")  # Separator for each file

        if all_uploads_successful:
            st.balloons()
            st.success("All selected files have been successfully processed! 🎉")
        else:
            st.error("Some files failed to upload. Check the error messages above.")

else:
    st.info("Upload some files to get started!", icon="⬆️")

st.markdown("---")
st.caption("Developed with Streamlit and Google Cloud Storage")