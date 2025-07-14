#!/bin/sh

# Exit immediately if a command exits with a non-zero status.
set -e

# Define the source path of the mounted secret and the target path Streamlit expects.
SECRET_MOUNT_PATH="/etc/secrets/secrets.toml"
STREAMLIT_CONFIG_PATH="/app/.streamlit/secrets.toml"

echo "Running entrypoint script..."
echo "Checking if secret exists at ${SECRET_MOUNT_PATH}"

# Check if the secret file exists at the mount path.
if [ -f "$SECRET_MOUNT_PATH" ]; then
    echo "Secret found. Creating .streamlit directory and copying the secret."

    # Create the .streamlit directory in the app's working directory.
    # The -p flag ensures it doesn't fail if the directory already exists.
    mkdir -p /app/.streamlit

    # Copy the secret from the mount path to the location Streamlit requires.
    cp "${SECRET_MOUNT_PATH}" "${STREAMLIT_CONFIG_PATH}"

    echo "Secret copied to ${STREAMLIT_CONFIG_PATH}"
else
    echo "Warning: Secret file not found at ${SECRET_MOUNT_PATH}. Streamlit will run without it."
    # This branch is useful for local development where the secret isn't mounted.
fi

# This is the crucial part:
# It takes all the arguments that were passed to the entrypoint (e.g., "streamlit run app.py")
# and executes them. This is what actually starts your Streamlit app.
exec "$@"