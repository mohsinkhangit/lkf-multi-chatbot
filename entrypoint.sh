#!/bin/sh

# Exit immediately if a command exits with a non-zero status.
set -e

# Define paths
SECRET_MOUNT_PATH="/etc/secrets/secrets.toml"
STREAMLIT_CONFIG_PATH="/app/.streamlit/secrets.toml"

echo "Running an improved entrypoint script..."

# Check if the secret file exists at the mount path.
if [ -f "$SECRET_MOUNT_PATH" ]; then
    echo "Secret found at ${SECRET_MOUNT_PATH}. Preparing to copy."

    # Create the target directory. The -p flag is important.
    mkdir -p /app/.streamlit

    # --- THE FIX ---
    # Use 'cat' and I/O redirection instead of 'cp'.
    # This reads the entire source file in one go and writes it to the destination,
    # which is more resilient to the source file being replaced in the background.
    cat "${SECRET_MOUNT_PATH}" > "${STREAMLIT_CONFIG_PATH}"

    echo "Secret successfully provisioned to ${STREAMLIT_CONFIG_PATH}"
else
    echo "Warning: Secret file not found at ${SECRET_MOUNT_PATH}. Streamlit may not have auth features."
fi

echo "Setup complete. Executing application..."

# Execute the main command passed to the container (e.g., streamlit run...)
exec "$@"