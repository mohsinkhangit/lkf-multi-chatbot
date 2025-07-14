# Start from a specific, lean Python base image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the dependency file first to leverage Docker's layer caching
COPY requirements.txt ./

# Install the Python dependencies
# Using --no-cache-dir reduces the image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . ./

# --- NEW: Copy the entrypoint script and make it executable ---
# This script will run first when the container starts.
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Expose the port that Streamlit will run on
EXPOSE 8501

# --- NEW: Set the entrypoint for the container ---
# When the container starts, it will execute this script.
ENTRYPOINT ["/app/entrypoint.sh"]

# --- MODIFIED: Set the default command that will be passed to the entrypoint ---
# The entrypoint script will execute this command after it finishes its setup.
# Using 8080 for Cloud Run's default port.
# server.headless=true is a recommended setting for running in the cloud.
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableCORS=false"]