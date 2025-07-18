import os
import logging
import base64
from google import genai
from google.genai import types
import google.cloud.logging
from typing import Iterator  # Import Iterator
import sys  # To configure logging for console output

# Instantiates a client for Cloud Logging
cloud_logging_client = google.cloud.logging.Client()
# Connects the Cloud Logging handler to the root logger
cloud_logging_client.setup_logging()

# Configure basic logging for local debugging as well, if Cloud Logging isn't active
# or for immediate console output. This is good for development.
logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format='%(asctime)s - %(levelname)s - %(message)s')

from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

# --- Model and Configuration Definitions ---
GEMINI_TOKEN_LIMIT = {
    "gemini-2.0-flash-lite-001": 8192,
    "gemini-2.0-flash-001": 8192,
    "gemini-2.5-pro": 65535,
    "gemini-2.5-flash": 65535
}

SAFETY_SETTINGS = [
    types.SafetySetting(
        category="HARM_CATEGORY_HATE_SPEECH",
        threshold="OFF"
    ), types.SafetySetting(
        category="HARM_CATEGORY_DANGEROUS_CONTENT",
        threshold="OFF"
    ), types.SafetySetting(
        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
        threshold="OFF"
    ), types.SafetySetting(
        category="HARM_CATEGORY_HARASSMENT",
        threshold="OFF"
    )]

# Initialize Gemini Client
client = None
try:
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("GOOGLE_CLOUD_REGION")
    if not project_id or not location:
        logging.error("GOOGLE_CLOUD_PROJECT or GOOGLE_CLOUD_REGION environment variables are not set.")
        raise ValueError("Missing GOOGLE_CLOUD_PROJECT or GOOGLE_CLOUD_REGION.")

    client = genai.Client(
        vertexai=True,
        project=project_id,
        location=location,
    )
except Exception as e:
    logging.critical(f"Failed to initialize Gemini client: {e}")
    # In a production app, you might want a more elaborate error handling or fallback
    # but for this context, the error will be caught during function calls.
# Buffer to ensure some space for response

def generate_response(model_id: str, history_messages: list, grounding_source: bool = False) -> Iterator[str]:
    """
    Generates a response from a Gemini model using LangChain message objects.
    Handles multimodal content (text and file metadata) within history_messages.

    Args:
        model_id: The ID of the Gemini model to use.
        history_messages: A list of LangChain BaseMessage objects (HumanMessage, AIMessage).
                          HumanMessage content can now be a string or a list of (string | dict).
                          Dicts are expected to be file metadata (e.g., {'gcs_uri': '...', 'mime_type': '...'}).
        grounding_source: Whether to enable Google Search grounding.
    Returns:
        An iterator of strings, where each string is a chunk of the generated response.
    """
    if client is None:
        yield "Error: Gemini client not initialized. Check environment variables (GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_REGION)."
        return

    contents = []
    for message in history_messages:
        parts_for_gemini_content = []
        role = ""

        if isinstance(message, HumanMessage):
            role = "user"
            # Handle multimodal content in HumanMessage.content
            if isinstance(message.content, list):
                for part in message.content:
                    if isinstance(part, str):
                        parts_for_gemini_content.append(genai.types.Part.from_text(text=part))
                    elif isinstance(part, dict) and 'gcs_uri' in part and 'mime_type' in part:
                        try:
                            # Use gemini.types.Part.from_uri for GCS files
                            parts_for_gemini_content.append(
                                genai.types.Part.from_uri(file_uri=part['gcs_uri'], mime_type=part['mime_type'])
                            )
                        except Exception as e:
                            logging.warning(f"Failed to create Gemini Part from URI {part.get('gcs_uri')}: {e}")
                            # Fallback: add a text placeholder if file part fails to load.
                            parts_for_gemini_content.append(genai.types.Part.from_text(
                                text=f"[Failed to load file: {part.get('name', 'unknown file')}]"
                            ))
                    else:
                        logging.warning(f"Skipping unrecognized part in HumanMessage content: {type(part)} - {part}")
                        parts_for_gemini_content.append(genai.types.Part.from_text(
                            text=f"[Unrecognized content type: {type(part)}]"
                        ))
            else:  # Simple string content for HumanMessage
                parts_for_gemini_content.append(genai.types.Part.from_text(text=message.content))

        elif isinstance(message, AIMessage):
            role = "model"
            # AIMessage content is assumed to be a string. If it becomes multimodal, this needs adjustment.
            parts_for_gemini_content.append(genai.types.Part.from_text(text=message.content))
        else:
            logging.warning(f"Skipping unsupported LangChain message type: {type(message)}")
            continue  # Skip unsupported message types

        if parts_for_gemini_content:  # Only add content object if there are parts to include
            contents.append(
                genai.types.Content(
                    role=role,
                    parts=parts_for_gemini_content
                )
            )

    logging.debug(f"Constructed Gemini 'contents' for API call: {contents}")

    tools = [genai.types.Tool(google_search=genai.types.GoogleSearch())] if grounding_source else None

    # Calculate max_output_tokens
    # This is a simplified approach to token budgeting. For more accurate
    # budgeting, explicit token counting for inputs is recommended.
    model_max_output_tokens = GEMINI_TOKEN_LIMIT.get(model_id, 32768)

    generate_content_config = genai.types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        max_output_tokens=model_max_output_tokens,
        safety_settings=SAFETY_SETTINGS,
        tools=tools
    )

    try:
        response_stream = client.models.generate_content_stream(
            model=model_id,
            contents=contents,
            config=generate_content_config
        )
        for chunk in response_stream:
            if not chunk.candidates or not chunk.candidates[0].content.parts:
                logging.debug(f"Received empty or malformed chunk: {chunk}")
                continue

            # Yield text parts directly
            if hasattr(chunk, 'text') and chunk.text:
                yield chunk.text

            # Iterate over individual parts within the chunk's content for special types
            for part in chunk.candidates[0].content.parts:
                # FIX: Replaced part.is_set(...) with hasattr and direct attribute check
                if hasattr(part,
                           'grounding_metadata') and part.grounding_metadata and part.grounding_metadata.web_search_queries:
                    search_queries = ", ".join(part.grounding_metadata.web_search_queries)
                    yield f"\n\n*Information based on Google Search for: {search_queries}*"
                # For tool_code (function calls), access directly
                if hasattr(part, 'tool_code') and part.tool_code and hasattr(part.tool_code, 'code'):
                    yield f"\n\n*Tool code executed: \n```python\n{part.tool_code.code}\n```*"
                # For tool_response (function responses), access directly
                if hasattr(part, 'tool_response') and part.tool_response:
                    # Depending on how tool_response is structured, you might want to format it.
                    # For now, just indicate that a tool response was received.
                    yield f"\n\n*Tool response received. Details: {part.tool_response}*"

    except Exception as e:
        logging.error(f"An error occurred during Gemini content generation: {e}")
        yield f"Error during generation: {e}"


def generate_topic_from_text(model_id: str, text: str) -> str:
    """Generates a concise topic/summary from a given text using Gemini."""
    if client is None:
        logging.error("Gemini client not initialized for topic generation.")
        return "Untitled Conversation"

    try:
        logging.info(f"Generating topic for text: {text[:50]}...")
        # Use a reasonable max_output_tokens for a concise topic title
        generate_content_config = types.GenerateContentConfig(
            temperature=1,  # Slightly lower temperature for more direct summary
            top_p=0.95,
            max_output_tokens=50,  # Aim for shorter topic titles
            safety_settings=SAFETY_SETTINGS
        )
        result = ""
        prompt = f"Summarize the following text into a very concise topic title. Do not include any introductory phrases like 'The topic is:' or 'Summary'. Just the title.\n\nText: {text}"
        contents = [
            types.Content(
                role='user',
                parts=[types.Part.from_text(text=prompt)]
            )
        ]

        # Use the model_id passed; a 'flash' model is usually suitable for this task
        for chunk in client.models.generate_content_stream(
                model=model_id,
                contents=contents,
                config=generate_content_config,
        ):
            if not chunk.candidates or not chunk.candidates[0].content.parts:
                continue

            # Ensure we only append text parts for the topic
            if hasattr(chunk, 'text') and chunk.text:
                result += chunk.text

        if result:
            final_topic = result.strip()
            # Basic cleanup: remove quotes if the model wrapped output in them
            if final_topic.startswith('"') and final_topic.endswith('"'):
                final_topic = final_topic[1:-1]
            return final_topic

        logging.warning("Topic generation resulted in an empty string.")
        return "Untitled Conversation"
    except Exception as e:
        logging.error(f"Error generating topic: {e}")
        return "Untitled Conversation"