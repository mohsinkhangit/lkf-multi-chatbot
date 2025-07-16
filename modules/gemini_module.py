import os
import base64
from google import genai
from google.genai import types

from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

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

client = genai.Client(
    vertexai=True,
    project=os.environ["GOOGLE_CLOUD_PROJECT"],
    location=os.environ["GOOGLE_CLOUD_REGION"],
)

MAX_CONTEXT_TOKENS_BUFFER = 2000 # Buffer to ensure some space for response

def generate_response(model_id: str, history_messages: list, grounding_source: bool = False, processed_files: list = []) -> str:
    """
    Generates a response from a Gemini model using LangChain message objects.

    Args:
        model_id: The ID of the Gemini model to use.
        history_messages: A list of LangChain BaseMessage objects (HumanMessage, AIMessage).
        grounding_source: Whether to enable Google Search grounding.
        processed_files: A list of file contents to include in the response
    Returns:
        A string containing the generated response from the model.
    """
    print(history_messages)
    model = model_id
    # --- THE MAIN CHANGE: Convert LangChain objects to Gemini's format ---
    contents = []
    for message in history_messages:
        # Determine the role based on the message object's type
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "model"
        else:
            # Skip unsupported message types to avoid errors
            continue
        contents.append(
            genai.types.Content(
                role=role,
                parts=[genai.types.Part.from_text(text=message.content)],
            )
        )
    document_content = []
    if processed_files:
        for file in processed_files:
            document_content.append(
                genai.types.Part.from_uri(
                    file_uri=file['gcs_uri'],
                    mime_type=file['mime_type']
                )
            )

        # for file in processed_files:
        #     # Assuming file is a dictionary with 'name' and 'content' keys
        #     if file['file_type'] == 'image_url':
        #         try:
        #             document_content.append(types.Part.from_bytes(
        #                 data=base64.b64decode(file['content']),
        #                 mime_type=file['mime_type'],
        #             ))
        #         except Exception as e:
        #             print(f"Error decoding base64 image: {e} for file {file.get('content', 'unknown')}")
        #             document_content.append(
        #                 genai.types.Part.from_text(f"[Error processing image: {file.get('source_file')}]"))
        #
        #     elif file['file_type'] == 'document_text':
        #         document_content.append(genai.types.Part.from_text(
        #             text=f"Content from {file.get('source_file', 'document')}: \n" + file["content"]))

        if document_content:
            # Add the document content as a separate user message
            contents.append(
                genai.types.Content(
                    role="user",
                    parts=document_content
                )
            )
    # Tool configuration remains the same
    tools = [genai.types.Tool(google_search=genai.types.GoogleSearch())] if grounding_source else None

    generate_content_config = genai.types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        max_output_tokens=GEMINI_TOKEN_LIMIT[model],  # Adjusted to a more standard max
        # A response_modalities parameter may not be valid in all library versions
        safety_settings=SAFETY_SETTINGS,
        tools=tools
    )
    result_string = ""
    try:
        response = client.models.generate_content(
            model=model,
            contents=contents,  # Use the trimmed/prepared contents
            config=generate_content_config
        )
        # Stream response and accumulate text
        result_string = response.text
        # After streaming completes, get token usage from the response_stream object itself
        # This access to .usage_metadata is per the Gemini API docs for stream responses.
        if hasattr(response, 'usage_metadata'):
            # prompt_token_count is the input tokens actually used for the generation
            # candidates_token_count is the output tokens generated
            output_tokens_generated = response.usage_metadata.candidates_token_count
            # Note: response_stream.usage_metadata.prompt_token_count should match final_input_tokens here.
            # We'll use our calculated final_input_tokens for consistency with trimming logic.

    except Exception as e:
        print(f"An error occurred during Gemini content generation: {e}")
        return f"Error during generation: {e}" # Return initial input tokens in case of early error

    return result_string

def generate_topic_from_text(model, text):
    """Generates a concise topic/summary from a given text using Gemini."""
    try:
        print(f"Generating topic for text: {text[:50]}...")  # Log the start of topic generation
        generate_content_config = types.GenerateContentConfig(
            temperature=1,
            top_p=0.95,
            max_output_tokens=200,
            response_modalities=["TEXT"],
            safety_settings=SAFETY_SETTINGS
        )# Use a fast model for topic
        result = ""
        prompt = f"Summarize the following text into a very concise topic title. Prioritize clarity over length. Do not include any introductory phrases like 'The topic is:' or 'Summary:'. Just the title.\n\nText: {text}"
        contents = [
            types.Content(
                role='user',
                parts=[types.Part.from_text(text=prompt)]
            )
        ]

        for chunk in client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=generate_content_config,
        ):
            if not chunk.candidates or not chunk.candidates[0].content.parts:
                continue
            result += chunk.text
            print(f"Chunk received: {chunk.text}")
        if result:
            return result.strip()
        return "Untitled Conversation"
    except Exception as e:
        print(f"Error generating topic: {e}")
        return "Untitled Conversation" # Fallback topic
