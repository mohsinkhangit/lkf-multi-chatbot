import os
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
def generate_response(model_id: str, history_messages: list, grounding_source: bool = False) -> str:
    """
    Generates a response from a Gemini model using LangChain message objects.

    Args:
        model_id: The ID of the Gemini model to use.
        history_messages: A list of LangChain BaseMessage objects (HumanMessage, AIMessage).
        grounding_source: Whether to enable Google Search grounding.

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

    result = ""
    try:
        # The API call logic remains the same
        for chunk in client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=generate_content_config,
        ):
            # Check for chunk validity
            if chunk.candidates and chunk.candidates[0].content.parts:
                result += chunk.text
    except Exception as e:
        print(f"An error occurred during content generation: {e}")
        return ""  # Return empty string on error

    # --- BETTER DESIGN: Return only the result string ---
    # The calling function will be responsible for adding this to the history.
    return result

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
