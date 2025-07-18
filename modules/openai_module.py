import os
import asyncio
from openai import AzureOpenAI

# from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    trim_messages,
)
from langchain_core.messages.utils import count_tokens_approximately

endpoint = os.environ['AZURE_ENDPOINT_URL']

model_deployment = {
    "gpt-4o": {"deployment_name": "gpt-4o", "max_tokens": 4096, "context_length": 128000},
    "gpt-4o-mini": {"deployment_name": "gpt-4o-mini", "max_tokens": 16384, "context_length": 128000},
}

subscription_key = os.environ['AZURE_OPENAI_API_KEY']
api_version = os.environ['AZURE_OPENAI_API_VERSION']

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)


def generate_response(model_id: str, history_messages: list,
                      grounding_source: bool = False) -> str:
    """
    Generates a response from a Gemini model using LangChain message objects.

    Args:
        model_id: The ID of the Gemini model to use.
        history_messages: A list of LangChain BaseMessage objects (HumanMessage, AIMessage)
        processed_files: A list of file metadata to include in the response.
        grounding_source: Whether to enable Google Search grounding.

    Returns:
        A string containing the generated response from the model.
    """
    max_tokens = model_deployment[model_id]["max_tokens"]
    context_limit = model_deployment[model_id]["context_length"]
    model_deployment_name = model_deployment[model_id]["deployment_name"]
    messages = []
    trimmed_messages = trim_messages(
        history_messages,
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=context_limit,
        start_on="human",
        end_on="human",
        include_system=True,
        allow_partial=False,
    )
    for message in trimmed_messages:
        if isinstance(message, HumanMessage):
            role = "user"
            # Handle multimodal content in HumanMessage.content
            if isinstance(message.content, list):
                for part in message.content:
                    if isinstance(part, str):
                        messages.append({
                            "role": role,
                            "content": part
                        })
                    #TODO: Handle other content types like files or images
        elif isinstance(message, AIMessage):
            messages.append({
                "role": "assistant",
                "content": message.content
            })
    try:
        response_stream = client.chat.completions.create(
            messages=messages,
            max_tokens=max_tokens,
            temperature=1.0,
            top_p=1.0,
            model=model_deployment_name,
            stream=True  # Crucial for streaming
        )

        for chunk in response_stream:
            # Check if there's content in the chunk and yield it
            if chunk.choices and chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    except Exception as e:
        print(f"An error occurred during OpenAI content generation: {e}")
        yield f"Error during generation: {e}"  # Yield an error message if an exception occurs


def generate_topic_from_text(model_id, text):
    """Generates a concise topic/summary from a given text using Gemini."""
    try:
        model_deployment_name = model_deployment[model_id]["deployment_name"]  # Log the start of topic generation
        prompt = f"Summarize the following text into a very concise topic title. Prioritize clarity over length. Do not include any introductory phrases like 'The topic is:' or 'Summary:'. Just the title.\n\nText: {text}"
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that generates concise topic titles from text."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        response = client.chat.completions.create(
            messages=messages,
            max_tokens=100,
            temperature=1.0,
            top_p=1.0,
            model=model_deployment_name
        )
        result = response.choices[0].message.content
        return result
    except Exception as e:
        print(f"Error generating topic: {e}")
        return "Untitled Conversation"  # Fallback topic
