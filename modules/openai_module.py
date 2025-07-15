import os
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
    "gpt-4o": {"deployment_name":"gpt-4o", "max_tokens": 4096, "context_length": 128000},
    "gpt-4o-mini": {"deployment_name":"gpt-4o-mini", "max_tokens": 16384, "context_length": 128000},
}


subscription_key = os.environ['AZURE_OPENAI_API_KEY']
api_version = os.environ['AZURE_OPENAI_API_VERSION']

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
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
    max_tokens = model_deployment[model_id]["max_tokens"]
    context_limit = model_deployment[model_id]["context_length"]
    model_deployment_name = model_deployment[model_id]["deployment_name"]
    messages = []
    trimmed_messages = trim_messages(
        history_messages,
        # Keep the last <= n_count tokens of the messages.
        strategy="last",
        # Remember to adjust based on your model
        # or else pass a custom token_counter
        token_counter=count_tokens_approximately,
        # Most chat models expect that chat history starts with either:
        # (1) a HumanMessage or
        # (2) a SystemMessage followed by a HumanMessage
        # Remember to adjust based on the desired conversation
        # length
        max_tokens=context_limit,
        # Most chat models expect that chat history starts with either:
        # (1) a HumanMessage or
        # (2) a SystemMessage followed by a HumanMessage
        start_on="human",
        # Most chat models expect that chat history ends with either:
        # (1) a HumanMessage or
        # (2) a ToolMessage
        end_on=("human"),
        # Usually, we want to keep the SystemMessage
        # if it's present in the original history.
        # The SystemMessage has special instructions for the model.
        include_system=True,
        allow_partial=False,
    )
    for message in trimmed_messages:
        # Determine the role based on the message object's type
        if isinstance(message, HumanMessage):
            messages.append({
                "role": 'user',
                "content": message.content
            })
        elif isinstance(message, AIMessage):
            messages.append({
                "role": "assistant",
                "content": message.content
            })
        else:
            # Skip unsupported message types to avoid errors
            continue
    response = client.chat.completions.create(
        messages=messages,
        max_tokens=max_tokens,
        temperature=1.0,
        top_p=1.0,
        model=model_deployment_name
    )
    result = response.choices[0].message.content
    print(result)
    return result

def generate_topic_from_text(model_id, text):
    """Generates a concise topic/summary from a given text using Gemini."""
    try:
        model_deployment_name = model_deployment[model_id]["deployment_name"] # Log the start of topic generation
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
        return "Untitled Conversation" # Fallback topic


