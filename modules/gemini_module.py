import os

from google import genai
from google.genai import types

from dotenv import load_dotenv
load_dotenv()

def generate_response(model_id, history, grounding_source=None):
  client = genai.Client(
      vertexai=True,
      project=os.environ["GOOGLE_CLOUD_PROJECT"],
      location=os.environ["GOOGLE_CLOUD_REGION"],
  )

  model = model_id
  contents = []
  if history is None:
    history = []
  for message in history:
    contents.append(types.Content(
      role=message["role"] if message["role"] == "user" else "model",
      parts=[
        types.Part.from_text(text=message["content"])
      ]
    ))
  tools = None
  if grounding_source:
    tools = [
      types.Tool(google_search=types.GoogleSearch())
    ]
  generate_content_config = types.GenerateContentConfig(
    temperature = 1,
    top_p = 0.95,
    max_output_tokens = 8192,
    response_modalities = ["TEXT"],
    safety_settings = [types.SafetySetting(
      category="HARM_CATEGORY_HATE_SPEECH",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_DANGEROUS_CONTENT",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_HARASSMENT",
      threshold="OFF"
    )],
    tools= tools
  )
  result = ""
  for chunk in client.models.generate_content_stream(
    model = model,
    contents = contents,
    config = generate_content_config,
    ):
    if not chunk.candidates or not chunk.candidates[0].content.parts:
      continue
    result += chunk.text
  history.append(
    {
        "role": "assistant",
        "content": result
    })
  return history