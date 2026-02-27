from google import genai
from google.genai import types

def generate():
    # Use Vertex AI with ADC, but Gemini 3 preview requires location = "global"
    client = genai.Client(
        vertexai=True,
        project="medassistai-488422",
        location="global",          # <— key change
    )

    model = "gemini-3-flash-preview"

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text="hi")]
        ),
    ]

    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        max_output_tokens=512,
    )

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        if not chunk.candidates or not chunk.candidates[0].content or not chunk.candidates[0].content.parts:
            continue
        print(chunk.text, end="")

if __name__ == "__main__":
    generate()