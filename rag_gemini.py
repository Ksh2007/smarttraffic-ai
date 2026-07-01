import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

def call_gemini(prompt: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return "⚠️ GEMINI_API_KEY not found."

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        return f"Gemini API Error: {e}"


if __name__ == "__main__":
    test_prompt = "Hello Gemini! Reply with 'Gemini connection successful.'"
    print(call_gemini(test_prompt))