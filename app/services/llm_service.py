from groq import Groq
import os


# Load API key from environment
api_key = os.getenv("GROQ_API_KEY")
# Optional safety check
if not api_key:
    raise ValueError("GROQ_API_KEY is not set in environment variables")
client = Groq(api_key=api_key)

def get_llm_response(question: str) -> str:
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": question}],
            model="llama-3.3-70b-versatile",
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"