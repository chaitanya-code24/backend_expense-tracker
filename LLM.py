# llm.py
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

def get_llm_response(prompt: str, model="llama3-70b-8192") -> str:
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful expense assistant."},
                {"role": "user", "content": prompt}
            ],
            model=model,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"LLM Error: {str(e)}"
