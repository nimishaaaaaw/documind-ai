import os
from openai import OpenAI, APITimeoutError, APIConnectionError, APIError
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in environment variables.")

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

MODEL_NAME = "llama-3.3-70b-versatile"


def generate_answer(prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert document analysis assistant. "
                        "You answer questions strictly based on provided document context. "
                        "You are precise, concise, and always cite your reasoning. "
                        "When asked to decompose queries, return clean numbered lists only."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content

    except APITimeoutError:
        return "The AI service timed out. Please try again."
    except APIConnectionError:
        return "Unable to connect to the AI service."
    except APIError:
        return "An error occurred while generating the response."
    except Exception:
        return "Unexpected error occurred while processing your request."