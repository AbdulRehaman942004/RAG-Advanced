from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client_groq = OpenAI(
        api_key=os.environ.get("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
)

def confidence_score(
    user_query: str,
    model_name: str = "openai/gpt-oss-20b"
) -> str:

    prompt = f"""
        You are a RAG-based query analysis assistant for an application. The app answers user queries based only on the document "Oxford Guide-2022.pdf". 

        Your task is as follows:

        1. Receive a user query.
        2. Analyze if the query is related to the content of Oxford or "Oxford Guide-2022.pdf".
        3. Assign a confidence score between 0 and 1:
        - 0 = completely unrelated to Oxford.
        - 1 = fully related to Oxford.

        6. Output ONLY the confidence score as a decimal number**. No explanations, no text, no punctuation other than the decimal number. 

        Example:
        - Input: "whats Oxford universty famous for?"
        - Output: 0.95
    """
    answer = client_groq.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_query}
        ],
    )

    confidence_score = answer.choices[0].message.content or ""
    return confidence_score


print(confidence_score("Oxford"))