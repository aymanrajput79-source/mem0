import logging
from typing import List

from app.utils.prompts import MEMORY_CATEGORIZATION_PROMPT
from dotenv import load_dotenv
from mem0.utils.factory import LlmFactory
from mem0.configs.llms.base import BaseLlmConfig
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

# Initialize Gemini LLM instead of OpenAI
config = BaseLlmConfig(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=2000
)
llm = LlmFactory.create(provider_name="gemini", config=config)

class MemoryCategories(BaseModel):
    categories: List[str]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15))
def get_categories_for_memory(memory: str) -> List[str]:
    try:
        messages = [
            {"role": "system", "content": MEMORY_CATEGORIZATION_PROMPT},
            {"role": "user", "content": memory}
        ]

        # Use Gemini LLM instead of OpenAI
        response = llm.generate_response(messages)
        
        # Parse the response to extract categories
        # This is a simplified approach - you might need to adjust based on the actual response format
        import json
        try:
            # Try to parse as JSON first
            parsed_response = json.loads(response)
            if isinstance(parsed_response, dict) and "categories" in parsed_response:
                categories = parsed_response["categories"]
            else:
                categories = parsed_response if isinstance(parsed_response, list) else [str(parsed_response)]
        except json.JSONDecodeError:
            # If not JSON, try to extract categories from text
            # This is a simple heuristic - you might need to improve this based on actual responses
            categories = [cat.strip().lower() for cat in response.split(",") if cat.strip()]
        
        return [cat.strip().lower() for cat in categories]

    except Exception as e:
        logging.error(f"[ERROR] Failed to get categories: {e}")
        raise