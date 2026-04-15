import os
from typing import Optional, Type, TypeVar
from langchain_groq import ChatGroq
from langchain_core.pydantic_v1 import BaseModel
from dotenv import load_dotenv

load_dotenv()

T = TypeVar("T", bound=BaseModel)

class LLMClient:
    """Wrapper for Groq LLM to handle completions and structured output."""

    def __init__(self, model_name: str = "llama-3.1-70b-versatile", temperature: float = 0.1):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            # Fallback for demonstration if key is missing
            print("Warning: GROQ_API_KEY not found in environment.")
        
        self.llm = ChatGroq(
            model_name=model_name,
            temperature=temperature,
            groq_api_key=api_key
        )

    def get_completion(self, prompt: str, system_message: str = "You are a helpful medical assistant.") -> str:
        """Standard chat completion."""
        messages = [
            ("system", system_message),
            ("human", prompt)
        ]
        response = self.llm.invoke(messages)
        return response.content

    def get_structured_output(
        self, 
        prompt: str, 
        output_schema: Type[T], 
        system_message: str = "You are a clinical AI assistant that provides structured JSON output."
    ) -> T:
        """Completion with guaranteed schema enforcement."""
        structured_llm = self.llm.with_structured_output(output_schema)
        messages = [
            ("system", system_message),
            ("human", prompt)
        ]
        return structured_llm.invoke(messages)

# Example usage for testing (if run directly)
if __name__ == "__main__":
    class SimpleDiagnosis(BaseModel):
        condition: str
        confidence: float
        recommendation: str

    client = LLMClient()
    # test_prompt = "Patient has high cholesterol and blood pressure. What is the diagnosis?"
    # result = client.get_structured_output(test_prompt, SimpleDiagnosis)
    # print(result)
