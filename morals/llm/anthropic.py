# morals/llm/anthropic.py
import os
from typing import Dict, Any, Optional
import anthropic  # You'll need to pip install anthropic

from .base import LLMInterface


class AnthropicInterface(LLMInterface):
    """Interface for Anthropic's Claude models."""
    
    def __init__(self, 
                 model_name: str = "claude-3-haiku-20240307", 
                 api_key: Optional[str] = None,
                 max_tokens: int = 1000):
        super().__init__(model_name, api_key)
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key must be provided or set as ANTHROPIC_API_KEY environment variable")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.max_tokens = max_tokens
    
    async def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate a response from Claude."""
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        temperature = kwargs.get("temperature", 0.0)  # Default to deterministic
        
        # Create the message
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Return just the text content
        return response.content[0].text