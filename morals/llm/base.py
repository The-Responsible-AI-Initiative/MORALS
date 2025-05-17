# morals/llm/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class LLMInterface(ABC):
    """Base class for LLM interfaces."""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key
    
    @abstractmethod
    async def generate_response(self, prompt: str, **kwargs) -> str:
        """
        Generate a response from the LLM based on the prompt.
        
        Args:
            prompt: Input prompt for the LLM
            **kwargs: Additional model-specific parameters
            
        Returns:
            The LLM's response text
        """
        pass
    
    @property
    def model_info(self) -> Dict[str, Any]:
        """Return information about the model."""
        return {
            "name": self.model_name,
            "interface": self.__class__.__name__
        }