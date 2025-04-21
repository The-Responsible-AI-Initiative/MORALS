# morals/evaluation/response_processor.py
import re
from typing import Dict, Any, Tuple, Optional


class MFQResponseProcessor:
    """Processes LLM responses to MFQ questions."""
    
    def __init__(self):
        # Regular expression pattern to extract score
        self.score_pattern = r'Score\s*\(0-5\):\s*(\d)'
        
        # Pattern to extract reasoning (everything after "Reasoning:" label)
        self.reasoning_pattern = r'(?:Reasoning|reasoning):\s*(.*?)(?:\n\n|\Z)'
    
    def process_response(self, response_text: str) -> Tuple[Optional[int], Optional[str]]:
        """
        Process an LLM response to extract score and reasoning.
        
        Args:
            response_text: The raw text response from an LLM
            
        Returns:
            A tuple of (score, reasoning)
        """
        # Extract score
        score_match = re.search(self.score_pattern, response_text, re.DOTALL)
        score = int(score_match.group(1)) if score_match else None
        
        # Extract reasoning
        reasoning_match = re.search(self.reasoning_pattern, response_text, re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else None
        
        return score, reasoning
    
    def validate_response(self, score: Optional[int], reasoning: Optional[str]) -> bool:
        """
        Validate that a processed response is complete.
        
        Args:
            score: The extracted score
            reasoning: The extracted reasoning
            
        Returns:
            True if the response is valid, False otherwise
        """
        if score is None:
            return False
        
        if score < 0 or score > 5:
            return False
        
        if reasoning is None or len(reasoning.strip()) == 0:
            return False
        
        return True