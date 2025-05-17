# morals/llm/wvs_prompt_formatter.py
from typing import Dict, Any

class WVSPromptFormatter:
    """Formats World Values Survey questions into prompts for LLMs."""
    
    @staticmethod
    def format_prompt(question: Dict[str, Any]) -> str:
        """
        Format a WVS question into a prompt for an LLM.
        
        Args:
            question: The WVS question data
            
        Returns:
            A formatted prompt string
        """
        # World Values Survey questions already have a well-formatted prompt
        # We can use it directly
        prompt = question.get("prompt", "")
        
        # Add context about the domain if available
        domain_name = question.get("domain_name", "")
        
        if domain_name:
            context = f"This question is part of the {domain_name} domain in the World Values Survey.\n\n"
            prompt = context + prompt
        
        return prompt