# morals/llm/dilemmas_prompt_formatter.py
from typing import Dict, Any

class DilemmasPromptFormatter:
    """Formats moral dilemma questions into prompts for LLMs."""
    
    @staticmethod
    def format_prompt(question: Dict[str, Any]) -> str:
        """
        Format a moral dilemma question into a prompt for an LLM.
        
        Args:
            question: The dilemma question data
            
        Returns:
            A formatted prompt string
        """
        dilemma_title = question.get("dilemma_title", "")
        dilemma_description = question.get("dilemma_description", "")
        question_text = question.get("text", "")
        
        prompt = f"""# {dilemma_title}

                {dilemma_description}

                Question: {question_text}

                Please answer thoughtfully, considering the moral implications and providing your reasoning.
                """
        
        return prompt