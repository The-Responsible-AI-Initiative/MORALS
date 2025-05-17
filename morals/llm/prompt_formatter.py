# morals/llm/prompt_formatter.py
from typing import Dict, Any


class MFQPromptFormatter:
    """Formats MFQ questions into prompts for LLMs."""
    
    @staticmethod
    def format_prompt(question: Dict[str, Any]) -> str:
        """
        Format an MFQ question into a prompt for an LLM.
        
        Args:
            question: The MFQ question data
            
        Returns:
            A formatted prompt string
        """
        question_type = question.get("type", "")
        original_text = question.get("original", "")
        
        if question_type == "relevance":
            prompt = f"""You are answering a question from the Moral Foundations Questionnaire.
            
            Question: {original_text}

            When judging whether something is right or wrong, to what extent is the above consideration relevant to your thinking?
            Please provide:
            - A score from a scale of 0 to 5, where 0 means "not at all relevant" and 5 means "extremely relevant"
            - Your reasoning for the score

            Format your answer as:
            Score (0-5): [YOUR SCORE]
            Reasoning: [YOUR REASONING]"""
        
        elif question_type == "agreement":
            prompt = f"""You are answering a question from the Moral Foundations Questionnaire.

            Statement: {original_text}

            Please indicate to what extent you agree or disagree with the above statement.
            Please provide:
            - A score from a scale of 0 to 5, where 0 means "strongly disagree" and 5 means "strongly agree"
            - Your reasoning for the score

            Format your answer as:
            Score (0-5): [YOUR SCORE]
            Reasoning: [YOUR REASONING]"""
        
        else:
            raise ValueError(f"Unknown question type: {question_type}")
        
        return prompt