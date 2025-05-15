# morals/instruments/dilemmas.py
from typing import Dict, List, Any, Optional
from .base import Instrument


class MoralDilemmasInstrument(Instrument):
    """
    Implementation of the Moral Dilemmas instrument.
    This class handles ethical dilemma scenarios with open-ended questions.
    """
    
    def __init__(self, data_path: Optional[str] = None, data: Optional[Dict] = None):
        # Call parent's init
        super().__init__(data_path, data)
        # Set dilemmas attribute
        self.dilemmas = self.data.get("dilemmas", [])
        # Now validate after all attributes are set
        self.validate()
    
    def validate(self) -> None:
        """Validate dilemmas-specific data structure."""
        # Skip parent validation since dilemmas.json doesn't have a metadata field
        if not isinstance(self.data, dict):
            raise ValueError("Dilemmas data must be a dictionary")
        
        if "dilemmas" not in self.data:
            raise ValueError("Dilemmas data must contain 'dilemmas' list")
        
        for dilemma in self.dilemmas:
            if "id" not in dilemma:
                raise ValueError("Each dilemma must have an id")
            
            if "title" not in dilemma:
                raise ValueError(f"Dilemma {dilemma.get('id')} must have a title")
            
            if "description" not in dilemma:
                raise ValueError(f"Dilemma {dilemma.get('id')} must have a description")
            
            if "questions" not in dilemma:
                raise ValueError(f"Dilemma {dilemma.get('id')} must have questions")
            
            for question in dilemma.get("questions", []):
                if "id" not in question:
                    raise ValueError(f"Question in dilemma {dilemma.get('id')} must have an id")
                
                if "text" not in question:
                    raise ValueError(f"Question {question.get('id')} must have text")
                
                if "ground_truth" not in question:
                    raise ValueError(f"Question {question.get('id')} must have ground_truth")
    
    def get_all_questions(self) -> List[Dict[str, Any]]:
        """Get all questions from all dilemmas."""
        questions = []
        
        for dilemma in self.dilemmas:
            dilemma_id = dilemma.get("id")
            dilemma_title = dilemma.get("title")
            dilemma_description = dilemma.get("description")
            
            for question in dilemma.get("questions", []):
                # Create a question copy with added dilemma context
                question_copy = question.copy()
                question_copy["dilemma_id"] = dilemma_id
                question_copy["dilemma_title"] = dilemma_title
                question_copy["dilemma_description"] = dilemma_description
                questions.append(question_copy)
        
        return questions
    
    def get_questions_by_dilemma(self, dilemma_id: str) -> List[Dict[str, Any]]:
        """Get all questions for a specific dilemma."""
        for dilemma in self.dilemmas:
            if dilemma.get("id") == dilemma_id:
                questions = []
                dilemma_title = dilemma.get("title")
                dilemma_description = dilemma.get("description")
                
                for question in dilemma.get("questions", []):
                    # Create a question copy with added dilemma context
                    question_copy = question.copy()
                    question_copy["dilemma_id"] = dilemma_id
                    question_copy["dilemma_title"] = dilemma_title
                    question_copy["dilemma_description"] = dilemma_description
                    questions.append(question_copy)
                    
                return questions
        
        raise KeyError(f"Dilemma {dilemma_id} not found")
    
    def get_dilemma_by_id(self, dilemma_id: str) -> Dict[str, Any]:
        """Get a specific dilemma by its ID."""
        for dilemma in self.dilemmas:
            if dilemma.get("id") == dilemma_id:
                return dilemma
        
        raise KeyError(f"Dilemma {dilemma_id} not found")
    
    def get_dilemma_titles(self) -> Dict[str, str]:
        """Get a mapping of dilemma IDs to their titles."""
        return {dilemma.get("id"): dilemma.get("title") 
                for dilemma in self.dilemmas}
    
    def get_prompt_for_question(self, question_id: str) -> str:
        """Get the prompt for a specific question, including dilemma context."""
        # Get the full question with dilemma context
        question = self.get_question_by_id(question_id)
        
        # Extract information from the question
        dilemma_title = question.get("dilemma_title", "")
        dilemma_description = question.get("dilemma_description", "")
        question_text = question.get("text", "")
        
        # Construct a prompt with the dilemma context and question
        prompt = f"""# {dilemma_title}

{dilemma_description}

Question: {question_text}

Please answer thoughtfully, considering the moral implications and providing your reasoning.
"""
        return prompt
    
    def get_formatted_id(self, dilemma_id: str, question_id: str) -> str:
        """Create a combined ID for a question in a specific dilemma."""
        return f"{dilemma_id}-{question_id}"
    
    def parse_formatted_id(self, formatted_id: str) -> tuple:
        """Parse a formatted ID into dilemma_id and question_id."""
        parts = formatted_id.split("-", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid formatted ID: {formatted_id}")
        
        return parts[0], parts[1]