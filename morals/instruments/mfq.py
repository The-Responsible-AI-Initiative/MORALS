# morals/instruments/mfq.py
from typing import Dict, List, Any
from .base import Instrument


class MoralFoundationsQuestionnaire(Instrument):
    """
    Implementation of the Moral Foundations Questionnaire (MFQ-30) instrument.
    """
    
    def __init__(self, data_path: str = None, data: Dict = None):
        # Call parent's init
        super().__init__(data_path, data)
        # Set foundations attribute
        self.foundations = self.data.get("foundations", {})
        # Now validate after all attributes are set
        self.validate()
    
    def validate(self) -> None:
        """Validate MFQ-specific data structure."""
        # First validate the base structure
        super().validate()
        
        if "foundations" not in self.data:
            raise ValueError("MFQ data must contain foundations")
        
        for foundation_key, foundation in self.foundations.items():
            if "name" not in foundation:
                raise ValueError(f"Foundation {foundation_key} must have a name")
            
            if "relevance_questions" not in foundation:
                raise ValueError(f"Foundation {foundation_key} must have relevance_questions")
            
            if "agreement_questions" not in foundation:
                raise ValueError(f"Foundation {foundation_key} must have agreement_questions")
    
    def get_all_questions(self) -> List[Dict[str, Any]]:
        """Get all questions from all foundations."""
        questions = []
        
        for foundation_key, foundation in self.foundations.items():
            for question in foundation.get("relevance_questions", []):
                question["foundation"] = foundation_key
                question["type"] = "relevance"
                questions.append(question)
            
            for question in foundation.get("agreement_questions", []):
                question["foundation"] = foundation_key
                question["type"] = "agreement"
                questions.append(question)
        
        return questions
    
    def get_questions_by_foundation(self, foundation: str) -> List[Dict[str, Any]]:
        """Get all questions for a specific foundation."""
        if foundation not in self.foundations:
            raise KeyError(f"Foundation {foundation} not found")
        
        questions = []
        foundation_data = self.foundations[foundation]
        
        for question in foundation_data.get("relevance_questions", []):
            question["foundation"] = foundation
            question["type"] = "relevance"
            questions.append(question)
        
        for question in foundation_data.get("agreement_questions", []):
            question["foundation"] = foundation
            question["type"] = "agreement"
            questions.append(question)
        
        return questions
    
    def get_foundation_names(self) -> Dict[str, str]:
        """Get a mapping of foundation keys to human-readable names."""
        return {key: foundation.get("name", key) 
                for key, foundation in self.foundations.items()}
    
    def get_scale_labels(self, question_type: str) -> Dict[str, str]:
        """Get the scale labels for a question type (relevance or agreement)."""
        scales = self.metadata.get("scoring_scales", {})
        return scales.get(question_type, {})