# morals/instruments/wvs.py
from typing import Dict, List, Any, Optional
from .base import Instrument


class WorldValuesSurveyInstrument(Instrument):
    """
    Implementation of the World Values Survey (WVS) instrument.
    This class handles WVS questions across different domains of values.
    """
    
    def __init__(self, data_path: Optional[str] = None, data: Optional[Dict] = None):
        # Call parent's init
        super().__init__(data_path, data)
        # Set domains attribute
        self.domains = self.data.get("domains", {})
        # Validate after attributes are set
        self.validate()
    
    def validate(self) -> None:
        """Validate WVS-specific data structure."""
        # Validate the base structure
        super().validate()
        
        if "domains" not in self.data:
            raise ValueError("WVS data must contain domains")
        
        for domain_key, domain in self.domains.items():
            if "name" not in domain:
                raise ValueError(f"Domain {domain_key} must have a name")
            
            if "description" not in domain:
                raise ValueError(f"Domain {domain_key} must have a description")
            
            if "questions" not in domain:
                raise ValueError(f"Domain {domain_key} must have questions")
            
            for question in domain.get("questions", []):
                if "id" not in question:
                    raise ValueError(f"Question in domain {domain_key} must have an id")
                
                if "prompt" not in question:
                    raise ValueError(f"Question {question.get('id')} must have a prompt")
                
                if "ground_truth" not in question:
                    raise ValueError(f"Question {question.get('id')} must have ground_truth")
    
    def get_all_questions(self) -> List[Dict[str, Any]]:
        """Get all questions from all domains."""
        questions = []
        
        for domain_key, domain in self.domains.items():
            for question in domain.get("questions", []):
                # Create a copy of the question with added domain context
                question_copy = question.copy()
                question_copy["domain"] = domain_key
                question_copy["domain_name"] = domain.get("name")
                question_copy["domain_description"] = domain.get("description")
                questions.append(question_copy)
        
        return questions
    
    def get_questions_by_domain(self, domain_key: str) -> List[Dict[str, Any]]:
        """Get all questions for a specific domain."""
        if domain_key not in self.domains:
            raise KeyError(f"Domain {domain_key} not found")
        
        domain = self.domains[domain_key]
        questions = []
        
        for question in domain.get("questions", []):
            # Create a copy of the question with added domain context
            question_copy = question.copy()
            question_copy["domain"] = domain_key
            question_copy["domain_name"] = domain.get("name")
            question_copy["domain_description"] = domain.get("description")
            questions.append(question_copy)
        
        return questions
    
    def get_questions_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get all questions of a specific category (e.g., importance, agreement)."""
        questions = []
        
        for domain_key, domain in self.domains.items():
            for question in domain.get("questions", []):
                if question.get("category") == category:
                    # Create a copy of the question with added domain context
                    question_copy = question.copy()
                    question_copy["domain"] = domain_key
                    question_copy["domain_name"] = domain.get("name")
                    question_copy["domain_description"] = domain.get("description")
                    questions.append(question_copy)
        
        return questions
    
    def get_questions_by_topic(self, topic: str) -> List[Dict[str, Any]]:
        """Get all questions on a specific topic."""
        questions = []
        
        for domain_key, domain in self.domains.items():
            for question in domain.get("questions", []):
                if question.get("topic") == topic:
                    # Create a copy of the question with added domain context
                    question_copy = question.copy()
                    question_copy["domain"] = domain_key
                    question_copy["domain_name"] = domain.get("name")
                    question_copy["domain_description"] = domain.get("description")
                    questions.append(question_copy)
        
        return questions
    
    def get_domain_names(self) -> Dict[str, str]:
        """Get a mapping of domain keys to human-readable names."""
        return {key: domain.get("name", key) 
                for key, domain in self.domains.items()}
    
    def get_prompt_for_question(self, question_id: str) -> str:
        """Get the prompt for a specific question."""
        question = self.get_question_by_id(question_id)
        return question.get("prompt", "")
    
    def get_scoring_scale(self, scale_type: str) -> Dict[str, str]:
        """Get the scoring scale for a specific type (e.g., importance_scale)."""
        scales = self.metadata.get("scoring_scales", {})
        return scales.get(scale_type, {})
    
    def get_question_category_scale(self, question_id: str) -> Dict[str, str]:
        """Get the appropriate scoring scale for a specific question based on its category."""
        question = self.get_question_by_id(question_id)
        category = question.get("category")
        
        if category == "importance":
            return self.get_scoring_scale("importance_scale")
        elif category == "agreement":
            return self.get_scoring_scale("agreement_scale")
        elif category == "frequency":
            return self.get_scoring_scale("frequency_scale")
        
        return {}