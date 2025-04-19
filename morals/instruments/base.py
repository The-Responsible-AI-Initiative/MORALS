# morals/instruments/base.py
import json
from pathlib import Path
from typing import Dict, List, Any, Optional


class Instrument:
    """Base class for all moral evaluation instruments."""
    
    def __init__(self, data_path: Optional[str] = None, data: Optional[Dict] = None):
        """
        Initialize an instrument from a file path or directly from data.
        
        Args:
            data_path: Path to the JSON file containing instrument data
            data: Direct dictionary of instrument data
        """
        if data is not None:
            self.data = data
        elif data_path is not None:
            self.data = self._load_data(data_path)
        else:
            raise ValueError("Either data_path or data must be provided")
        
        self.metadata = self.data.get("metadata", {})
        # No validate call here
    
    def _load_data(self, data_path: str) -> Dict:
        """Load instrument data from a JSON file."""
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"Instrument file not found: {data_path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def validate(self) -> None:
        """Validate the instrument data structure."""
        if not isinstance(self.data, dict):
            raise ValueError("Instrument data must be a dictionary")
        
        if "metadata" not in self.data:
            raise ValueError("Instrument data must contain metadata")
    
    def get_all_questions(self) -> List[Dict[str, Any]]:
        """Get all questions from the instrument."""
        raise NotImplementedError("Subclasses must implement get_all_questions")
    
    def get_question_by_id(self, question_id: str) -> Dict[str, Any]:
        """Get a specific question by its ID."""
        for question in self.get_all_questions():
            if question.get("id") == question_id:
                return question
        raise KeyError(f"Question with ID {question_id} not found")
    
    def get_prompt_for_question(self, question_id: str) -> str:
        """Get the prompt for a specific question."""
        question = self.get_question_by_id(question_id)
        return question.get("prompt", "")