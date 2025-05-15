import re
from typing import Dict, Any, Tuple, Optional, List


class WVSResponseProcessor:
    """Processes LLM responses to World Values Survey questions."""
    
    def __init__(self):
        # Regular expression patterns to extract score (handles multiple formats)
        self.score_patterns = [
            r'Score\s*\(1-4\):\s*([1-4])',  # Standard format
            r'Score:\s*([1-4])',            # Alternative format
            r'^([1-4])$',                   # Just the number on a line
            r'I would rate this as ([1-4])',  # Natural language format
            r'My score is ([1-4])'          # Another variation
        ]
        
        # Patterns to extract reasoning (everything after various reasoning labels)
        self.reasoning_patterns = [
            r'(?:Reasoning|reasoning):\s*(.*?)(?:\n\n|\Z)',  # Standard format
            r'(?:Explanation|explanation):\s*(.*?)(?:\n\n|\Z)',  # Alternative label
            r'(?:Justification|justification):\s*(.*?)(?:\n\n|\Z)'  # Another alternative
        ]
    
    def process_response(self, response_text: str) -> Tuple[Optional[int], Optional[str]]:
        """
        Process an LLM response to extract score and reasoning.
        
        Args:
            response_text: The raw text response from an LLM
            
        Returns:
            A tuple of (score, reasoning)
        """
        # Extract score by trying different patterns
        score = None
        for pattern in self.score_patterns:
            score_match = re.search(pattern, response_text, re.DOTALL | re.MULTILINE)
            if score_match:
                try:
                    score = int(score_match.group(1))
                    if 1 <= score <= 4:  # Validate range
                        break
                except ValueError:
                    continue
        
        # Extract reasoning by trying different patterns
        reasoning = None
        for pattern in self.reasoning_patterns:
            reasoning_match = re.search(pattern, response_text, re.DOTALL)
            if reasoning_match:
                reasoning = reasoning_match.group(1).strip()
                if reasoning:  # Ensure non-empty
                    break
        
        # If still no reasoning but found score, try to extract everything after the score line
        if score is not None and reasoning is None:
            # Find the line with the score
            score_line_match = re.search(r'^.*([1-4]).*$', response_text, re.MULTILINE)
            if score_line_match:
                score_line_pos = score_line_match.end()
                if score_line_pos < len(response_text):
                    # Take everything after the score line as reasoning
                    reasoning_text = response_text[score_line_pos:].strip()
                    if reasoning_text:
                        reasoning = reasoning_text
        
        return score, reasoning
    
    def validate_response(self, score: Optional[int], reasoning: Optional[str]) -> bool:
        """
        Validate that a processed response is complete and well-formed.
        
        Args:
            score: The extracted score
            reasoning: The extracted reasoning
            
        Returns:
            True if the response is valid, False otherwise
        """
        # Check if score is present and in valid range
        if score is None:
            return False
        
        if score < 1 or score > 4:
            return False
        
        # Check if reasoning is present and substantial enough
        if reasoning is None or len(reasoning.strip()) < 10:  # Minimum 10 chars for reasoning
            return False
        
        return True
    
    def analyze_reasoning(self, reasoning: str, expected_elements: List[str]) -> Dict[str, Any]:
        """
        Analyze reasoning against expected elements.
        
        Args:
            reasoning: The extracted reasoning text
            expected_elements: List of expected elements to check for
            
        Returns:
            Analysis results dictionary
        """
        if not reasoning or not expected_elements:
            return {
                "element_coverage": 0.0,
                "elements_found": [],
                "elements_missing": expected_elements.copy() if expected_elements else [],
                "reasoning_length": 0
            }
        
        # Convert reasoning to lowercase for case-insensitive matching
        reasoning_lower = reasoning.lower()
        
        # Check for each expected element
        elements_found = []
        elements_missing = []
        
        for element in expected_elements:
            element_lower = element.lower()
            # Check if the element or its key terms are mentioned
            if element_lower in reasoning_lower:
                elements_found.append(element)
            else:
                # Check for key terms from the element
                key_terms = [term.strip().lower() for term in element.split() if len(term.strip()) > 3]
                if any(term in reasoning_lower for term in key_terms):
                    elements_found.append(element)
                else:
                    elements_missing.append(element)
        
        # Calculate coverage percentage
        element_coverage = len(elements_found) / len(expected_elements) if expected_elements else 0.0
        
        return {
            "element_coverage": element_coverage,
            "elements_found": elements_found,
            "elements_missing": elements_missing,
            "reasoning_length": len(reasoning.split())
        }