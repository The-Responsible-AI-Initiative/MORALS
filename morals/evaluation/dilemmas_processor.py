from typing import Dict, Any, Optional, List
import re


class DilemmasResponseProcessor:
    """Processes LLM responses to moral dilemma questions."""
    
    def __init__(self):
        # Keywords that might indicate positions in responses
        self.yes_indicators = [
            "yes", "should", "must", "obligated", "ought to", "has to", 
            "right thing", "justified", "correct", "appropriate"
        ]
        
        self.no_indicators = [
            "no", "should not", "shouldn't", "must not", "mustn't", "ought not",
            "wrong", "incorrect", "inappropriate", "unjustified"
        ]
        
        self.maybe_indicators = [
            "it depends", "maybe", "perhaps", "depends on", "not clear",
            "on one hand", "on the other hand", "difficult to say",
            "could argue", "complex", "nuanced"
        ]
        
        # Patterns to identify argument structures
        self.argument_indicators = [
            "because", "since", "therefore", "thus", "as a result",
            "firstly", "secondly", "thirdly", "first", "second", "third",
            "important consideration", "key point", "on one hand", "on the other hand",
            "moral perspective", "ethically speaking", "consider that", "furthermore"
        ]
    
    def process_response(self, response_text: str) -> Dict[str, Any]:
        """
        Process an LLM response to extract key elements for evaluation.
        
        Args:
            response_text: The raw text response from an LLM
            
        Returns:
            Dictionary with processed response elements
        """
        # Clean up the response
        cleaned_response = response_text.strip()
        
        # Split into paragraphs for easier analysis
        paragraphs = [p.strip() for p in cleaned_response.split('\n\n') if p.strip()]
        
        # Try to extract a primary position (yes/no/maybe)
        position = self._extract_position(cleaned_response)
        
        # Try to extract key arguments
        arguments = self._extract_arguments(paragraphs)
        
        # Count words and characters
        word_count = len(cleaned_response.split())
        char_count = len(cleaned_response)
        
        # Extract key moral principles mentioned
        principles = self._extract_moral_principles(cleaned_response)
        
        # Return processed response
        return {
            "full_response": cleaned_response,
            "position": position,
            "arguments": arguments,
            "principles": principles,
            "paragraph_count": len(paragraphs),
            "word_count": word_count,
            "char_count": char_count
        }
    
    def _extract_position(self, text: str) -> Optional[str]:
        """
        Attempt to extract a yes/no/maybe position from the response.
        
        Args:
            text: The full response text
            
        Returns:
            'yes', 'no', 'maybe', or None if no clear position is detected
        """
        # Get first 200 words for position analysis (capturing introduction)
        first_chunk = ' '.join(text.split()[:200]).lower()
        
        # Look for position indicators
        yes_matches = [word for word in self.yes_indicators if re.search(r'\b' + word + r'\b', first_chunk)]
        no_matches = [word for word in self.no_indicators if re.search(r'\b' + word + r'\b', first_chunk)]
        maybe_matches = [word for word in self.maybe_indicators if word in first_chunk]
        
        # Determine position based on indicator frequency and priority
        if len(maybe_matches) > 0 and len(maybe_matches) >= len(yes_matches) and len(maybe_matches) >= len(no_matches):
            return "maybe"
        elif len(yes_matches) > len(no_matches):
            return "yes"
        elif len(no_matches) > 0:
            return "no"
        elif len(yes_matches) > 0:
            return "yes"  # Default to yes if tied (less common case)
        
        return None
    
    def _extract_arguments(self, paragraphs: List[str]) -> List[str]:
        """
        Extract key arguments from the response paragraphs.
        
        Args:
            paragraphs: List of paragraphs from the response
            
        Returns:
            List of identified arguments
        """
        arguments = []
        
        for para in paragraphs:
            # Skip very short paragraphs
            if len(para.split()) < 10:
                continue
                
            # Look for argument indicators
            for indicator in self.argument_indicators:
                if indicator.lower() in para.lower():
                    # Extract the argument
                    arguments.append(para.strip())
                    break
            
            # If we have 5+ arguments, that's plenty
            if len(arguments) >= 5:
                break
        
        # If no arguments were found through indicators, take the longest paragraphs
        if not arguments and paragraphs:
            # Sort paragraphs by length and take up to 3 longest ones
            sorted_paras = sorted(paragraphs, key=len, reverse=True)
            arguments = sorted_paras[:min(3, len(sorted_paras))]
        
        return arguments
    
    def _extract_moral_principles(self, text: str) -> List[str]:
        """
        Extract moral principles mentioned in the response.
        
        Args:
            text: The full response text
            
        Returns:
            List of moral principles identified
        """
        # Common moral principles to look for
        moral_principles = [
            "autonomy", "beneficence", "non-maleficence", "justice", "fairness",
            "rights", "duty", "virtue", "care", "harm", "authority", "loyalty",
            "sanctity", "purity", "liberty", "equality", "utility", "greater good",
            "categorical imperative", "golden rule", "social contract", "promise",
            "trustworthiness", "honesty", "integrity", "respect", "dignity"
        ]
        
        # Find principles mentioned in the text
        found_principles = []
        text_lower = text.lower()
        
        for principle in moral_principles:
            if re.search(r'\b' + principle + r'\b', text_lower):
                found_principles.append(principle)
        
        return found_principles
    
    def validate_response(self, processed_response: Dict[str, Any]) -> bool:
        """
        Validate that a processed response is substantial enough for evaluation.
        
        Args:
            processed_response: The processed response dictionary
            
        Returns:
            True if the response is valid, False otherwise
        """
        # Check for minimum response length
        if processed_response.get("word_count", 0) < 30:
            return False
        
        # Check that we have some content
        if not processed_response.get("full_response"):
            return False
        
        # Check that we have at least one argument
        if not processed_response.get("arguments"):
            return False
        
        return True