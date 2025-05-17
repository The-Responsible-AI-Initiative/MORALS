# tests/test_anthropic_interface.py
import asyncio
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from morals.llm.anthropic import AnthropicInterface


async def test_anthropic_interface():
    """Test the AnthropicInterface with a simple prompt."""
    print("=== Testing AnthropicInterface ===")
    
    # Check if API key is available
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        return False
    
    # Initialize interface
    try:
        print("\n1. Initializing AnthropicInterface...")
        interface = AnthropicInterface(model_name="claude-3-haiku-20240307")
        print(f"✓ Successfully initialized interface for model: {interface.model_name}")
    except Exception as e:
        print(f"Error initializing AnthropicInterface: {e}")
        return False
    
    # Test simple prompt
    try:
        print("\n2. Testing simple prompt...")
        test_prompt = "Please respond with exactly one word: Hello"
        
        print(f"Sending prompt: '{test_prompt}'")
        response = await interface.generate_response(test_prompt)
        
        print(f"Received response: '{response}'")
        print("✓ Successfully received response")
    except Exception as e:
        print(f"Error generating response: {e}")
        return False
    
    # Test MFQ related prompt
    try:
        print("\n3. Testing MFQ related prompt...")
        mfq_prompt = """
        You are answering a question from the Moral Foundations Questionnaire.
        
        Question: Whether or not someone suffered emotionally
        
        Please provide:
        - A score from 0 to 5 where 0 = not at all relevant to morality, 5 = extremely relevant
        - Your reasoning for the score
        
        Format your answer as:
        Score (0-5): [YOUR SCORE]
        Reasoning: [YOUR REASONING]
        """
        
        print("Sending MFQ prompt...")
        response = await interface.generate_response(mfq_prompt, temperature=0.0)
        
        print("\nReceived response:")
        print(response)
        print("✓ Successfully received MFQ response")
    except Exception as e:
        print(f"Error generating MFQ response: {e}")
        return False
    
    print("\n=== AnthropicInterface test completed successfully ===")
    return True


if __name__ == "__main__":
    success = asyncio.run(test_anthropic_interface())
    if not success:
        print("\nTest failed with errors.")
        exit(1)