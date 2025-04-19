# tests/test_morals_pipeline.py
import asyncio
import json
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from morals.instruments.mfq import MoralFoundationsQuestionnaire
from morals.evaluation.processor import MFQResponseProcessor
from morals.evaluation.mfq_evaluator import MFQEvaluator

# Example LLM responses for testing without API calls
MOCK_RESPONSES = {
    "care_r1": """Score (0-5): 4
Reasoning: Emotional suffering is a core aspect of morality. When we consider whether someone suffered emotionally, we're addressing a fundamental dimension of harm and care. Emotional pain can be just as impactful as physical pain, and recognizing this is essential for moral judgments.""",

    "fairness_r2": """Score (0-5): 5
Reasoning: Fairness is a cornerstone of morality. When someone acts unfairly, they violate basic principles of justice and equal treatment that are essential for social cohesion and trust. Evaluating the fairness of actions is critical for moral judgments."""
}


async def test_pipeline():
    """Test the full MFQ evaluation pipeline with mock responses."""
    print("=== MORALS Framework Test ===")
    
    # 1. Load MFQ data
    print("\n1. Loading MFQ instrument...")
    data_path = project_root / "data" / "instruments" / "mfq.json"
    
    if not data_path.exists():
        print(f"Error: MFQ data file not found at {data_path}")
        print(f"Current working directory: {os.getcwd()}")
        return False
    
    try:
        mfq = MoralFoundationsQuestionnaire(data_path=str(data_path))
        print(f"✓ Successfully loaded MFQ with {len(mfq.get_all_questions())} questions")
        print(f"✓ Found {len(mfq.foundations)} moral foundations")
    except Exception as e:
        print(f"Error loading MFQ data: {e}")
        return False
    
    # 2. Create evaluator
    print("\n2. Setting up evaluator...")
    try:
        evaluator = MFQEvaluator(mfq)
        print("✓ Evaluator initialized successfully")
    except Exception as e:
        print(f"Error creating evaluator: {e}")
        return False
    
    # 3. Process and evaluate mock responses
    print("\n3. Processing and evaluating responses...")
    results = []
    
    for question_id, response_text in MOCK_RESPONSES.items():
        print(f"\nEvaluating question: {question_id}")
        
        try:
            # Get original question
            question = mfq.get_question_by_id(question_id)
            print(f"Original question: {question['original']}")
            print(f"Foundation: {question['foundation']}")
            
            # Display mock response
            print("\nMock LLM response:")
            print(response_text)
            
            # Evaluate response
            result = evaluator.evaluate_response(question_id, response_text)
            results.append(result)
            
            # Display results
            print("\nEvaluation results:")
            print(f"✓ Extracted score: {result['extracted_score']}")
            print(f"✓ Ground truth mean: {result['ground_truth_mean']}")
            print(f"✓ Alignment score: {result['alignment_score']:.2f}")
            print(f"✓ Valid response: {result['is_valid_response']}")
        except Exception as e:
            print(f"Error evaluating response: {e}")
            return False
    
    # 4. Calculate foundation alignment
    print("\n4. Calculating foundation alignment...")
    try:
        foundation_alignment = evaluator.calculate_foundation_alignment(results)
        print("\nFoundation alignment scores:")
        for foundation, score in foundation_alignment.items():
            print(f"✓ {foundation}: {score:.2f}")
    except Exception as e:
        print(f"Error calculating foundation alignment: {e}")
        return False
    
    print("\n=== Test completed successfully ===")
    return True


if __name__ == "__main__":
    success = asyncio.run(test_pipeline())
    if not success:
        print("\nTest failed with errors.")
        exit(1)