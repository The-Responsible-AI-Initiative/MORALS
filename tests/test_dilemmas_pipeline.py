# tests/test_dilemmas_pipeline.py
import asyncio
import json
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from morals.instruments.dilemmas import MoralDilemmasInstrument
from morals.evaluation.dilemmas_processor import DilemmasResponseProcessor
from morals.evaluation.dilemmas_evaluator import DilemmasEvaluator

# Example LLM responses for testing without API calls
MOCK_RESPONSES = {
    "Dilemma_I-Q1": """
I believe Joe should have a discussion with his father about the situation rather than flatly refusing to give him the money. 

While Joe has worked hard to earn the money specifically for camp and his father did promise him he could use it for that purpose, there are multiple moral considerations at play here.

First, Joe's father has parental authority, but this authority isn't absolute. The father made a clear promise that created legitimate expectations, and breaking that promise without good reason damages trust in their relationship. Joe earned the money through his own efforts, giving him a reasonable claim to decide how it's used.

However, family relationships also involve supporting each other in times of need. The situation calls for communication - Joe could express his disappointment about missing camp while trying to understand his father's position. Perhaps there's a compromise where Joe could lend part of the money and still attend camp, or the father could find alternative ways to fund his fishing trip.

The most important thing is maintaining mutual respect. Joe has legitimate grounds to remind his father of his promise and the work he did, but should do so respectfully while acknowledging their relationship.
""",

    "Dilemma_III-Q1": """
This is a profound ethical dilemma that pits the value of human life against property rights and the rule of law.

I believe Heinz should steal the drug. When a human life is at stake - especially that of a spouse - and all legal means have been exhausted, the moral imperative to save a life outweighs the prohibition against theft. The druggist's position seems particularly unethical given the extreme markup (charging 10x the cost) and unwillingness to work with Heinz despite knowing the life-or-death circumstances.

Several ethical frameworks support this position:
1. Utilitarian reasoning suggests that saving a life creates greater overall good than protecting property rights in this specific case
2. From a care ethics perspective, Heinz has a special obligation to his wife that supersedes other considerations
3. Even Kantian ethics might permit this action if we consider whether we'd want everyone to have the right to steal life-saving medication when no alternatives exist

However, Heinz should be prepared to accept the legal consequences of his actions, minimize harm by taking only what's needed, and attempt to make restitution later. The moral rightness of an action doesn't necessarily eliminate its legal consequences.

This case highlights the tension between formal laws and deeper moral principles when they come into conflict.
"""
}


async def test_pipeline():
    """Test the full Moral Dilemmas evaluation pipeline with mock responses."""
    print("=== MORALS Dilemmas Framework Test ===")
    
    # 1. Load Dilemmas data
    print("\n1. Loading Moral Dilemmas instrument...")
    data_path = project_root / "data" / "instruments" / "dilemmas.json"
    
    if not data_path.exists():
        print(f"Error: Dilemmas data file not found at {data_path}")
        print(f"Current working directory: {os.getcwd()}")
        return False
    
    try:
        dilemmas = MoralDilemmasInstrument(data_path=str(data_path))
        print(f"✓ Successfully loaded Moral Dilemmas with {len(dilemmas.get_all_questions())} questions")
        print(f"✓ Found {len(dilemmas.dilemmas)} distinct dilemmas")
    except Exception as e:
        print(f"Error loading Dilemmas data: {e}")
        return False
    
    # 2. Create evaluator
    print("\n2. Setting up evaluator...")
    try:
        evaluator = DilemmasEvaluator(dilemmas)
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
            # Split combined ID
            dilemma_id, q_id = question_id.split("-")
            
            # Get the dilemma and question
            dilemma = dilemmas.get_dilemma_by_id(dilemma_id)
            questions = dilemmas.get_questions_by_dilemma(dilemma_id)
            question = next((q for q in questions if q.get("id") == q_id), None)
            
            print(f"Dilemma: {dilemma['title']}")
            print(f"Question: {question['text']}")
            
            # Display truncated mock response (first paragraph)
            first_para = response_text.strip().split('\n\n')[0] if '\n\n' in response_text else response_text
            print("\nMock LLM response (excerpt):")
            print(f"{first_para}...")
            
            # Evaluate response
            result = evaluator.evaluate_response(question_id, response_text)
            results.append(result)
            
            # Display results
            print("\nEvaluation results:")
            
            # Show the detected position
            position = result['processed_response'].get('position')
            print(f"✓ Position detected: {position if position else 'None'}")
            
            # Show word count
            print(f"✓ Word count: {result['processed_response'].get('word_count')}")
            
            # Show scores
            print(f"✓ Semantic similarity: {result['semantic_similarity']:.2f}")
            print(f"✓ Criteria satisfaction: {result['criteria_satisfaction']:.2f}")
            print(f"✓ Reasoning quality: {result['reasoning_score']:.2f}")
            print(f"✓ Overall score: {result['overall_score']:.2f}")
            
            # Show principles detected
            principles = result['processed_response'].get('principles', [])
            if principles:
                print(f"✓ Principles detected: {', '.join(principles[:3])}" + 
                      (f" and {len(principles)-3} more" if len(principles) > 3 else ""))
            
            print(f"✓ Valid response: {result['is_valid_response']}")
        except Exception as e:
            print(f"Error evaluating response: {e}")
            print(f"Exception details: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    # 4. Calculate dilemma scores
    print("\n4. Calculating dilemma scores...")
    try:
        dilemma_scores = evaluator.calculate_dilemma_scores(results)
        print("\nDilemma scores:")
        for dilemma_id, scores in dilemma_scores.items():
            print(f"✓ {dilemma_id} - {scores.get('title')}: {scores.get('avg_overall_score'):.2f}")
            
        # Also calculate aggregate scores
        aggregate_scores = evaluator.calculate_aggregate_scores(results)
        print("\nAggregate scores across all dilemmas:")
        print(f"✓ Average overall score: {aggregate_scores.get('avg_overall_score'):.2f}")
        print(f"✓ Average semantic similarity: {aggregate_scores.get('avg_semantic_similarity'):.2f}")
        print(f"✓ Average criteria satisfaction: {aggregate_scores.get('avg_criteria_satisfaction'):.2f}")
        print(f"✓ Average reasoning quality: {aggregate_scores.get('avg_reasoning_score'):.2f}")
    except Exception as e:
        print(f"Error calculating scores: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n=== Test completed successfully ===")
    return True


if __name__ == "__main__":
    success = asyncio.run(test_pipeline())
    if not success:
        print("\nTest failed with errors.")
        exit(1)