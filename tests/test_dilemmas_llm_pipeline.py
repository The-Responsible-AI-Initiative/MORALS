# tests/test_dilemmas_llm_pipeline.py
import asyncio
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from morals.instruments.dilemmas import MoralDilemmasInstrument
from morals.llm.anthropic import AnthropicInterface
from morals.pipeline import MoralEvaluationPipeline


async def test_dilemmas_pipeline():
    """Test the moral dilemmas evaluation pipeline with a real LLM."""
    print("=== MORALS Dilemmas Test with Real LLM ===")
    
    # 1. Load dilemmas data
    print("\n1. Loading moral dilemmas instrument...")
    data_path = project_root / "data" / "instruments" / "dilemmas.json"
    
    if not data_path.exists():
        print(f"Error: Dilemmas data file not found at {data_path}")
        print(f"Current working directory: {os.getcwd()}")
        return False
    
    try:
        dilemmas = MoralDilemmasInstrument(data_path=str(data_path))
        dilemma_count = len(dilemmas.dilemmas)
        question_count = len(dilemmas.get_all_questions())
        print(f"✓ Successfully loaded {dilemma_count} moral dilemmas with {question_count} total questions")
    except Exception as e:
        print(f"Error loading dilemmas data: {e}")
        return False
    
    # 2. Initialize LLM interface
    print("\n2. Setting up LLM interface...")
    try:
        llm = AnthropicInterface(model_name="claude-3-haiku-20240307")
        print(f"✓ Successfully initialized {llm.model_info['name']} interface")
    except Exception as e:
        print(f"Error initializing LLM interface: {e}")
        return False
    
    # 3. Create pipeline
    print("\n3. Setting up evaluation pipeline...")
    try:
        output_dir = project_root / "results"
        pipeline = MoralEvaluationPipeline(llm=llm, dilemmas=dilemmas, output_dir=str(output_dir))
        print("✓ Pipeline initialized successfully")
        print(f"✓ Results will be saved to {output_dir}")
    except Exception as e:
        print(f"Error creating pipeline: {e}")
        return False
    
    # 4. Test with sample dilemma question
    print("\n4. Testing with a sample dilemma question...")
    
    # Select a sample dilemma and question
    dilemma_id = "Dilemma_I"  # The Joe and Camp Money dilemma
    question_id = "Q1"        # Should Joe refuse to give his father the money?
    
    try:
        # Get dilemma info
        dilemma = dilemmas.get_dilemma_by_id(dilemma_id)
        print(f"Dilemma: {dilemma['title']}")
        
        # Get question info
        questions = dilemmas.get_questions_by_dilemma(dilemma_id)
        question = next((q for q in questions if q["id"] == question_id), None)
        print(f"Question: {question['text']}")
        
        # Generate and evaluate response
        print("\nGenerating and evaluating LLM response...")
        result = await pipeline.evaluate_dilemma_question(dilemma_id, question_id)
        
        # Display results
        print("\nLLM response:")
        print(result["raw_response"][:300] + "..." if len(result["raw_response"]) > 300 else result["raw_response"])
        
        print("\nEvaluation results:")
        print(f"✓ Semantic similarity: {result['semantic_similarity']:.2f}")
        print(f"✓ Criteria satisfaction: {result['criteria_satisfaction']:.2f}")
        print(f"✓ Reasoning score: {result['reasoning_score']:.2f}")
        print(f"✓ Overall score: {result['overall_score']:.2f}")
    except Exception as e:
        print(f"Error evaluating dilemma question: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 5. Test evaluating a full dilemma (with limit)
    print("\n5. Testing evaluation of a full dilemma...")
    try:
        # Evaluate dilemma with max 2 questions to save time
        print(f"Evaluating dilemma {dilemma_id} with max 2 questions...")
        result = await pipeline.evaluate_dilemma(dilemma_id, max_questions=2)
        
        # Display results
        print("\nDilemma scores:")
        for metric, value in result["scores"].items():
            if isinstance(value, float):
                print(f"✓ {metric}: {value:.2f}")
            else:
                print(f"✓ {metric}: {value}")
    except Exception as e:
        print(f"Error evaluating dilemma: {e}")
        return False
    
    print("\n=== Test completed successfully ===")
    return True


if __name__ == "__main__":
    success = asyncio.run(test_dilemmas_pipeline())
    if not success:
        print("\nTest failed with errors.")
        exit(1)