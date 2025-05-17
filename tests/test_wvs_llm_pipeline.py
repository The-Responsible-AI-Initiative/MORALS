# tests/test_wvs_llm_pipeline.py
import asyncio
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from morals.instruments.wvs import WorldValuesSurveyInstrument
from morals.llm.anthropic import AnthropicInterface
from morals.pipeline import MoralEvaluationPipeline


async def test_wvs_pipeline():
    """Test the World Values Survey evaluation pipeline with a real LLM."""
    print("=== MORALS WVS Test with Real LLM ===")
    
    # 1. Load WVS data
    print("\n1. Loading World Values Survey instrument...")
    data_path = project_root / "data" / "instruments" / "wvs.json"
    
    if not data_path.exists():
        print(f"Error: WVS data file not found at {data_path}")
        print(f"Current working directory: {os.getcwd()}")
        return False
    
    try:
        wvs = WorldValuesSurveyInstrument(data_path=str(data_path))
        domain_count = len(wvs.domains)
        question_count = len(wvs.get_all_questions())
        print(f"✓ Successfully loaded WVS with {domain_count} domains and {question_count} total questions")
    except Exception as e:
        print(f"Error loading WVS data: {e}")
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
        pipeline = MoralEvaluationPipeline(llm=llm, wvs=wvs, output_dir=str(output_dir))
        print("✓ Pipeline initialized successfully")
        print(f"✓ Results will be saved to {output_dir}")
    except Exception as e:
        print(f"Error creating pipeline: {e}")
        return False
    
    # 4. Test with sample WVS question
    print("\n4. Testing with a sample WVS question...")
    
    # Select a sample question (first one from core_values domain)
    domain_key = "core_values"
    questions = wvs.get_questions_by_domain(domain_key)
    if not questions:
        print(f"Error: No questions found in domain {domain_key}")
        return False
    
    question = questions[0]
    question_id = question["id"]
    
    try:
        print(f"Domain: {question['domain_name']}")
        print(f"Question ID: {question_id}")
        print(f"Category: {question['category']}")
        print(f"Topic: {question['topic']}")
        
        # Generate and evaluate response
        print("\nGenerating and evaluating LLM response...")
        result = await pipeline.evaluate_wvs_question(question_id)
        
        # Display results
        print("\nLLM response:")
        print(result["raw_response"][:300] + "..." if len(result["raw_response"]) > 300 else result["raw_response"])
        
        print("\nEvaluation results:")
        print(f"✓ Extracted score: {result['extracted_score']}")
        print(f"✓ Ground truth mean: {result['ground_truth_mean']}")
        print(f"✓ In acceptable range: {result['in_acceptable_range']}")
        
        if result.get("overall_alignment") is not None:
            print(f"✓ Overall alignment: {result['overall_alignment']:.2f}")
        
        print(f"✓ Reasoning quality: {result['reasoning_quality']:.2f}")
        
        # Show reasoning analysis
        print("\nReasoning analysis:")
        for element in result["reasoning_analysis"].get("elements_found", []):
            print(f"✓ Found: {element}")
        for element in result["reasoning_analysis"].get("elements_missing", []):
            print(f"✗ Missing: {element}")
    except Exception as e:
        print(f"Error evaluating WVS question: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 5. Test evaluating a domain
    print("\n5. Testing evaluation of a domain...")
    try:
        # Evaluate domain with max 2 questions to save time
        print(f"Evaluating domain {domain_key} with max 2 questions...")
        result = await pipeline.evaluate_wvs_domain(domain_key, max_questions=2)
        
        # Display results
        print("\nDomain metrics:")
        for metric, value in result["metrics"].items():
            if isinstance(value, float):
                print(f"✓ {metric}: {value:.2f}")
            else:
                print(f"✓ {metric}: {value}")
    except Exception as e:
        print(f"Error evaluating domain: {e}")
        return False
    
    # 6. Test evaluating a category
    print("\n6. Testing evaluation of a category...")
    try:
        # Choose a category (e.g., importance or agreement)
        category = "importance"
        print(f"Evaluating category {category} with max 2 questions...")
        result = await pipeline.evaluate_wvs_category(category, max_questions=2)
        
        # Display results
        print("\nCategory metrics:")
        for metric, value in result["metrics"].items():
            if isinstance(value, float):
                print(f"✓ {metric}: {value:.2f}")
            else:
                print(f"✓ {metric}: {value}")
    except Exception as e:
        print(f"Error evaluating category: {e}")
        return False
    
    print("\n=== Test completed successfully ===")
    return True


if __name__ == "__main__":
    success = asyncio.run(test_wvs_pipeline())
    if not success:
        print("\nTest failed with errors.")
        exit(1)