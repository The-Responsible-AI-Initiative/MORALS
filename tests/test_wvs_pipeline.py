# tests/test_wvs_pipeline.py
import asyncio
import json
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from morals.instruments.wvs import WorldValuesSurveyInstrument
from morals.evaluation.wvs_processor import WVSResponseProcessor
from morals.evaluation.wvs_evaluator import WVSEvaluator

# Example LLM responses for testing without API calls
MOCK_RESPONSES = {
    "cv_1": """Score (1-4): 1
Reasoning: Family is very important in life as it provides the fundamental social structure for emotional support, personal development, and stability. Family relationships are often our first and most enduring connections, teaching us values, providing security, and offering unconditional love and acceptance. The family unit serves as the primary context for learning interpersonal skills, emotional intelligence, and cultural transmission. Additionally, families create a support network that can help individuals navigate challenges and celebrate successes throughout life.""",

    "st_1": """Score (1-4): 2
Reasoning: I agree that most people can be trusted, but with reasonable caution. While most people generally act with good intentions in everyday interactions, complete trust without verification isn't always prudent. Trust often depends on context - we might trust strangers differently in various situations. My view is based on the understanding that people generally follow social norms and have inherent empathy, but individual experiences, cultural factors, and specific circumstances can all influence trustworthiness. A balanced approach that recognizes both human goodness and the reality of occasional breaches of trust seems most reasonable.""",

    "wv_1": """Score (1-4): 1
Reasoning: Work is very important in life for several key reasons. Beyond providing financial stability and meeting basic needs, work offers a sense of purpose, identity, and personal fulfillment. It allows individuals to contribute meaningfully to society while developing skills and relationships. Work structures our time and provides opportunities for achievement and growth. The significance of work extends beyond economic necessity to encompass psychological wellbeing, social connection, and self-actualization. Even when considering work-life balance, the role of purposeful work remains central to a well-rounded life."""
}


async def test_pipeline():
    """Test the full WVS evaluation pipeline with mock responses."""
    print("=== World Values Survey Framework Test ===")
    
    # 1. Load WVS data
    print("\n1. Loading WVS instrument...")
    data_path = project_root / "data" / "instruments" / "wvs.json"
    
    if not data_path.exists():
        print(f"Error: WVS data file not found at {data_path}")
        print(f"Current working directory: {os.getcwd()}")
        return False
    
    try:
        wvs = WorldValuesSurveyInstrument(data_path=str(data_path))
        print(f"✓ Successfully loaded WVS with {len(wvs.get_all_questions())} questions")
        print(f"✓ Found {len(wvs.domains)} domains")
    except Exception as e:
        print(f"Error loading WVS data: {e}")
        return False
    
    # 2. Create evaluator
    print("\n2. Setting up evaluator...")
    try:
        evaluator = WVSEvaluator(wvs)
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
            question = wvs.get_question_by_id(question_id)
            domain = question.get("domain", "")
            domain_name = question.get("domain_name", "")
            category = question.get("category", "")
            topic = question.get("topic", "")
            
            print(f"Domain: {domain_name} ({domain})")
            print(f"Category: {category}, Topic: {topic}")
            
            # Display mock response (truncated for readability)
            response_preview = response_text.split("\n")[0]
            if len(response_text) > 100:
                print(f"\nMock LLM response (excerpt):\n{response_preview}...")
            else:
                print(f"\nMock LLM response:\n{response_text}")
            
            # Evaluate response
            result = evaluator.evaluate_response(question_id, response_text)
            results.append(result)
            
            # Display results
            print("\nEvaluation results:")
            print(f"✓ Extracted score: {result['extracted_score']}")
            print(f"✓ Ground truth mean: {result['ground_truth_mean']} (std: {result['ground_truth_std']})")
            
            if result['in_acceptable_range']:
                print(f"✓ Score is within acceptable range: {result['ground_truth_acceptable_range']}")
            else:
                print(f"✗ Score is outside acceptable range: {result['ground_truth_acceptable_range']}")
            
            # Show score metrics
            score_metrics = result.get('score_metrics', {})
            if 'normalized_distance' in score_metrics:
                print(f"✓ Score distance: {score_metrics['normalized_distance']:.2f} (normalized)")
            
            if 'z_score' in score_metrics:
                print(f"✓ Z-score: {score_metrics['z_score']:.2f}")
            
            if 'percentile' in score_metrics:
                print(f"✓ Percentile: {score_metrics['percentile']:.1f}%")
            
            # Show reasoning analysis
            reasoning_analysis = result.get('reasoning_analysis', {})
            element_coverage = reasoning_analysis.get('element_coverage', 0) * 100
            print(f"✓ Reasoning element coverage: {element_coverage:.1f}%")
            
            elements_found = reasoning_analysis.get('elements_found', [])
            if elements_found:
                print(f"✓ Elements found: {', '.join(elements_found)}")
            
            elements_missing = reasoning_analysis.get('elements_missing', [])
            if elements_missing:
                print(f"✗ Elements missing: {', '.join(elements_missing)}")
            
            # Show overall alignment
            overall_alignment = result.get('overall_alignment', 0) * 100
            print(f"✓ Overall alignment: {overall_alignment:.1f}%")
            
            print(f"✓ Valid response: {result['is_valid_response']}")
        except Exception as e:
            print(f"Error evaluating response: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # 4. Calculate domain metrics
    print("\n4. Calculating domain metrics...")
    try:
        domain_metrics = evaluator.calculate_domain_metrics(results)
        print("\nDomain metrics:")
        for domain, metrics in domain_metrics.items():
            avg_alignment = metrics.get('avg_alignment', 0) * 100 if metrics.get('avg_alignment') is not None else 0
            avg_reasoning_quality = metrics.get('avg_reasoning_quality', 0) * 100
            acceptable_ratio = metrics.get('acceptable_range_ratio', 0) * 100
            
            print(f"✓ {metrics.get('name', domain)}:")
            print(f"  - Alignment: {avg_alignment:.1f}%")
            print(f"  - Reasoning quality: {avg_reasoning_quality:.1f}%")
            print(f"  - Acceptable range ratio: {acceptable_ratio:.1f}%")
    except Exception as e:
        print(f"Error calculating domain metrics: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 5. Calculate overall metrics
    print("\n5. Calculating overall metrics...")
    try:
        overall_metrics = evaluator.calculate_overall_metrics(results)
        print("\nOverall metrics:")
        
        valid_ratio = overall_metrics.get('valid_responses', 0) / overall_metrics.get('total_questions', 1) * 100
        avg_alignment = overall_metrics.get('avg_overall_alignment', 0) * 100 if overall_metrics.get('avg_overall_alignment') is not None else 0
        avg_reasoning = overall_metrics.get('avg_reasoning_quality', 0) * 100
        acceptable_ratio = overall_metrics.get('acceptable_range_ratio', 0) * 100
        
        print(f"✓ Valid responses: {overall_metrics.get('valid_responses', 0)}/{overall_metrics.get('total_questions', 0)} ({valid_ratio:.1f}%)")
        print(f"✓ Average alignment: {avg_alignment:.1f}%")
        print(f"✓ Average reasoning quality: {avg_reasoning:.1f}%")
        print(f"✓ Acceptable range ratio: {acceptable_ratio:.1f}%")
    except Exception as e:
        print(f"Error calculating overall metrics: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 6. Analyze category performance
    print("\n6. Analyzing category performance...")
    try:
        category_metrics = evaluator.analyze_category_performance(results)
        print("\nCategory performance:")
        for category, metrics in category_metrics.items():
            avg_alignment = metrics.get('avg_alignment', 0) * 100 if metrics.get('avg_alignment') is not None else 0
            avg_reasoning_quality = metrics.get('avg_reasoning_quality', 0) * 100
            
            print(f"✓ {category}:")
            print(f"  - Questions: {metrics.get('question_count', 0)}")
            print(f"  - Alignment: {avg_alignment:.1f}%")
            print(f"  - Reasoning quality: {avg_reasoning_quality:.1f}%")
    except Exception as e:
        print(f"Error analyzing category performance: {e}")
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