# tests/test_full_moral_pipeline.py
import asyncio
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from morals.instruments.mfq import MoralFoundationsQuestionnaire
from morals.instruments.dilemmas import MoralDilemmasInstrument
from morals.instruments.wvs import WorldValuesSurveyInstrument
from morals.llm.anthropic import AnthropicInterface
from morals.pipeline import MoralEvaluationPipeline


async def test_full_pipeline():
   """Test the complete moral evaluation pipeline with MFQ, Dilemmas, and WVS."""
   print("=== MORALS Full Evaluation Pipeline Test ===")
   
   # 1. Load instruments
   print("\n1. Loading moral instruments...")
   mfq_path = project_root / "data" / "instruments" / "mfq.json"
   dilemmas_path = project_root / "data" / "instruments" / "dilemmas.json"
   wvs_path = project_root / "data" / "instruments" / "wvs.json"
   
   # Check if instrument files exist
   mfq_exists = mfq_path.exists()
   dilemmas_exists = dilemmas_path.exists()
   wvs_exists = wvs_path.exists()
   
   if not mfq_exists and not dilemmas_exists and not wvs_exists:
       print("Error: No instrument data files found.")
       return False
   
   # Load available instruments
   mfq = None
   dilemmas = None
   wvs = None
   
   if mfq_exists:
       try:
           mfq = MoralFoundationsQuestionnaire(data_path=str(mfq_path))
           print(f"✓ Successfully loaded MFQ with {len(mfq.foundations)} foundations")
       except Exception as e:
           print(f"Error loading MFQ data: {e}")
   else:
       print("MFQ data file not found, skipping.")
   
   if dilemmas_exists:
       try:
           dilemmas = MoralDilemmasInstrument(data_path=str(dilemmas_path))
           print(f"✓ Successfully loaded {len(dilemmas.dilemmas)} moral dilemmas")
       except Exception as e:
           print(f"Error loading dilemmas data: {e}")
   else:
       print("Dilemmas data file not found, skipping.")
   
   if wvs_exists:
       try:
           wvs = WorldValuesSurveyInstrument(data_path=str(wvs_path))
           print(f"✓ Successfully loaded WVS with {len(wvs.domains)} domains")
       except Exception as e:
           print(f"Error loading WVS data: {e}")
   else:
       print("WVS data file not found, skipping.")
   
   if not mfq and not dilemmas and not wvs:
       print("Error: Could not load any instruments.")
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
       pipeline = MoralEvaluationPipeline(
           llm=llm,
           mfq=mfq,
           dilemmas=dilemmas,
           wvs=wvs,
           output_dir=str(output_dir)
       )
       print("✓ Pipeline initialized successfully")
       print(f"✓ Results will be saved to {output_dir}")
   except Exception as e:
       print(f"Error creating pipeline: {e}")
       return False
   
   # 4. Run sample evaluations
   print("\n4. Running sample evaluations...")
   
   # Test MFQ
   if mfq:
       print("\n4.1. Testing MFQ (one foundation, limited questions)...")
       try:
           # Select a foundation to test
           foundation = list(mfq.foundations.keys())[0]
           print(f"Evaluating foundation: {mfq.get_foundation_names().get(foundation, foundation)}")
           
           # Evaluate with just one question per foundation to save time
           result = await pipeline.evaluate_mfq_foundation(foundation, max_questions=1)
           
           # Display results
           print(f"✓ Alignment score: {result['alignment_score']:.2f if result['alignment_score'] else 'N/A'}")
       except Exception as e:
           print(f"Error evaluating MFQ: {e}")
           return False
   
   # Test Dilemmas
   if dilemmas:
       print("\n4.2. Testing Dilemmas (one dilemma, limited questions)...")
       try:
           # Select a dilemma to test
           dilemma_id = dilemmas.dilemmas[0].get("id")
           dilemma_title = dilemmas.dilemmas[0].get("title")
           print(f"Evaluating dilemma: {dilemma_title}")
           
           # Evaluate with just one question to save time
           result = await pipeline.evaluate_dilemma(dilemma_id, max_questions=1)
           
           # Display results
           print("✓ Dilemma scores:")
           scores = result["scores"]
           print(f"  - Overall score: {scores.get('avg_overall_score', 0):.2f}")
           print(f"  - Semantic similarity: {scores.get('avg_semantic_similarity', 0):.2f}")
           print(f"  - Criteria satisfaction: {scores.get('avg_criteria_satisfaction', 0):.2f}")
           print(f"  - Reasoning score: {scores.get('avg_reasoning_score', 0):.2f}")
       except Exception as e:
           print(f"Error evaluating dilemmas: {e}")
           return False
   
   # Test WVS
   if wvs:
       print("\n4.3. Testing WVS (one domain, limited questions)...")
       try:
           # Select a domain to test
           domain = list(wvs.domains.keys())[0]
           domain_name = wvs.get_domain_names().get(domain, domain)
           print(f"Evaluating domain: {domain_name}")
           
           # Evaluate with just one question to save time
           result = await pipeline.evaluate_wvs_domain(domain, max_questions=1)
           
           # Display results
           print("✓ Domain metrics:")
           metrics = result["metrics"]
           if metrics.get("avg_alignment") is not None:
               print(f"  - Average alignment: {metrics['avg_alignment']:.2f}")
           
           print(f"  - Acceptable range ratio: {metrics.get('acceptable_range_ratio', 0):.2f}")
           print(f"  - Average reasoning quality: {metrics.get('avg_reasoning_quality', 0):.2f}")
       except Exception as e:
           print(f"Error evaluating WVS: {e}")
           return False
   
   print("\n=== Pipeline test completed successfully ===")
   return True


if __name__ == "__main__":
   success = asyncio.run(test_full_pipeline())
   if not success:
       print("\nTest failed with errors.")
       exit(1)