# debug/debug_mfq.py
import json
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from morals.instruments.mfq import MoralFoundationsQuestionnaire

def main():
    # Create an absolute path using Path
    data_path = project_root / "data" / "instruments" / "mfq.json"
    
    print(f"=== MFQ Debugging Tool ===")
    print(f"Using project root: {project_root}")
    print(f"Looking for file at: {data_path}")
    
    # Check if file exists
    if data_path.exists():
        print(f"✓ File exists!")
    else:
        print(f"✗ File NOT found!")
        print(f"Current directory contents: {os.listdir(os.getcwd())}")
        data_dir = project_root / "data"
        if data_dir.exists():
            print(f"data/ directory contents: {os.listdir(data_dir)}")
            instruments_dir = data_dir / "instruments"
            if instruments_dir.exists():
                print(f"data/instruments/ directory contents: {os.listdir(instruments_dir)}")
        return
    
    # Try to load the JSON directly
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        print("\n✓ Successfully loaded raw JSON data")
        print(f"Keys in the data: {list(raw_data.keys())}")
        
        if "foundations" in raw_data:
            print(f"✓ Found 'foundations' key with foundations: {list(raw_data['foundations'].keys())}")
        else:
            print("✗ ERROR: No 'foundations' key in the JSON data!")
    except Exception as e:
        print(f"\n✗ Error loading JSON file: {str(e)}")
        return
    
    # Try to initialize the MFQ class
    print("\nAttempting to initialize MoralFoundationsQuestionnaire class...")
    try:
        mfq = MoralFoundationsQuestionnaire(data_path=str(data_path))
        print("✓ Successfully initialized MFQ class")
        print(f"✓ Has foundations attribute: {hasattr(mfq, 'foundations')}")
        if hasattr(mfq, 'foundations'):
            print(f"✓ Foundations keys: {list(mfq.foundations.keys())}")
            
            # Test if get_all_questions works
            questions = mfq.get_all_questions()
            print(f"✓ Found {len(questions)} questions across all foundations")
            
            # Try to get a specific question
            if questions:
                sample_id = questions[0]["id"]
                try:
                    question = mfq.get_question_by_id(sample_id)
                    print(f"\nSample question ({sample_id}):")
                    print(f"  Original: {question.get('original', 'N/A')}")
                    print(f"  Type: {question.get('type', 'N/A')}")
                    print(f"  Foundation: {question.get('foundation', 'N/A')}")
                    
                    gt = question.get("ground_truth", {})
                    print(f"  Ground truth mean: {gt.get('mean_score', 'N/A')}")
                    
                    # Test getting prompt
                    prompt = mfq.get_prompt_for_question(sample_id)
                    print(f"\nPrompt for {sample_id} (truncated):")
                    print(prompt[:100] + "..." if len(prompt) > 100 else prompt)
                    
                    print("\nAll tests passed successfully!")
                except Exception as e:
                    print(f"✗ Error accessing question details: {str(e)}")
            else:
                print("✗ No questions found in the MFQ data")
    except Exception as e:
        print(f"✗ Error initializing MFQ class: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()