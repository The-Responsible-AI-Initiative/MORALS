# test_mfq.py
from morals.instruments.mfq import MoralFoundationsQuestionnaire

def main():
    # Initialize the MFQ instrument
    mfq = MoralFoundationsQuestionnaire(data_path="data/instruments/mfq.json")
    
    # Print some basic info
    print(f"MFQ loaded with {len(mfq.get_all_questions())} questions")
    print("Foundations:")
    for key, name in mfq.get_foundation_names().items():
        foundation_questions = mfq.get_questions_by_foundation(key)
        print(f"  - {name} ({key}): {len(foundation_questions)} questions")
    
    # Get and print a sample question
    sample_id = "care_r1"
    sample_question = mfq.get_question_by_id(sample_id)
    print(f"\nSample question ({sample_id}):")
    print(f"  Original: {sample_question['original']}")
    print(f"  Type: {sample_question['type']}")
    print(f"  Foundation: {sample_question['foundation']}")
    print(f"  Ground truth mean: {sample_question['ground_truth']['mean_score']}")
    
    # Get the prompt for the sample question
    prompt = mfq.get_prompt_for_question(sample_id)
    print(f"\nPrompt for {sample_id}:")
    print(prompt)

if __name__ == "__main__":
    main()