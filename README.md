# MORALS: Moral Reasoning Assessment of Large Language Models

**MORALS** is an open-source framework for evaluating the moral reasoning capabilities of large language models (LLMs). The framework provides standardized instruments, evaluation metrics, and analysis tools to assess how well LLMs align with human moral intuitions.

## 🚀 Features

- **Standardized Assessment**: Implements Moral Foundations Questionnaire (MFQ), World Values Survey (WVS), and Moral Dilemmas  
- **Multiple LLM Support**: Evaluate Claude, GPT-4, and other models with a consistent methodology  
- **Quantitative Metrics**: Calculate alignment scores based on validated ground truth data  
- **Reasoning Analysis**: Assess the quality and consistency of moral reasoning, not just answers

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/morals.git
cd morals

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy the example config and edit it
cp config.yaml.example config.yaml
# Edit config.yaml with your preferred text editor

export ANTHROPIC_API_KEY="your_api_key_here"
export OPENAI_API_KEY="your_api_key_here"

# Run evaluation with Claude on 5 MFQ questions
python -m morals.cli.main --provider anthropic --limit 5

# Evaluate a specific moral foundation
python -m morals.cli.main --foundation care

# Use a different model (e.g., GPT-4)
python -m morals.cli.main --provider openai --model gpt-4

morals/
├── data/
│   └── instruments/
│       ├── mfq.json
│       ├── wvs.json
│       └── dilemmas.json
│
├── morals/
│   ├── instruments/
│   │   ├── base.py
│   │   ├── mfq.py
│   │   ├── wvs.py
│   │   └── dilemmas.py
│   │
│   ├── llm/
│   │   ├── base.py
│   │   ├── anthropic.py
│   │   ├── openai.py
│   │   └── factory.py
│   │
│   ├── evaluation/
│   │   ├── processor.py
│   │   ├── mfq_evaluator.py
│   │   └── metrics.py
│   │
│   ├── config.py
│   └── cli/
│       └── main.py
│
├── tests/
├── config.yaml.example
├── requirements.txt
└── README.md

import asyncio
from morals.instruments.mfq import MoralFoundationsQuestionnaire
from morals.llm.factory import LLMFactory
from morals.evaluation.mfq_evaluator import MFQEvaluator

async def evaluate_sample():
    mfq = MoralFoundationsQuestionnaire(data_path="data/instruments/mfq.json")
    llm = LLMFactory.create(provider="anthropic")
    evaluator = MFQEvaluator(mfq)

    question_id = "care_r1"
    prompt = mfq.get_prompt_for_question(question_id)
    response = await llm.generate_response(prompt)
    result = evaluator.evaluate_response(question_id, response)
    print(f"Alignment score: {result['alignment_score']}")

if __name__ == "__main__":
    asyncio.run(evaluate_sample())

@misc{morals2025,
  author = {Your Name},
  title = {MORALS: Moral Reasoning Assessment of Large Language Models},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/morals}
}



Let me know if you want a version with shields.io badges, a logo, or a prettier layout with HTML inside Markdown!

