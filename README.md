# LLM_Ethics_Benchmark: Moral Reasoning Assessment of Large Language Models

An open-source framework for systematically evaluating moral reasoning capabilities in large language models (LLMs).
Overview

## Overview

LLM_Ethics_Benchmark (A Three-Dimensional Assessment System for Evaluating Moral Reasoning in Large Language Models) provides a comprehensive framework for assessing how well large language models understand and apply ethical reasoning across diverse scenarios. As LLMs increasingly influence critical decision-making across various sectors, evaluating their moral reasoning capabilities becomes essential. Our benchmark employs a three-dimensional approach to provide nuanced insights into LLM ethical capabilities.

## 🚀 Features

- **Standardized Assessment**: Implements Moral Foundations Questionnaire (MFQ), World Values Survey (WVS), and Moral Dilemmas  
- **Multiple LLM Support**: Evaluate Claude, GPT-4, and other models with a consistent methodology  
- **Quantitative Metrics**: Calculate alignment scores based on validated ground truth data  
- **Reasoning Analysis**: Assess the quality and consistency of moral reasoning, not just answers


## Installation

```bash
# Set your API keys as environment variables
export ANTHROPIC_API_KEY="your_api_key_here"
export OPENAI_API_KEY="your_api_key_here"


## 📊 Key Components

### 1. Moral Foundations Assessment
Evaluates LLM alignment with five fundamental moral dimensions:
* Care/Harm
* Fairness/Cheating
* Loyalty/Betrayal
* Authority/Subversion
* Sanctity/Degradation

### 2. Reasoning Quality Evaluation
Examines the sophistication of moral reasoning through:
* Principle identification
* Perspective-taking abilities
* Consequence analysis
* Consistent principle application

### 3. Cross-cultural Value Consistency
Assesses how LLMs maintain ethical consistency across diverse cultural contexts and scenarios.

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


## 📝 Citation

```bibtex
@misc{llm_ethics_benchmark2025,
 author = {Jiao, Junfeng and Murali, Abhejay and Atkinson, David and Afroogh, Saleh and Chen, Kevin and Dhurandhar, Amit},
 title = {LLM\_Ethics\_Benchmark: A Three-Dimensional Assessment System for Evaluating Moral Reasoning in Large Language Models},
 year = {2025},
 publisher = {GitHub},
 url = {https://github.com/yourusername/LLM_Ethics_Benchmark}
}




