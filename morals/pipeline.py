# morals/pipeline.py
import asyncio
from typing import Dict, List, Any, Optional
import json
from pathlib import Path

from .instruments.mfq import MoralFoundationsQuestionnaire
from .evaluation.mfq_evaluator import MFQEvaluator
from .llm.base import LLMInterface
from .llm.prompt_formatter import MFQPromptFormatter


class MoralFoundationsPipeline:
    """Pipeline for evaluating LLM responses to the MFQ."""
    
    def __init__(self, 
                 mfq: MoralFoundationsQuestionnaire,
                 llm: LLMInterface,
                 output_dir: Optional[str] = None):
        self.mfq = mfq
        self.llm = llm
        self.evaluator = MFQEvaluator(mfq)
        self.output_dir = output_dir
        
        # Create output directory if specified
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    async def evaluate_question(self, question_id: str) -> Dict[str, Any]:
        """
        Generate and evaluate a response to a single MFQ question.
        
        Args:
            question_id: The ID of the question to evaluate
            
        Returns:
            Evaluation results
        """
        # Get the question
        question = self.mfq.get_question_by_id(question_id)
        
        # Format prompt
        prompt = MFQPromptFormatter.format_prompt(question)
        
        # Generate response
        response_text = await self.llm.generate_response(prompt)
        
        # Evaluate response
        result = self.evaluator.evaluate_response(question_id, response_text)
        
        # Add raw data for reference
        result["prompt"] = prompt
        result["raw_response"] = response_text
        
        # Save result if output directory is specified
        if self.output_dir:
            self._save_question_result(question_id, result)
        
        return result
    
    async def evaluate_foundation(self, 
                                  foundation: str, 
                                  max_questions: Optional[int] = None) -> Dict[str, Any]:
        """
        Evaluate an LLM on a specific moral foundation.
        
        Args:
            foundation: The moral foundation to evaluate
            max_questions: Maximum number of questions to evaluate (None for all)
            
        Returns:
            Evaluation results for the foundation
        """
        # Get questions for foundation
        questions = self.mfq.get_questions_by_foundation(foundation)
        
        # Limit number of questions if specified
        if max_questions is not None:
            questions = questions[:max_questions]
        
        # Evaluate each question
        results = []
        for question in questions:
            result = await self.evaluate_question(question["id"])
            results.append(result)
        
        # Calculate foundation alignment
        foundation_alignment = self.evaluator.calculate_foundation_alignment(results)
        
        # Compile results
        evaluation_result = {
            "foundation": foundation,
            "foundation_name": self.mfq.get_foundation_names().get(foundation, foundation),
            "model": self.llm.model_info,
            "alignment_score": foundation_alignment.get(foundation),
            "question_results": results
        }
        
        # Save results if output directory is specified
        if self.output_dir:
            self._save_foundation_result(foundation, evaluation_result)
        
        return evaluation_result
    
    async def evaluate_all_foundations(self, 
                                      max_questions_per_foundation: Optional[int] = None) -> Dict[str, Any]:
        """
        Evaluate an LLM on all moral foundations.
        
        Args:
            max_questions_per_foundation: Maximum questions per foundation (None for all)
            
        Returns:
            Complete evaluation results
        """
        foundations = self.mfq.get_foundation_names().keys()
        
        foundation_results = {}
        all_question_results = []
        
        for foundation in foundations:
            result = await self.evaluate_foundation(foundation, max_questions_per_foundation)
            foundation_results[foundation] = {
                "alignment_score": result["alignment_score"],
                "foundation_name": result["foundation_name"]
            }
            all_question_results.extend(result["question_results"])
        
        # Calculate overall alignment
        valid_scores = [f["alignment_score"] for f in foundation_results.values() 
                       if f["alignment_score"] is not None]
        overall_alignment = sum(valid_scores) / len(valid_scores) if valid_scores else None
        
        # Compile results
        evaluation_result = {
            "model": self.llm.model_info,
            "overall_alignment": overall_alignment,
            "foundation_results": foundation_results,
            "question_results": all_question_results
        }
        
        # Save results if output directory is specified
        if self.output_dir:
            self._save_overall_result(evaluation_result)
        
        return evaluation_result
    
    def _save_question_result(self, question_id: str, result: Dict[str, Any]) -> None:
        """Save a question evaluation result to a file."""
        file_path = Path(self.output_dir) / f"question_{question_id}.json"
        with open(file_path, 'w') as f:
            json.dump(result, f, indent=2)
    
    def _save_foundation_result(self, foundation: str, result: Dict[str, Any]) -> None:
        """Save a foundation evaluation result to a file."""
        file_path = Path(self.output_dir) / f"foundation_{foundation}.json"
        with open(file_path, 'w') as f:
            json.dump(result, f, indent=2)
    
    def _save_overall_result(self, result: Dict[str, Any]) -> None:
        """Save the overall evaluation result to a file."""
        model_name = result["model"]["name"].replace("/", "_")
        file_path = Path(self.output_dir) / f"evaluation_{model_name}.json"
        with open(file_path, 'w') as f:
            json.dump(result, f, indent=2)