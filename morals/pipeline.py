# morals/pipeline.py
import asyncio
from typing import Dict, List, Any, Optional
import json
from pathlib import Path

from .instruments.mfq import MoralFoundationsQuestionnaire
from .instruments.dilemmas import MoralDilemmasInstrument
from .instruments.wvs import WorldValuesSurveyInstrument
from .evaluation.mfq_evaluator import MFQEvaluator
from .evaluation.dilemmas_evaluator import DilemmasEvaluator
from .evaluation.wvs_evaluator import WVSEvaluator
from .llm.base import LLMInterface
from .llm.prompt_formatter import MFQPromptFormatter
from .llm.dilemmas_prompt_formatter import DilemmasPromptFormatter
from .llm.wvs_prompt_formatter import WVSPromptFormatter


class MoralEvaluationPipeline:
    """Pipeline for evaluating LLM responses to moral questions."""
    
    def __init__(self, 
                 llm: LLMInterface,
                 mfq: Optional[MoralFoundationsQuestionnaire] = None,
                 dilemmas: Optional[MoralDilemmasInstrument] = None,
                 wvs: Optional[WorldValuesSurveyInstrument] = None,
                 output_dir: Optional[str] = None):
        self.llm = llm
        self.mfq = mfq
        self.dilemmas = dilemmas
        self.wvs = wvs
        self.output_dir = output_dir
        
        # Initialize evaluators if instruments are provided
        self.mfq_evaluator = MFQEvaluator(mfq) if mfq else None
        self.dilemmas_evaluator = DilemmasEvaluator(dilemmas) if dilemmas else None
        self.wvs_evaluator = WVSEvaluator(wvs) if wvs else None
        
        # Create output directory if specified
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    #-------------------- MFQ Methods --------------------#
    
    async def evaluate_mfq_question(self, question_id: str) -> Dict[str, Any]:
        """
        Generate and evaluate a response to a single MFQ question.
        
        Args:
            question_id: The ID of the question to evaluate
            
        Returns:
            Evaluation results
        """
        if not self.mfq:
            raise ValueError("MFQ instrument not initialized")
        
        # Get the question
        question = self.mfq.get_question_by_id(question_id)
        
        # Format prompt
        prompt = MFQPromptFormatter.format_prompt(question)
        
        # Generate response
        response_text = await self.llm.generate_response(prompt)
        
        # Evaluate response
        result = self.mfq_evaluator.evaluate_response(question_id, response_text)
        
        # Add raw data for reference
        result["prompt"] = prompt
        result["raw_response"] = response_text
        
        # Save result if output directory is specified
        if self.output_dir:
            self._save_result("mfq", question_id, result)
        
        return result
    
    async def evaluate_mfq_foundation(self, 
                                   foundation: str, 
                                   max_questions: Optional[int] = None) -> Dict[str, Any]:
        """
        Evaluate an LLM on a specific moral foundation from MFQ.
        
        Args:
            foundation: The moral foundation to evaluate
            max_questions: Maximum number of questions to evaluate (None for all)
            
        Returns:
            Evaluation results for the foundation
        """
        if not self.mfq:
            raise ValueError("MFQ instrument not initialized")
        
        # Get questions for foundation
        questions = self.mfq.get_questions_by_foundation(foundation)
        
        # Limit number of questions if specified
        if max_questions is not None:
            questions = questions[:max_questions]
        
        # Evaluate each question
        results = []
        for question in questions:
            result = await self.evaluate_mfq_question(question["id"])
            results.append(result)
        
        # Calculate foundation alignment
        foundation_alignment = self.mfq_evaluator.calculate_foundation_alignment(results)
        
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
            self._save_result("mfq", f"foundation_{foundation}", evaluation_result)
        
        return evaluation_result
    
    async def evaluate_all_mfq_foundations(self, 
                                       max_questions_per_foundation: Optional[int] = None) -> Dict[str, Any]:
        """
        Evaluate an LLM on all moral foundations from MFQ.
        
        Args:
            max_questions_per_foundation: Maximum questions per foundation (None for all)
            
        Returns:
            Complete evaluation results
        """
        if not self.mfq:
            raise ValueError("MFQ instrument not initialized")
        
        foundations = self.mfq.get_foundation_names().keys()
        
        foundation_results = {}
        all_question_results = []
        
        for foundation in foundations:
            result = await self.evaluate_mfq_foundation(foundation, max_questions_per_foundation)
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
            "instrument": "mfq",
            "model": self.llm.model_info,
            "overall_alignment": overall_alignment,
            "foundation_results": foundation_results,
            "question_results": all_question_results
        }
        
        # Save results if output directory is specified
        if self.output_dir:
            self._save_result("mfq", "overall", evaluation_result)
        
        return evaluation_result
    
    #-------------------- Dilemmas Methods --------------------#
    
    async def evaluate_dilemma_question(self, dilemma_id: str, question_id: str) -> Dict[str, Any]:
        """
        Generate and evaluate a response to a single moral dilemma question.
        
        Args:
            dilemma_id: The ID of the dilemma
            question_id: The ID of the question
            
        Returns:
            Evaluation results
        """
        if not self.dilemmas:
            raise ValueError("Dilemmas instrument not initialized")
        
        # Format the combined ID
        combined_id = self.dilemmas.get_formatted_id(dilemma_id, question_id)
        
        # Get the dilemma questions
        dilemma_questions = self.dilemmas.get_questions_by_dilemma(dilemma_id)
        
        # Find the specific question
        question = next((q for q in dilemma_questions if q["id"] == question_id), None)
        if not question:
            raise ValueError(f"Question {question_id} not found in dilemma {dilemma_id}")
        
        # Format prompt
        prompt = DilemmasPromptFormatter.format_prompt(question)
        
        # Generate response
        response_text = await self.llm.generate_response(prompt, max_tokens=1500)
        
        # Evaluate response
        result = self.dilemmas_evaluator.evaluate_response(combined_id, response_text)
        
        # Add raw data for reference
        result["prompt"] = prompt
        result["raw_response"] = response_text
        
        # Save result if output directory is specified
        if self.output_dir:
            self._save_result("dilemmas", combined_id, result)
        
        return result
    
    async def evaluate_dilemma(self, dilemma_id: str, max_questions: Optional[int] = None) -> Dict[str, Any]:
        """
        Evaluate an LLM on a specific moral dilemma.
        
        Args:
            dilemma_id: The ID of the dilemma to evaluate
            max_questions: Maximum number of questions to evaluate (None for all)
            
        Returns:
            Evaluation results for the dilemma
        """
        if not self.dilemmas:
            raise ValueError("Dilemmas instrument not initialized")
        
        # Get the dilemma
        dilemma = self.dilemmas.get_dilemma_by_id(dilemma_id)
        
        # Get questions for the dilemma
        questions = dilemma.get("questions", [])
        
        # Limit number of questions if specified
        if max_questions is not None:
            questions = questions[:max_questions]
        
        # Evaluate each question
        results = []
        for question in questions:
            question_id = question["id"]
            result = await self.evaluate_dilemma_question(dilemma_id, question_id)
            results.append(result)
        
        # Calculate dilemma scores
        dilemma_scores = self.dilemmas_evaluator.calculate_dilemma_scores({dilemma_id: results})
        
        # Compile results
        evaluation_result = {
            "dilemma_id": dilemma_id,
            "dilemma_title": dilemma.get("title"),
            "model": self.llm.model_info,
            "scores": dilemma_scores.get(dilemma_id, {}),
            "question_results": results
        }
        
        # Save results if output directory is specified
        if self.output_dir:
            self._save_result("dilemmas", f"dilemma_{dilemma_id}", evaluation_result)
        
        return evaluation_result
    
    async def evaluate_all_dilemmas(self, max_questions_per_dilemma: Optional[int] = None) -> Dict[str, Any]:
        """
        Evaluate an LLM on all moral dilemmas.
        
        Args:
            max_questions_per_dilemma: Maximum questions per dilemma (None for all)
            
        Returns:
            Complete evaluation results
        """
        if not self.dilemmas:
            raise ValueError("Dilemmas instrument not initialized")
        
        # Get all dilemmas
        dilemma_ids = [dilemma.get("id") for dilemma in self.dilemmas.dilemmas]
        
        dilemma_results = {}
        all_question_results = []
        
        # Evaluate each dilemma
        for dilemma_id in dilemma_ids:
            result = await self.evaluate_dilemma(dilemma_id, max_questions_per_dilemma)
            dilemma_results[dilemma_id] = {
                "title": result["dilemma_title"],
                "scores": result["scores"]
            }
            all_question_results.extend(result["question_results"])
        
        # Calculate aggregate scores
        aggregate_scores = self.dilemmas_evaluator.calculate_aggregate_scores(all_question_results)
        
        # Compile results
        evaluation_result = {
            "instrument": "dilemmas",
            "model": self.llm.model_info,
            "aggregate_scores": aggregate_scores,
            "dilemma_results": dilemma_results,
            "question_results": all_question_results
        }
        
        # Save results if output directory is specified
        if self.output_dir:
            self._save_result("dilemmas", "overall", evaluation_result)
        
        return evaluation_result
    
    #-------------------- WVS Methods --------------------#
    
    async def evaluate_wvs_question(self, question_id: str) -> Dict[str, Any]:
        """
        Generate and evaluate a response to a single WVS question.
        
        Args:
            question_id: The ID of the question to evaluate
            
        Returns:
            Evaluation results
        """
        if not self.wvs:
            raise ValueError("WVS instrument not initialized")
        
        # Get the question
        question = self.wvs.get_question_by_id(question_id)
        
        # Format prompt
        prompt = WVSPromptFormatter.format_prompt(question)
        
        # Generate response
        response_text = await self.llm.generate_response(prompt)
        
        # Evaluate response
        result = self.wvs_evaluator.evaluate_response(question_id, response_text)
        
        # Add raw data for reference
        result["prompt"] = prompt
        result["raw_response"] = response_text
        
        # Save result if output directory is specified
        if self.output_dir:
            self._save_result("wvs", question_id, result)
        
        return result
    
    async def evaluate_wvs_domain(self, 
                               domain: str, 
                               max_questions: Optional[int] = None) -> Dict[str, Any]:
        """
        Evaluate an LLM on a specific WVS domain.
        
        Args:
            domain: The domain to evaluate
            max_questions: Maximum number of questions to evaluate (None for all)
            
        Returns:
            Evaluation results for the domain
        """
        if not self.wvs:
            raise ValueError("WVS instrument not initialized")
        
        # Get questions for domain
        questions = self.wvs.get_questions_by_domain(domain)
        
        # Limit number of questions if specified
        if max_questions is not None:
            questions = questions[:max_questions]
        
        # Evaluate each question
        results = []
        for question in questions:
            result = await self.evaluate_wvs_question(question["id"])
            results.append(result)
        
        # Calculate domain metrics
        domain_metrics = self.wvs_evaluator.calculate_domain_metrics(results).get(domain, {})
        
        # Compile results
        evaluation_result = {
            "domain": domain,
            "domain_name": domain_metrics.get("name", domain),
            "model": self.llm.model_info,
            "metrics": domain_metrics,
            "question_results": results
        }
        
        # Save results if output directory is specified
        if self.output_dir:
            self._save_result("wvs", f"domain_{domain}", evaluation_result)
        
        return evaluation_result
    
    async def evaluate_wvs_category(self, 
                                 category: str, 
                                 max_questions: Optional[int] = None) -> Dict[str, Any]:
        """
        Evaluate an LLM on a specific WVS category (e.g., importance, agreement).
        
        Args:
            category: The category to evaluate
            max_questions: Maximum number of questions to evaluate (None for all)
            
        Returns:
            Evaluation results for the category
        """
        if not self.wvs:
            raise ValueError("WVS instrument not initialized")
        
        # Get questions for category
        questions = self.wvs.get_questions_by_category(category)
        
        # Limit number of questions if specified
        if max_questions is not None:
            questions = questions[:max_questions]
        
        # Evaluate each question
        results = []
        for question in questions:
            result = await self.evaluate_wvs_question(question["id"])
            results.append(result)
        
        # Calculate category metrics
        category_metrics = self.wvs_evaluator.analyze_category_performance(results).get(category, {})
        
        # Compile results
        evaluation_result = {
            "category": category,
            "model": self.llm.model_info,
            "metrics": category_metrics,
            "question_results": results
        }
        
        # Save results if output directory is specified
        if self.output_dir:
            self._save_result("wvs", f"category_{category}", evaluation_result)
        
        return evaluation_result
    
    async def evaluate_all_wvs_domains(self, 
                                    max_questions_per_domain: Optional[int] = None) -> Dict[str, Any]:
        """
        Evaluate an LLM on all WVS domains.
        
        Args:
            max_questions_per_domain: Maximum questions per domain (None for all)
            
        Returns:
            Complete evaluation results
        """
        if not self.wvs:
            raise ValueError("WVS instrument not initialized")
        
        # Get all domains
        domains = self.wvs.get_domain_names().keys()
        
        domain_results = {}
        all_question_results = []
        
        # Evaluate each domain
        for domain in domains:
            result = await self.evaluate_wvs_domain(domain, max_questions_per_domain)
            domain_results[domain] = {
                "name": result["domain_name"],
                "metrics": result["metrics"]
            }
            all_question_results.extend(result["question_results"])
        
        # Calculate overall metrics
        overall_metrics = self.wvs_evaluator.calculate_overall_metrics(all_question_results)
        
        # Calculate category performance
        category_performance = self.wvs_evaluator.analyze_category_performance(all_question_results)
        
        # Compile results
        evaluation_result = {
            "instrument": "wvs",
            "model": self.llm.model_info,
            "overall_metrics": overall_metrics,
            "domain_results": domain_results,
            "category_performance": category_performance,
            "question_results": all_question_results
        }
        
        # Save results if output directory is specified
        if self.output_dir:
            self._save_result("wvs", "overall", evaluation_result)
        
        return evaluation_result
    
    #-------------------- Helper Methods --------------------#
    
    def _save_result(self, instrument: str, result_id: str, result: Dict[str, Any]) -> None:
        """Save a result to a file."""
        model_name = self.llm.model_info["name"].replace("/", "_")
        file_path = Path(self.output_dir) / instrument / f"{result_id}_{model_name}.json"
        
        # Create subdirectory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(result, f, indent=2)