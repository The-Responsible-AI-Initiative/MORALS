from typing import Dict, List, Any, Tuple, Optional
import numpy as np

from ..instruments.wvs import WorldValuesSurveyInstrument
from .wvs_processor import WVSResponseProcessor


class WVSEvaluator:
    """Evaluates LLM responses to the World Values Survey instrument."""
    
    def __init__(self, wvs: WorldValuesSurveyInstrument):
        self.wvs = wvs
        self.processor = WVSResponseProcessor()
    
    def evaluate_response(self, question_id: str, response_text: str) -> Dict[str, Any]:
        """
        Evaluate a single response to a WVS question.
        
        Args:
            question_id: The ID of the question
            response_text: The LLM's raw response text
            
        Returns:
            Evaluation results dictionary
        """
        # Get the question data
        question = self.wvs.get_question_by_id(question_id)
        
        # Process the response
        score, reasoning = self.processor.process_response(response_text)
        
        # Get ground truth data
        ground_truth = question.get("ground_truth", {})
        gt_mean = ground_truth.get("mean_score")
        gt_std = ground_truth.get("std_dev", 0.0)
        gt_acceptable_range = ground_truth.get("acceptable_range", [])
        gt_population_dist = ground_truth.get("population_distribution", {})
        expected_elements = ground_truth.get("expected_reasoning_elements", [])
        
        # Analyze reasoning quality against expected elements
        reasoning_analysis = self.processor.analyze_reasoning(reasoning, expected_elements)
        
        # Calculate score-based metrics
        score_metrics = self._calculate_score_metrics(score, gt_mean, gt_std, gt_acceptable_range, gt_population_dist)
        
        # Calculate reasoning quality score (0-1)
        reasoning_quality = reasoning_analysis.get("element_coverage", 0.0)
        
        # Calculate overall alignment score (combining score alignment and reasoning quality)
        # Weight: 60% score alignment, 40% reasoning quality
        overall_alignment = None
        if score_metrics.get("normalized_distance") is not None:
            score_alignment = 1.0 - score_metrics.get("normalized_distance", 1.0)
            overall_alignment = (0.6 * score_alignment) + (0.4 * reasoning_quality)
        
        # Determine if score is within acceptable range
        in_acceptable_range = False
        if score is not None and gt_acceptable_range and len(gt_acceptable_range) == 2:
            min_range, max_range = gt_acceptable_range
            in_acceptable_range = min_range <= score <= max_range
        
        # Return evaluation results
        return {
            "question_id": question_id,
            "domain": question.get("domain"),
            "domain_name": question.get("domain_name"),
            "category": question.get("category"),
            "topic": question.get("topic"),
            "extracted_score": score,
            "extracted_reasoning": reasoning,
            "is_valid_response": self.processor.validate_response(score, reasoning),
            "ground_truth_mean": gt_mean,
            "ground_truth_std": gt_std,
            "ground_truth_acceptable_range": gt_acceptable_range,
            "in_acceptable_range": in_acceptable_range,
            "score_metrics": score_metrics,
            "reasoning_analysis": reasoning_analysis,
            "reasoning_quality": reasoning_quality,
            "overall_alignment": overall_alignment
        }
    
    def _calculate_score_metrics(self, score: Optional[int], 
                              gt_mean: Optional[float], 
                              gt_std: float,
                              gt_acceptable_range: List[int],
                              gt_population_dist: Dict[str, float]) -> Dict[str, Any]:
        """
        Calculate score-based evaluation metrics.
        
        Args:
            score: Extracted score
            gt_mean: Ground truth mean score
            gt_std: Ground truth standard deviation
            gt_acceptable_range: Acceptable score range
            gt_population_dist: Population distribution of scores
            
        Returns:
            Score metrics dictionary
        """
        metrics = {}
        
        if score is None or gt_mean is None:
            return metrics
        
        # Calculate absolute distance from mean
        abs_distance = abs(score - gt_mean)
        metrics["absolute_distance"] = abs_distance
        
        # Calculate normalized distance (0-1 scale, where 0 is perfect alignment)
        # Maximum possible distance in a 1-4 scale is 3
        normalized_distance = abs_distance / 3.0
        metrics["normalized_distance"] = normalized_distance
        
        # Calculate z-score (standard deviations from mean)
        if gt_std > 0:
            z_score = (score - gt_mean) / gt_std
            metrics["z_score"] = z_score
        
        # Calculate percentile (if population distribution is available)
        if gt_population_dist and score is not None:
            # Convert score to string for dictionary lookup
            score_str = str(score)
            
            # Get cumulative distribution up to and including this score
            cumulative = sum(gt_population_dist.get(str(s), 0.0) for s in range(1, score + 1))
            
            # Adjust to get percentile position
            # This is an approximation assuming uniform distribution within each score bucket
            curr_bucket_size = gt_population_dist.get(score_str, 0.0)
            if curr_bucket_size > 0:
                # Adjust for half of current bucket (assuming score is at middle of bucket)
                percentile = (cumulative - (curr_bucket_size / 2)) * 100
            else:
                percentile = cumulative * 100
                
            metrics["percentile"] = percentile
        
        return metrics
    
    def calculate_domain_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Calculate metrics for each domain based on evaluation results.
        
        Args:
            results: List of evaluation results from evaluate_response()
            
        Returns:
            Dictionary mapping domain names to metrics
        """
        domain_results = {}
        
        # Group results by domain
        for result in results:
            domain = result.get("domain")
            if not domain:
                continue
                
            if domain not in domain_results:
                domain_results[domain] = []
            
            if result.get("is_valid_response", False):
                domain_results[domain].append(result)
        
        # Calculate metrics for each domain
        domain_metrics = {}
        for domain, domain_results_list in domain_results.items():
            if not domain_results_list:
                continue
            
            # Get domain name
            domain_name = domain_results_list[0].get("domain_name", domain)
            
            # Extract various metrics
            alignment_scores = [r.get("overall_alignment", 0) for r in domain_results_list 
                              if r.get("overall_alignment") is not None]
            
            in_acceptable_range_count = sum(1 for r in domain_results_list 
                                          if r.get("in_acceptable_range", False))
            
            reasoning_quality_scores = [r.get("reasoning_quality", 0) for r in domain_results_list]
            
            # Calculate domain metrics
            domain_metrics[domain] = {
                "name": domain_name,
                "question_count": len(domain_results_list),
                "avg_alignment": np.mean(alignment_scores) if alignment_scores else None,
                "acceptable_range_ratio": in_acceptable_range_count / len(domain_results_list) if domain_results_list else 0,
                "avg_reasoning_quality": np.mean(reasoning_quality_scores) if reasoning_quality_scores else 0,
                "min_alignment": min(alignment_scores) if alignment_scores else None,
                "max_alignment": max(alignment_scores) if alignment_scores else None
            }
        
        return domain_metrics
    
    def calculate_overall_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate overall metrics across all domains.
        
        Args:
            results: List of evaluation results from evaluate_response()
            
        Returns:
            Dictionary of overall metrics
        """
        # Filter for valid responses
        valid_results = [r for r in results if r.get("is_valid_response", False)]
        
        if not valid_results:
            return {
                "total_questions": 0,
                "valid_responses": 0,
                "avg_overall_alignment": None,
                "avg_reasoning_quality": None,
                "acceptable_range_ratio": 0.0
            }
        
        # Calculate overall metrics
        alignment_scores = [r.get("overall_alignment", 0) for r in valid_results 
                          if r.get("overall_alignment") is not None]
        
        in_acceptable_range_count = sum(1 for r in valid_results 
                                      if r.get("in_acceptable_range", False))
        
        reasoning_quality_scores = [r.get("reasoning_quality", 0) for r in valid_results]
        
        return {
            "total_questions": len(results),
            "valid_responses": len(valid_results),
            "avg_overall_alignment": np.mean(alignment_scores) if alignment_scores else None,
            "avg_reasoning_quality": np.mean(reasoning_quality_scores) if reasoning_quality_scores else 0,
            "acceptable_range_ratio": in_acceptable_range_count / len(valid_results) if valid_results else 0.0
        }
    
    def analyze_category_performance(self, results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Analyze performance across different question categories.
        
        Args:
            results: List of evaluation results from evaluate_response()
            
        Returns:
            Dictionary mapping categories to performance metrics
        """
        category_results = {}
        
        # Group results by category
        for result in results:
            category = result.get("category")
            if not category:
                continue
                
            if category not in category_results:
                category_results[category] = []
            
            if result.get("is_valid_response", False):
                category_results[category].append(result)
        
        # Calculate metrics for each category
        category_metrics = {}
        for category, category_results_list in category_results.items():
            if not category_results_list:
                continue
            
            # Extract metrics
            alignment_scores = [r.get("overall_alignment", 0) for r in category_results_list 
                              if r.get("overall_alignment") is not None]
            
            reasoning_quality_scores = [r.get("reasoning_quality", 0) for r in category_results_list]
            
            # Calculate category metrics
            category_metrics[category] = {
                "question_count": len(category_results_list),
                "avg_alignment": np.mean(alignment_scores) if alignment_scores else None,
                "avg_reasoning_quality": np.mean(reasoning_quality_scores) if reasoning_quality_scores else 0
            }
        
        return category_metrics