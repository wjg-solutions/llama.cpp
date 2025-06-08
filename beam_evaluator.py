#!/usr/bin/env python3
"""
Beam Search Result Evaluator

This script evaluates beam search results from llama.cpp server and selects the best candidate
based on the original prompt and multiple evaluation criteria.
"""

import json
import re
import argparse
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from enum import Enum

class PromptType(Enum):
    CREATIVE = "creative"
    FACTUAL = "factual"
    CONVERSATIONAL = "conversational"
    TECHNICAL = "technical"
    QUESTION_ANSWER = "question_answer"

@dataclass
class BeamResult:
    rank: int
    text: str
    log_probability: float
    normalized_score: float
    length_normalized_log_probability: float
    generation_length: int
    is_finished: bool
    tokens: List[int]

@dataclass
class EvaluationScore:
    relevance_score: float
    quality_score: float
    completeness_score: float
    technical_score: float
    overall_score: float
    strengths: List[str]
    weaknesses: List[str]

class BeamEvaluator:
    def __init__(self):
        self.weights = {
            'relevance': 0.40,
            'quality': 0.30,
            'completeness': 0.20,
            'technical': 0.10
        }
    
    def classify_prompt_type(self, prompt: str) -> PromptType:
        """Classify the prompt type to adjust evaluation criteria."""
        prompt_lower = prompt.lower()
        
        # Question patterns
        if any(word in prompt_lower for word in ['what', 'how', 'why', 'when', 'where', 'who', '?']):
            if any(word in prompt_lower for word in ['explain', 'describe', 'define']):
                return PromptType.FACTUAL
            return PromptType.QUESTION_ANSWER
        
        # Creative patterns
        if any(word in prompt_lower for word in ['write', 'story', 'poem', 'creative', 'imagine', 'pretend']):
            return PromptType.CREATIVE
        
        # Technical patterns
        if any(word in prompt_lower for word in ['code', 'program', 'algorithm', 'function', 'debug', 'implement']):
            return PromptType.TECHNICAL
        
        # Conversational patterns
        if any(word in prompt_lower for word in ['hello', 'hi', 'chat', 'talk', 'discuss']):
            return PromptType.CONVERSATIONAL
        
        # Default to factual
        return PromptType.FACTUAL
    
    def evaluate_relevance(self, prompt: str, beam_text: str, prompt_type: PromptType) -> Tuple[float, List[str], List[str]]:
        """Evaluate how well the beam result addresses the original prompt."""
        strengths = []
        weaknesses = []
        score = 5.0  # Base score
        
        # Check if the response addresses the prompt
        prompt_keywords = set(re.findall(r'\b\w+\b', prompt.lower()))
        response_keywords = set(re.findall(r'\b\w+\b', beam_text.lower()))
        keyword_overlap = len(prompt_keywords.intersection(response_keywords)) / max(len(prompt_keywords), 1)
        
        if keyword_overlap > 0.3:
            score += 2.0
            strengths.append("Good keyword overlap with prompt")
        elif keyword_overlap < 0.1:
            score -= 2.0
            weaknesses.append("Poor keyword overlap with prompt")
        
        # Check for direct addressing of questions
        if prompt_type == PromptType.QUESTION_ANSWER:
            if any(word in beam_text.lower() for word in ['yes', 'no', 'because', 'the answer', 'is that']):
                score += 1.0
                strengths.append("Directly addresses the question")
        
        # Check coherence (simple heuristic)
        sentences = beam_text.split('.')
        if len(sentences) > 1:
            # Check if sentences flow logically (very basic check)
            if not any(sent.strip() == '' for sent in sentences[:-1]):
                score += 1.0
                strengths.append("Good sentence structure")
        
        # Penalize off-topic responses
        if len(beam_text.strip()) > 0 and keyword_overlap < 0.05:
            score -= 3.0
            weaknesses.append("Response appears off-topic")
        
        return min(max(score, 1.0), 10.0), strengths, weaknesses
    
    def evaluate_quality(self, beam_text: str, prompt_type: PromptType) -> Tuple[float, List[str], List[str]]:
        """Evaluate the quality and accuracy of the response."""
        strengths = []
        weaknesses = []
        score = 5.0  # Base score
        
        # Check for basic quality indicators
        if len(beam_text.strip()) == 0:
            return 1.0, [], ["Empty response"]
        
        # Grammar and structure (basic checks)
        if beam_text[0].isupper():
            score += 0.5
            strengths.append("Proper capitalization")
        
        # Check for complete sentences
        sentences = [s.strip() for s in beam_text.split('.') if s.strip()]
        complete_sentences = sum(1 for s in sentences if len(s) > 3 and s[0].isupper())
        if complete_sentences > 0:
            score += 1.0
            strengths.append("Contains complete sentences")
        
        # Check for repetition (quality issue)
        words = beam_text.lower().split()
        if len(words) > 10:
            unique_words = len(set(words))
            repetition_ratio = unique_words / len(words)
            if repetition_ratio < 0.5:
                score -= 2.0
                weaknesses.append("High repetition detected")
            elif repetition_ratio > 0.8:
                score += 1.0
                strengths.append("Good vocabulary diversity")
        
        # Technical content quality
        if prompt_type == PromptType.TECHNICAL:
            if any(word in beam_text.lower() for word in ['function', 'variable', 'algorithm', 'code', 'implementation']):
                score += 1.0
                strengths.append("Contains relevant technical terms")
        
        # Creative content quality
        if prompt_type == PromptType.CREATIVE:
            if len(beam_text) > 50:  # Sufficient length for creativity
                score += 1.0
                strengths.append("Adequate length for creative content")
        
        return min(max(score, 1.0), 10.0), strengths, weaknesses
    
    def evaluate_completeness(self, prompt: str, beam_text: str, is_finished: bool) -> Tuple[float, List[str], List[str]]:
        """Evaluate how complete the response is."""
        strengths = []
        weaknesses = []
        score = 5.0  # Base score
        
        # Finished responses are generally better
        if is_finished:
            score += 2.0
            strengths.append("Response reached natural completion")
        else:
            score -= 1.0
            weaknesses.append("Response appears incomplete")
        
        # Check if response ends abruptly
        if not is_finished and not beam_text.rstrip().endswith(('.', '!', '?', ':', ';')):
            score -= 1.5
            weaknesses.append("Response ends abruptly")
        
        # Length appropriateness (heuristic based on prompt)
        prompt_length = len(prompt.split())
        response_length = len(beam_text.split())
        
        if prompt_length < 10 and response_length > 100:
            score += 1.0
            strengths.append("Detailed response to simple prompt")
        elif prompt_length > 20 and response_length < 20:
            score -= 1.0
            weaknesses.append("Brief response to complex prompt")
        
        # Check for conclusion indicators
        if any(phrase in beam_text.lower() for phrase in ['in conclusion', 'to summarize', 'finally', 'in summary']):
            score += 1.0
            strengths.append("Contains conclusion indicators")
        
        return min(max(score, 1.0), 10.0), strengths, weaknesses
    
    def evaluate_technical_metrics(self, beam: BeamResult) -> Tuple[float, List[str], List[str]]:
        """Evaluate based on technical metrics from beam search."""
        strengths = []
        weaknesses = []
        score = 5.0  # Base score
        
        # Normalize scores to 1-10 range
        # Higher normalized_score is better
        if beam.normalized_score > -2.0:
            score += 2.0
            strengths.append("High model confidence")
        elif beam.normalized_score < -5.0:
            score -= 2.0
            weaknesses.append("Low model confidence")
        
        # Length considerations
        if beam.generation_length > 5:
            score += 1.0
            strengths.append("Adequate generation length")
        elif beam.generation_length < 3:
            score -= 1.0
            weaknesses.append("Very short generation")
        
        # Rank consideration (lower rank number is better)
        if beam.rank == 1:
            score += 1.0
            strengths.append("Top-ranked by model")
        elif beam.rank > 3:
            score -= 0.5
            weaknesses.append("Lower model ranking")
        
        return min(max(score, 1.0), 10.0), strengths, weaknesses
    
    def evaluate_beam(self, prompt: str, beam: BeamResult, prompt_type: PromptType) -> EvaluationScore:
        """Evaluate a single beam result."""
        relevance_score, rel_strengths, rel_weaknesses = self.evaluate_relevance(prompt, beam.text, prompt_type)
        quality_score, qual_strengths, qual_weaknesses = self.evaluate_quality(beam.text, prompt_type)
        completeness_score, comp_strengths, comp_weaknesses = self.evaluate_completeness(prompt, beam.text, beam.is_finished)
        technical_score, tech_strengths, tech_weaknesses = self.evaluate_technical_metrics(beam)
        
        # Calculate weighted overall score
        overall_score = (
            relevance_score * self.weights['relevance'] +
            quality_score * self.weights['quality'] +
            completeness_score * self.weights['completeness'] +
            technical_score * self.weights['technical']
        )
        
        all_strengths = rel_strengths + qual_strengths + comp_strengths + tech_strengths
        all_weaknesses = rel_weaknesses + qual_weaknesses + comp_weaknesses + tech_weaknesses
        
        return EvaluationScore(
            relevance_score=relevance_score,
            quality_score=quality_score,
            completeness_score=completeness_score,
            technical_score=technical_score,
            overall_score=overall_score,
            strengths=all_strengths,
            weaknesses=all_weaknesses
        )
    
    def select_best_beam(self, prompt: str, beam_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select the best beam result based on comprehensive evaluation."""
        if not beam_results:
            return {"error": "No beam results provided"}
        
        # Convert to BeamResult objects
        beams = []
        for i, result in enumerate(beam_results):
            beam = BeamResult(
                rank=result.get('rank', i + 1),
                text=result.get('text', ''),
                log_probability=result.get('log_probability', 0.0),
                normalized_score=result.get('normalized_score', 0.0),
                length_normalized_log_probability=result.get('length_normalized_log_probability', 0.0),
                generation_length=result.get('generation_length', 0),
                is_finished=result.get('is_finished', False),
                tokens=result.get('tokens', [])
            )
            beams.append(beam)
        
        # Classify prompt type
        prompt_type = self.classify_prompt_type(prompt)
        
        # Evaluate each beam
        evaluations = {}
        best_beam = None
        best_score = -1
        
        for beam in beams:
            evaluation = self.evaluate_beam(prompt, beam, prompt_type)
            evaluations[f"beam_{beam.rank}"] = {
                "relevance_score": evaluation.relevance_score,
                "quality_score": evaluation.quality_score,
                "completeness_score": evaluation.completeness_score,
                "technical_score": evaluation.technical_score,
                "overall_score": evaluation.overall_score,
                "strengths": evaluation.strengths,
                "weaknesses": evaluation.weaknesses
            }
            
            if evaluation.overall_score > best_score:
                best_score = evaluation.overall_score
                best_beam = beam
        
        # Determine confidence level
        scores = [eval_data["overall_score"] for eval_data in evaluations.values()]
        score_range = max(scores) - min(scores)
        if score_range < 1.0:
            confidence = "low"
        elif score_range < 2.0:
            confidence = "medium"
        else:
            confidence = "high"
        
        # Handle case where no beam was selected (shouldn't happen with valid input)
        if best_beam is None:
            return {"error": "No valid beam could be selected"}
        
        # Generate reasoning
        best_eval = evaluations[f"beam_{best_beam.rank}"]
        reasoning = f"Selected beam {best_beam.rank} with overall score {best_eval['overall_score']:.2f}. "
        if best_eval['strengths']:
            reasoning += f"Key strengths: {', '.join(best_eval['strengths'][:3])}. "
        if best_beam.rank != 1:
            reasoning += f"Despite being ranked #{best_beam.rank} by the model, this beam better fulfills the evaluation criteria. "
        
        return {
            "selected_beam": {
                "rank": best_beam.rank,
                "text": best_beam.text,
                "reasoning": reasoning
            },
            "evaluation_summary": evaluations,
            "comparison_notes": f"Evaluated {len(beams)} beams using {prompt_type.value} prompt classification. Score range: {score_range:.2f}",
            "confidence_level": confidence,
            "prompt_type": prompt_type.value
        }

def main():
    parser = argparse.ArgumentParser(description="Evaluate beam search results")
    parser.add_argument("--input", "-i", required=True, help="Input JSON file with prompt and beam results")
    parser.add_argument("--output", "-o", help="Output JSON file for evaluation results")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Load input data
    with open(args.input, 'r') as f:
        data = json.load(f)
    
    prompt = data.get('prompt', '')
    beam_results = data.get('beam_results', [])
    
    if not prompt:
        print("Error: No prompt found in input data")
        return 1
    
    if not beam_results:
        print("Error: No beam results found in input data")
        return 1
    
    # Evaluate beams
    evaluator = BeamEvaluator()
    result = evaluator.select_best_beam(prompt, beam_results)
    
    # Output results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Evaluation results written to {args.output}")
    else:
        print(json.dumps(result, indent=2))
    
    if args.verbose:
        print(f"\nSelected beam {result['selected_beam']['rank']}: {result['selected_beam']['text'][:100]}...")
        print(f"Reasoning: {result['selected_beam']['reasoning']}")
    
    return 0

if __name__ == "__main__":
    exit(main())