#!/usr/bin/env python3
"""
Test script for beam search evaluation system

This script provides example data and demonstrates the beam evaluation functionality
without requiring a running llama.cpp server.
"""

import json
from beam_evaluator import BeamEvaluator

def create_sample_beam_results():
    """Create sample beam results for testing."""
    return [
        {
            "rank": 1,
            "text": "Photosynthesis is the process by which plants convert sunlight into energy. Plants use chlorophyll in their leaves to capture light energy, which is then used to convert carbon dioxide from the air and water from the soil into glucose (sugar) and oxygen. This process is essential for life on Earth as it produces the oxygen we breathe and forms the base of most food chains.",
            "log_probability": -45.23,
            "normalized_score": -2.15,
            "length_normalized_log_probability": -0.89,
            "generation_length": 51,
            "is_finished": True,
            "tokens": [123, 456, 789, 101, 112]
        },
        {
            "rank": 2,
            "text": "Photosynthesis happens when plants make food from sunlight. The green stuff in leaves called chlorophyll catches the sun's energy. Then the plant takes in carbon dioxide through tiny holes in the leaves and water through the roots. Using the sun's energy, it mixes these together to make sugar for food and releases oxygen as a bonus that we can breathe.",
            "log_probability": -48.67,
            "normalized_score": -2.31,
            "length_normalized_log_probability": -0.95,
            "generation_length": 49,
            "is_finished": True,
            "tokens": [234, 567, 890, 202, 223]
        },
        {
            "rank": 3,
            "text": "In photosynthesis, light-dependent reactions occur in the thylakoids where photosystems I and II capture photons, leading to the splitting of water molecules and the generation of ATP and NADPH. The Calvin cycle then uses these energy carriers in the stroma to fix carbon dioxide into organic compounds through a series of enzymatic reactions involving RuBisCO.",
            "log_probability": -52.14,
            "normalized_score": -2.89,
            "length_normalized_log_probability": -1.12,
            "generation_length": 47,
            "is_finished": True,
            "tokens": [345, 678, 901, 303, 334]
        },
        {
            "rank": 4,
            "text": "Plants do photosynthesis to make food. They use sunlight and",
            "log_probability": -25.45,
            "normalized_score": -3.45,
            "length_normalized_log_probability": -2.31,
            "generation_length": 11,
            "is_finished": False,
            "tokens": [456, 789, 12, 404]
        }
    ]

def test_different_prompts():
    """Test the evaluator with different types of prompts."""
    evaluator = BeamEvaluator()
    
    test_cases = [
        {
            "name": "Simple Explanation Request",
            "prompt": "Explain how photosynthesis works in simple terms.",
            "expected_winner": 2,  # The simple explanation should win
            "beam_results": create_sample_beam_results()
        },
        {
            "name": "Technical Question",
            "prompt": "Describe the molecular mechanisms of photosynthesis.",
            "expected_winner": 3,  # The technical explanation should win
            "beam_results": create_sample_beam_results()
        },
        {
            "name": "General Question",
            "prompt": "What is photosynthesis?",
            "expected_winner": 1,  # The balanced explanation should win
            "beam_results": create_sample_beam_results()
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        print(f"\n{'='*60}")
        print(f"Testing: {test_case['name']}")
        print(f"Prompt: {test_case['prompt']}")
        print(f"{'='*60}")
        
        evaluation = evaluator.select_best_beam(
            test_case['prompt'], 
            test_case['beam_results']
        )
        
        selected_rank = evaluation['selected_beam']['rank']
        expected_rank = test_case['expected_winner']
        
        print(f"Selected beam: {selected_rank}")
        print(f"Expected beam: {expected_rank}")
        print(f"Match: {'✓' if selected_rank == expected_rank else '✗'}")
        print(f"Reasoning: {evaluation['selected_beam']['reasoning']}")
        print(f"Confidence: {evaluation['confidence_level']}")
        print(f"Prompt type: {evaluation['prompt_type']}")
        
        # Show evaluation scores
        print(f"\nEvaluation Scores:")
        for beam_key, scores in evaluation['evaluation_summary'].items():
            rank = beam_key.split('_')[1]
            print(f"  Beam {rank}: Overall={scores['overall_score']:.2f} "
                  f"(R:{scores['relevance_score']:.1f}, "
                  f"Q:{scores['quality_score']:.1f}, "
                  f"C:{scores['completeness_score']:.1f}, "
                  f"T:{scores['technical_score']:.1f})")
        
        results.append({
            'test_case': test_case['name'],
            'selected': selected_rank,
            'expected': expected_rank,
            'match': selected_rank == expected_rank,
            'evaluation': evaluation
        })
    
    return results

def test_creative_prompt():
    """Test with a creative writing prompt."""
    creative_beams = [
        {
            "rank": 1,
            "text": "The old lighthouse stood sentinel against the storm, its beacon cutting through the darkness like a sword of light. Captain Sarah gripped the wheel tighter as waves crashed over the bow of her small fishing vessel. She had weathered many storms in her thirty years at sea, but this one felt different—more personal, as if the ocean itself had a grudge against her.",
            "log_probability": -89.45,
            "normalized_score": -1.85,
            "length_normalized_log_probability": -1.45,
            "generation_length": 62,
            "is_finished": True,
            "tokens": list(range(100, 162))
        },
        {
            "rank": 2,
            "text": "There was a lighthouse and a storm. A woman was on a boat. The waves were big and scary. She was trying to get to safety but it was hard because of the weather conditions that were very bad.",
            "log_probability": -67.23,
            "normalized_score": -2.15,
            "length_normalized_log_probability": -1.89,
            "generation_length": 35,
            "is_finished": True,
            "tokens": list(range(200, 235))
        }
    ]
    
    evaluator = BeamEvaluator()
    prompt = "Write the opening paragraph of a story about a lighthouse keeper during a storm."
    
    print(f"\n{'='*60}")
    print(f"Testing Creative Writing")
    print(f"Prompt: {prompt}")
    print(f"{'='*60}")
    
    evaluation = evaluator.select_best_beam(prompt, creative_beams)
    
    print(f"Selected beam: {evaluation['selected_beam']['rank']}")
    print(f"Reasoning: {evaluation['selected_beam']['reasoning']}")
    print(f"Selected text: {evaluation['selected_beam']['text'][:100]}...")
    
    return evaluation

def main():
    """Run all tests."""
    print("Beam Search Evaluation System Test")
    print("=" * 60)
    
    # Test different prompt types
    results = test_different_prompts()
    
    # Test creative writing
    creative_result = test_creative_prompt()
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    matches = sum(1 for r in results if r['match'])
    total = len(results)
    
    print(f"Factual/Technical Tests: {matches}/{total} matches with expected results")
    print(f"Creative Test: Beam {creative_result['selected_beam']['rank']} selected")
    
    print(f"\nKey Insights:")
    print(f"- Simple explanation requests favor accessible language")
    print(f"- Technical questions prefer detailed, accurate responses")
    print(f"- Creative prompts prioritize engaging, well-written content")
    print(f"- Incomplete responses (is_finished=False) are penalized")
    print(f"- The evaluator can override model rankings when quality differs significantly")
    
    # Save detailed results
    output_file = "test_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'factual_tests': results,
            'creative_test': creative_result
        }, f, indent=2)
    
    print(f"\nDetailed results saved to {output_file}")

if __name__ == "__main__":
    main()