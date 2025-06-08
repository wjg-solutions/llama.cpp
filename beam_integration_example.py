#!/usr/bin/env python3
"""
Example integration of beam search evaluation with llama.cpp server

This script demonstrates how to:
1. Send a request to llama.cpp server with beam search enabled
2. Evaluate the beam results using the beam evaluator
3. Return the best beam result
"""

import json
import requests
import argparse
from beam_evaluator import BeamEvaluator

class LlamaBeamClient:
    def __init__(self, server_url: str = "http://localhost:8080"):
        self.server_url = server_url
        self.evaluator = BeamEvaluator()
    
    def generate_with_beam_search(self, prompt: str, beam_width: int = 4, **kwargs) -> dict:
        """
        Generate text using beam search and evaluate results.
        
        Args:
            prompt: The input prompt
            beam_width: Number of beams to maintain (default: 4)
            **kwargs: Additional parameters for the completion request
        
        Returns:
            Dictionary containing the best beam result and evaluation details
        """
        # Prepare request payload
        payload = {
            "prompt": prompt,
            "beam_width": beam_width,
            "beam_length_penalty": kwargs.get("beam_length_penalty", 1.0),
            "beam_diversity_penalty": kwargs.get("beam_diversity_penalty", 0.0),
            "beam_early_stopping": kwargs.get("beam_early_stopping", True),
            "beam_deterministic": kwargs.get("beam_deterministic", True),
            "n_predict": kwargs.get("n_predict", 128),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "stream": False  # We need the complete response for evaluation
        }
        
        # Add any additional parameters
        for key, value in kwargs.items():
            if key not in payload:
                payload[key] = value
        
        try:
            # Send request to llama.cpp server
            response = requests.post(
                f"{self.server_url}/completion",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Check if beam results are present
            beam_results = result.get("beam_results", [])
            if not beam_results:
                return {
                    "error": "No beam results returned from server",
                    "server_response": result
                }
            
            # Evaluate beam results
            evaluation = self.evaluator.select_best_beam(prompt, beam_results)
            
            # Combine server response with evaluation
            return {
                "prompt": prompt,
                "server_response": result,
                "beam_evaluation": evaluation,
                "best_result": {
                    "text": evaluation["selected_beam"]["text"],
                    "rank": evaluation["selected_beam"]["rank"],
                    "reasoning": evaluation["selected_beam"]["reasoning"]
                }
            }
            
        except requests.exceptions.RequestException as e:
            return {"error": f"Request failed: {str(e)}"}
        except json.JSONDecodeError as e:
            return {"error": f"Invalid JSON response: {str(e)}"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}
    
    def compare_beam_vs_greedy(self, prompt: str, **kwargs) -> dict:
        """
        Compare beam search results with greedy decoding.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional parameters
        
        Returns:
            Comparison results
        """
        # Generate with beam search
        beam_result = self.generate_with_beam_search(prompt, **kwargs)
        
        # Generate with greedy decoding (beam_width=1)
        greedy_kwargs = kwargs.copy()
        greedy_kwargs["beam_width"] = 1
        greedy_result = self.generate_with_beam_search(prompt, **greedy_kwargs)
        
        return {
            "prompt": prompt,
            "beam_search": beam_result,
            "greedy_decoding": greedy_result,
            "comparison": {
                "beam_better": beam_result.get("beam_evaluation", {}).get("selected_beam", {}).get("rank", 1) == 1,
                "beam_text": beam_result.get("best_result", {}).get("text", ""),
                "greedy_text": greedy_result.get("best_result", {}).get("text", "")
            }
        }

def main():
    parser = argparse.ArgumentParser(description="Test beam search evaluation with llama.cpp server")
    parser.add_argument("--server", default="http://localhost:8080", help="Server URL")
    parser.add_argument("--prompt", required=True, help="Input prompt")
    parser.add_argument("--beam-width", type=int, default=4, help="Beam width")
    parser.add_argument("--compare", action="store_true", help="Compare with greedy decoding")
    parser.add_argument("--output", help="Output file for results")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    client = LlamaBeamClient(args.server)
    
    if args.compare:
        result = client.compare_beam_vs_greedy(
            args.prompt,
            beam_width=args.beam_width
        )
    else:
        result = client.generate_with_beam_search(
            args.prompt,
            beam_width=args.beam_width
        )
    
    # Output results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Results written to {args.output}")
    
    if args.verbose:
        print(json.dumps(result, indent=2))
    else:
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            best_result = result.get("best_result", {})
            print(f"Best result (rank {best_result.get('rank', 'unknown')}):")
            print(f"Text: {best_result.get('text', 'No text available')}")
            print(f"Reasoning: {best_result.get('reasoning', 'No reasoning available')}")

if __name__ == "__main__":
    main()