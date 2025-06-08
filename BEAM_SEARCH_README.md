# Beam Search Implementation and Evaluation System

This document describes the beam search implementation added to llama.cpp and the evaluation system for selecting the best beam results.

## Overview

The `feat/server-beam` branch adds comprehensive beam search capabilities to the llama.cpp server, allowing generation of multiple candidate completions and intelligent selection of the best result based on various evaluation criteria.

## Changes Summary

### Core Implementation Changes

1. **Command Line Arguments** ([`common/arg.cpp`](common/arg.cpp))
   - `--beams N`: Set beam width (default: 1, greedy)
   - `--beam-length-penalty N`: Length normalization penalty (default: 1.0)
   - `--beam-diversity-penalty N`: Diversity penalty (default: 0.0)
   - `--no-beam-early-stopping`: Disable early stopping
   - `--beam-deterministic`: Enable deterministic beam search

2. **Common Parameters** ([`common/common.h`](common/common.h))
   - Added beam search parameters to `common_params` struct
   - Support for length penalty, diversity penalty, early stopping, and deterministic mode

3. **Enhanced Sampling** ([`common/sampling.cpp`](common/sampling.cpp), [`common/sampling.h`](common/sampling.h))
   - `common_sampler_get_candidate_probs()`: Get probabilities before final sampling
   - `common_sampler_get_candidate_probs_fast()`: Optimized version for beam search
   - `common_sampler_reset_candidates()`: Reset candidate state

4. **Server Integration** ([`tools/server/server.cpp`](tools/server/server.cpp))
   - Complete beam search algorithm implementation
   - Beam candidate management with scoring and ranking
   - Memory limits and safety checks
   - Streaming support for beam search
   - JSON API integration

### Key Features

- **Multiple Beam Management**: Maintains multiple candidate sequences simultaneously
- **Length Normalization**: Prevents bias toward shorter sequences
- **Diversity Penalty**: Encourages diverse outputs
- **Early Stopping**: Terminates when sufficient good candidates are found
- **Deterministic Mode**: Ensures reproducible results
- **Memory Management**: Configurable memory limits and token limits per candidate
- **Streaming Support**: Real-time streaming of consensus beam results

## Beam Search Parameters

### Basic Parameters
- `beam_width` (int): Number of beams to maintain (1 = greedy, >1 = beam search)
- `beam_length_penalty` (float): Length normalization factor (1.0 = no penalty)
- `beam_diversity_penalty` (float): Diversity encouragement (0.0 = no penalty)

### Advanced Parameters
- `beam_early_stopping` (bool): Stop when enough finished candidates found
- `beam_deterministic` (bool): Use deterministic tie-breaking for reproducibility
- `beam_max_memory_mb` (int): Maximum memory usage in MB
- `beam_max_tokens_per_candidate` (int): Maximum tokens per beam candidate

## API Usage

### HTTP API Example

```bash
curl -X POST http://localhost:8080/completion \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain how photosynthesis works",
    "beam_width": 4,
    "beam_length_penalty": 1.2,
    "beam_diversity_penalty": 0.1,
    "n_predict": 128
  }'
```

### Response Format

```json
{
  "content": "Best beam result text...",
  "beam_results": [
    {
      "rank": 1,
      "text": "Generated text for beam 1",
      "log_probability": -15.234,
      "normalized_score": -12.456,
      "length_normalized_log_probability": -0.789,
      "generation_length": 25,
      "is_finished": true,
      "tokens": [123, 456, 789]
    }
  ]
}
```

## Evaluation System

### Beam Evaluator ([`beam_evaluator.py`](beam_evaluator.py))

The beam evaluator automatically selects the best beam result based on multiple criteria:

#### Evaluation Criteria (Weighted)
1. **Relevance and Coherence (40%)**: How well the response addresses the prompt
2. **Quality and Accuracy (30%)**: Language quality and factual correctness
3. **Completeness (20%)**: Whether the response feels complete
4. **Technical Metrics (10%)**: Model confidence and beam search scores

#### Prompt Type Classification
- **Creative**: Stories, poems, creative writing
- **Factual**: Explanations, definitions, factual queries
- **Technical**: Code, algorithms, technical explanations
- **Question/Answer**: Direct questions requiring specific answers
- **Conversational**: Chat, discussion prompts

#### Usage Example

```python
from beam_evaluator import BeamEvaluator

evaluator = BeamEvaluator()
result = evaluator.select_best_beam(prompt, beam_results)

print(f"Selected beam {result['selected_beam']['rank']}")
print(f"Text: {result['selected_beam']['text']}")
print(f"Reasoning: {result['selected_beam']['reasoning']}")
```

### Integration Example ([`beam_integration_example.py`](beam_integration_example.py))

Complete client implementation that:
- Sends beam search requests to llama.cpp server
- Evaluates results using the beam evaluator
- Returns the best beam with reasoning

```python
from beam_integration_example import LlamaBeamClient

client = LlamaBeamClient("http://localhost:8080")
result = client.generate_with_beam_search(
    "Write a short story about a robot",
    beam_width=4,
    beam_length_penalty=1.1
)

print(result["best_result"]["text"])
```

## Evaluation Prompt ([`beam_evaluation_prompt.md`](beam_evaluation_prompt.md))

Comprehensive prompt template for manual evaluation of beam search results, including:
- Detailed evaluation criteria
- Step-by-step evaluation process
- Output format specification
- Special considerations for different prompt types

## Performance Considerations

### Memory Usage
- Each beam candidate stores tokens, text, and metadata
- Memory usage scales with `beam_width × average_sequence_length`
- Configurable limits prevent excessive memory consumption

### Computational Cost
- Beam search requires evaluating multiple candidates at each step
- Cost scales roughly linearly with beam width
- Fast candidate probability extraction optimizes performance

### Quality vs. Speed Trade-offs
- Higher beam width generally improves quality but increases cost
- Length penalty helps balance quality and efficiency
- Early stopping reduces unnecessary computation

## Best Practices

### Parameter Tuning
- Start with `beam_width=4` for most applications
- Use `beam_length_penalty=1.1-1.3` to encourage complete responses
- Add small `beam_diversity_penalty=0.1-0.3` for creative tasks
- Enable `beam_deterministic=true` for reproducible results

### Use Cases
- **Creative Writing**: Higher diversity penalty, moderate length penalty
- **Factual Q&A**: Lower beam width, focus on accuracy
- **Code Generation**: Deterministic mode, early stopping enabled
- **Long-form Content**: Higher length penalty, larger beam width

### Monitoring
- Monitor memory usage with large beam widths
- Check beam result diversity to ensure effective search
- Evaluate consensus beam selection for streaming applications

## Testing

### Manual Testing
```bash
# Test basic beam search
python beam_integration_example.py \
  --prompt "Explain quantum computing" \
  --beam-width 4 \
  --verbose

# Compare with greedy decoding
python beam_integration_example.py \
  --prompt "Write a haiku about programming" \
  --beam-width 3 \
  --compare
```

### Evaluation Testing
```bash
# Evaluate beam results from file
python beam_evaluator.py \
  --input beam_results.json \
  --output evaluation.json \
  --verbose
```

## Future Enhancements

### Potential Improvements
1. **Advanced Scoring**: Incorporate semantic similarity metrics
2. **Context Awareness**: Better handling of multi-turn conversations
3. **Domain-Specific Evaluation**: Specialized evaluators for different domains
4. **Real-time Adaptation**: Dynamic parameter adjustment based on prompt analysis
5. **Batch Processing**: Efficient evaluation of multiple prompts simultaneously

### Integration Opportunities
1. **Chat Applications**: Multi-turn conversation beam search
2. **Content Generation**: Automated content creation with quality assurance
3. **Code Assistance**: Programming help with multiple solution candidates
4. **Educational Tools**: Multiple explanation approaches for learning

## Troubleshooting

### Common Issues
1. **High Memory Usage**: Reduce beam width or set memory limits
2. **Slow Performance**: Use fast candidate extraction, enable early stopping
3. **Poor Quality Results**: Adjust length penalty, increase beam width
4. **Inconsistent Results**: Enable deterministic mode
5. **Empty Beam Results**: Check server configuration and model compatibility

### Debug Information
- Server logs include beam search memory usage and candidate counts
- Evaluation system provides detailed scoring breakdown
- Verbose modes available for both server and evaluation components

## Conclusion

The beam search implementation provides a powerful tool for generating high-quality text completions. Combined with the intelligent evaluation system, it enables automatic selection of the best results based on comprehensive quality criteria. This system is particularly valuable for applications requiring consistent, high-quality text generation.