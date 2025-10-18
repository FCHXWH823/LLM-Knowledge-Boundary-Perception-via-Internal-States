"""
Example usage of extract_embeddings.py

This script demonstrates how to use the EmbeddingExtractor class
programmatically (without using the command-line interface).

Note: This is an example script that requires torch and transformers to be installed.
"""

from extract_embeddings import EmbeddingExtractor
import json


def example_basic_usage():
    """Basic example: Extract embeddings from a simple prompt"""
    print("="*80)
    print("EXAMPLE 1: Basic Usage")
    print("="*80)
    
    # Initialize the extractor with a model
    # Using gpt2 as an example (you can replace with any model path)
    extractor = EmbeddingExtractor(
        model_path="gpt2",
        device="cuda"  # or "cpu" if no GPU available
    )
    
    # Extract embeddings
    result = extractor.extract_embeddings(
        prompt="What is artificial intelligence?",
        layer_spec="mid",  # Extract from middle layer
        max_new_tokens=30
    )
    
    # Print summary
    print(f"\nInput: {result['input_prompt']}")
    print(f"Generated: {result['generated_text']}")
    print(f"Extracted from layer(s): {result['layers']}")
    print(f"Hidden size: {result['hidden_size']}")
    print(f"Number of generated tokens: {len(result['generated_tokens'])}")
    
    # Access embeddings
    print(f"\nLast input token: {result['last_input_token_text']}")
    print(f"Last input token embedding shape: {len(result['last_input_token_embedding']['layer_6'])}")
    
    print("\nFirst generated token:")
    first_token = result['generated_tokens_embeddings'][0]
    print(f"  Token: {first_token['token_text']}")
    print(f"  Embedding shape: {len(first_token['embeddings']['layer_6'])}")
    
    return result


def example_multiple_layers():
    """Example: Extract embeddings from multiple layers"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Multiple Layers")
    print("="*80)
    
    extractor = EmbeddingExtractor(model_path="gpt2", device="cpu")
    
    # Extract from all layers
    result = extractor.extract_embeddings(
        prompt="Explain machine learning",
        layer_spec="all",
        max_new_tokens=20
    )
    
    print(f"\nExtracted from {len(result['layers'])} layers")
    print(f"Layers: {result['layers'][:5]}... (showing first 5)")
    
    # Show embeddings from multiple layers for first token
    first_token = result['generated_tokens_embeddings'][0]
    print(f"\nFirst generated token '{first_token['token_text']}' has embeddings from:")
    for layer_key in list(first_token['embeddings'].keys())[:3]:
        print(f"  {layer_key}: {len(first_token['embeddings'][layer_key])} dimensions")
    
    return result


def example_save_to_file():
    """Example: Extract and save embeddings to a file"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Save to File")
    print("="*80)
    
    extractor = EmbeddingExtractor(model_path="gpt2", device="cpu")
    
    result = extractor.extract_embeddings(
        prompt="What is deep learning?",
        layer_spec="last",
        max_new_tokens=25
    )
    
    # Save to JSON file
    output_file = "example_embeddings.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Embeddings saved to {output_file}")
    print(f"File contains:")
    print(f"  - Input prompt and tokens")
    print(f"  - Last input token embedding")
    print(f"  - {len(result['generated_tokens_embeddings'])} generated token embeddings")
    
    return result


def example_analyze_embeddings():
    """Example: Analyze the extracted embeddings"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Analyze Embeddings")
    print("="*80)
    
    extractor = EmbeddingExtractor(model_path="gpt2", device="cpu")
    
    result = extractor.extract_embeddings(
        prompt="Hello world",
        layer_spec="mid",
        max_new_tokens=10
    )
    
    # Analyze embeddings
    import statistics
    
    print("\nAnalyzing last input token embedding:")
    last_input_emb = result['last_input_token_embedding']['layer_6']
    print(f"  Mean: {statistics.mean(last_input_emb):.4f}")
    print(f"  Std Dev: {statistics.stdev(last_input_emb):.4f}")
    print(f"  Min: {min(last_input_emb):.4f}")
    print(f"  Max: {max(last_input_emb):.4f}")
    
    print("\nAnalyzing first generated token embedding:")
    first_gen_emb = result['generated_tokens_embeddings'][0]['embeddings']['layer_6']
    print(f"  Token: {result['generated_tokens_embeddings'][0]['token_text']}")
    print(f"  Mean: {statistics.mean(first_gen_emb):.4f}")
    print(f"  Std Dev: {statistics.stdev(first_gen_emb):.4f}")
    print(f"  Min: {min(first_gen_emb):.4f}")
    print(f"  Max: {max(first_gen_emb):.4f}")
    
    return result


def example_compare_layers():
    """Example: Compare embeddings across different layers"""
    print("\n" + "="*80)
    print("EXAMPLE 5: Compare Layers")
    print("="*80)
    
    extractor = EmbeddingExtractor(model_path="gpt2", device="cpu")
    
    prompt = "Python programming"
    
    # Extract from first, middle, and last layers
    results = {}
    for layer_spec in ['0', 'mid', 'last']:
        result = extractor.extract_embeddings(
            prompt=prompt,
            layer_spec=layer_spec,
            max_new_tokens=5
        )
        results[layer_spec] = result
    
    print(f"\nComparing embeddings across layers for prompt: '{prompt}'")
    print(f"Generated text: {results['mid']['generated_text']}")
    
    import statistics
    
    for layer_spec, result in results.items():
        layer_idx = result['layers'][0]
        layer_key = f"layer_{layer_idx}"
        emb = result['last_input_token_embedding'][layer_key]
        
        print(f"\n{layer_spec} (layer {layer_idx}):")
        print(f"  Mean activation: {statistics.mean(emb):.4f}")
        print(f"  Std Dev: {statistics.stdev(emb):.4f}")
    
    return results


def example_batch_processing():
    """Example: Process multiple prompts"""
    print("\n" + "="*80)
    print("EXAMPLE 6: Batch Processing")
    print("="*80)
    
    extractor = EmbeddingExtractor(model_path="gpt2", device="cpu")
    
    prompts = [
        "What is AI?",
        "Define machine learning",
        "Explain neural networks"
    ]
    
    results = []
    for i, prompt in enumerate(prompts, 1):
        print(f"\nProcessing prompt {i}/{len(prompts)}: {prompt}")
        result = extractor.extract_embeddings(
            prompt=prompt,
            layer_spec="mid",
            max_new_tokens=15
        )
        results.append(result)
        print(f"  Generated: {result['generated_text']}")
    
    print(f"\n✓ Processed {len(results)} prompts")
    
    # Save all results
    output_file = "batch_embeddings.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✓ All results saved to {output_file}")
    
    return results


def main():
    """Run all examples"""
    print("\n" + "="*80)
    print("EMBEDDING EXTRACTOR - USAGE EXAMPLES")
    print("="*80)
    print("\nThese examples demonstrate various ways to use the EmbeddingExtractor.")
    print("Note: Examples use gpt2 model for demonstration purposes.")
    print("\nStarting examples...\n")
    
    try:
        # Run examples
        example_basic_usage()
        example_multiple_layers()
        example_save_to_file()
        example_analyze_embeddings()
        example_compare_layers()
        example_batch_processing()
        
        print("\n" + "="*80)
        print("✓ ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("="*80)
        print("\nGenerated files:")
        print("  - example_embeddings.json")
        print("  - batch_embeddings.json")
        
    except Exception as e:
        print(f"\n✗ Error running examples: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
