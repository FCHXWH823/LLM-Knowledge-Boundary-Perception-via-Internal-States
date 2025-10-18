"""
Simple test script for extract_embeddings.py

This script tests the embedding extraction functionality without requiring
a large model download. It uses the smallest available model (gpt2) for testing.

Note: This is a basic functional test. For full testing, you would need access
to the actual models used in the repository.
"""

import sys
import os
import json

# Test if we can import the module
try:
    from extract_embeddings import EmbeddingExtractor
    print("✓ Successfully imported EmbeddingExtractor")
except ImportError as e:
    print(f"✗ Failed to import: {e}")
    sys.exit(1)

def test_basic_functionality():
    """Test basic embedding extraction with gpt2 (small model for testing)"""
    print("\n" + "="*80)
    print("TEST: Basic Embedding Extraction")
    print("="*80)
    
    try:
        # Use gpt2 as it's small and commonly available
        print("\nInitializing extractor with gpt2...")
        extractor = EmbeddingExtractor(model_path="gpt2", device="cpu")
        print("✓ Extractor initialized successfully")
        
        # Test extraction with mid layer
        print("\nExtracting embeddings (mid layer)...")
        result = extractor.extract_embeddings(
            prompt="Hello world",
            layer_spec="mid",
            max_new_tokens=5
        )
        print("✓ Embeddings extracted successfully")
        
        # Verify result structure
        print("\nVerifying result structure...")
        required_keys = [
            'input_prompt',
            'input_tokens',
            'input_text_tokens',
            'last_input_token_id',
            'last_input_token_text',
            'last_input_token_embedding',
            'generated_text',
            'generated_tokens',
            'generated_text_tokens',
            'generated_tokens_embeddings',
            'layers',
            'layer_spec',
            'num_layers',
            'hidden_size'
        ]
        
        for key in required_keys:
            if key not in result:
                print(f"✗ Missing key: {key}")
                return False
            else:
                print(f"  ✓ {key}: present")
        
        # Verify data types and content
        print("\nVerifying data content...")
        assert isinstance(result['input_tokens'], list), "input_tokens should be a list"
        assert isinstance(result['generated_tokens'], list), "generated_tokens should be a list"
        assert isinstance(result['last_input_token_embedding'], dict), "last_input_token_embedding should be a dict"
        assert isinstance(result['generated_tokens_embeddings'], list), "generated_tokens_embeddings should be a list"
        assert len(result['generated_tokens_embeddings']) > 0, "Should have at least one generated token"
        assert len(result['layers']) > 0, "Should have at least one layer"
        print("✓ Data types are correct")
        
        # Check embedding dimensions
        print("\nVerifying embedding dimensions...")
        for layer_key, embedding in result['last_input_token_embedding'].items():
            assert isinstance(embedding, list), f"{layer_key} embedding should be a list"
            assert len(embedding) == result['hidden_size'], f"{layer_key} embedding size mismatch"
            print(f"  ✓ {layer_key}: {len(embedding)} dimensions")
        
        for token_emb in result['generated_tokens_embeddings']:
            assert 'token_id' in token_emb, "Each token should have token_id"
            assert 'token_text' in token_emb, "Each token should have token_text"
            assert 'embeddings' in token_emb, "Each token should have embeddings"
            for layer_key, embedding in token_emb['embeddings'].items():
                assert len(embedding) == result['hidden_size'], f"Token embedding size mismatch"
        print("✓ All embeddings have correct dimensions")
        
        print("\n" + "="*80)
        print("✓ ALL TESTS PASSED")
        print("="*80)
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_layer_specifications():
    """Test different layer specification options"""
    print("\n" + "="*80)
    print("TEST: Layer Specifications")
    print("="*80)
    
    try:
        print("\nInitializing extractor...")
        extractor = EmbeddingExtractor(model_path="gpt2", device="cpu")
        
        layer_specs = ['last', 'mid', '0', '6']
        
        for spec in layer_specs:
            print(f"\nTesting layer spec: {spec}")
            result = extractor.extract_embeddings(
                prompt="Test",
                layer_spec=spec,
                max_new_tokens=3
            )
            print(f"  ✓ Layer {spec}: extracted from layer(s) {result['layers']}")
        
        print("\n✓ All layer specifications work correctly")
        return True
        
    except Exception as e:
        print(f"\n✗ Layer specification test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_json_serialization():
    """Test that results can be serialized to JSON"""
    print("\n" + "="*80)
    print("TEST: JSON Serialization")
    print("="*80)
    
    try:
        print("\nInitializing extractor...")
        extractor = EmbeddingExtractor(model_path="gpt2", device="cpu")
        
        print("\nExtracting embeddings...")
        result = extractor.extract_embeddings(
            prompt="Test JSON",
            layer_spec="mid",
            max_new_tokens=5
        )
        
        print("\nSerializing to JSON...")
        json_str = json.dumps(result, indent=2)
        print(f"✓ Successfully serialized ({len(json_str)} bytes)")
        
        print("\nDeserializing from JSON...")
        result_loaded = json.loads(json_str)
        print("✓ Successfully deserialized")
        
        print("\nVerifying data integrity...")
        assert result_loaded['input_prompt'] == result['input_prompt']
        assert result_loaded['layers'] == result['layers']
        assert result_loaded['hidden_size'] == result['hidden_size']
        print("✓ Data integrity preserved")
        
        print("\n✓ JSON serialization test passed")
        return True
        
    except Exception as e:
        print(f"\n✗ JSON serialization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("="*80)
    print("EMBEDDING EXTRACTOR TEST SUITE")
    print("="*80)
    print("\nThis test suite will verify the embedding extraction functionality.")
    print("Note: Tests use gpt2 model which will be downloaded if not cached.")
    print("\nStarting tests...")
    
    results = []
    
    # Run tests
    results.append(("Basic Functionality", test_basic_functionality()))
    results.append(("Layer Specifications", test_layer_specifications()))
    results.append(("JSON Serialization", test_json_serialization()))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("\n" + "="*80)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("="*80)
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("="*80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
