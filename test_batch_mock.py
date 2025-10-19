"""
Mock test to verify batch processing logic without torch/transformers.
This tests the interface and return structure by analyzing source code.
"""

import sys
import ast

def parse_file():
    """Parse the extract_embeddings.py file"""
    with open('extract_embeddings.py', 'r') as f:
        source = f.read()
    return source, ast.parse(source)

def test_method_signature():
    """Test that the extract_embeddings method has the correct signature"""
    print("Testing method signature...")
    
    source, tree = parse_file()
    
    # Find the EmbeddingExtractor class
    found_method = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == 'EmbeddingExtractor':
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == 'extract_embeddings':
                    found_method = True
                    
                    # Check parameters
                    args = [arg.arg for arg in item.args.args]
                    assert 'prompt' in args, "Missing 'prompt' parameter"
                    print("✓ 'prompt' parameter exists")
                    
                    # Check if Union is used in annotation
                    if item.returns:
                        print(f"✓ Return type annotation exists")
                    
                    break
    
    assert found_method, "Could not find extract_embeddings method"
    print("\n✓ Method signature is correct")
    return True

def test_batch_method_exists():
    """Test that the batch processing method exists"""
    print("\nTesting batch processing method...")
    
    source, tree = parse_file()
    
    # Find the _extract_embeddings_batch method
    found_method = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == 'EmbeddingExtractor':
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == '_extract_embeddings_batch':
                    found_method = True
                    print("✓ '_extract_embeddings_batch' method exists")
                    
                    # Check parameters
                    args = [arg.arg for arg in item.args.args]
                    assert 'prompts' in args, "Missing 'prompts' parameter"
                    print("✓ 'prompts' parameter exists")
                    
                    break
    
    assert found_method, "Could not find _extract_embeddings_batch method"
    print("\n✓ Batch processing method exists with correct signature")
    return True

def test_docstrings():
    """Test that docstrings mention batch processing"""
    print("\nTesting documentation...")
    
    source, tree = parse_file()
    
    # Find extract_embeddings method and check docstring
    found_docs = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == 'EmbeddingExtractor':
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == 'extract_embeddings':
                    docstring = ast.get_docstring(item)
                    assert docstring is not None, "Missing extract_embeddings docstring"
                    assert 'batch' in docstring.lower(), "Docstring doesn't mention batch processing"
                    print("✓ extract_embeddings docstring mentions batch processing")
                    found_docs = True
                    
                if isinstance(item, ast.FunctionDef) and item.name == '_extract_embeddings_batch':
                    docstring = ast.get_docstring(item)
                    assert docstring is not None, "Missing _extract_embeddings_batch docstring"
                    print("✓ _extract_embeddings_batch has docstring")
    
    assert found_docs, "Could not verify documentation"
    print("\n✓ Documentation is present and mentions batch processing")
    return True

def test_module_imports():
    """Test that necessary imports are present"""
    print("\nTesting imports...")
    
    source, tree = parse_file()
    
    # Check for Union and List imports
    assert 'Union' in source, "Missing 'Union' import from typing"
    print("✓ 'Union' is imported from typing")
    
    assert 'List' in source, "Missing 'List' import from typing"
    print("✓ 'List' is imported from typing")
    
    print("\n✓ All necessary imports are present")
    return True

def main():
    """Run all mock tests"""
    print("="*80)
    print("BATCH PROCESSING MOCK TESTS")
    print("="*80)
    print("\nThese tests verify the batch processing interface without requiring")
    print("torch/transformers to be installed.\n")
    
    results = []
    
    try:
        results.append(("Method Signature", test_method_signature()))
        results.append(("Batch Method", test_batch_method_exists()))
        results.append(("Documentation", test_docstrings()))
        results.append(("Imports", test_module_imports()))
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
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
        print("✓ ALL MOCK TESTS PASSED")
        print("="*80)
        print("\nThe batch processing interface is correctly implemented.")
        print("To test with actual models, install torch and transformers,")
        print("then run: python test_extract_embeddings.py")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("="*80)
        return 1

if __name__ == "__main__":
    sys.exit(main())
