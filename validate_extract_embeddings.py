"""
Validation script for extract_embeddings.py

This script validates the structure and documentation of extract_embeddings.py
without requiring torch/transformers to be installed.
"""

import ast
import sys
import os


def validate_file_structure(filepath):
    """Validate the structure of the Python file"""
    print(f"\nValidating {filepath}...")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Parse the AST
    try:
        tree = ast.parse(content)
        print("✓ Valid Python syntax")
    except SyntaxError as e:
        print(f"✗ Syntax error: {e}")
        return False
    
    # Find classes and functions
    classes = []
    functions = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            classes.append(node.name)
        elif isinstance(node, ast.FunctionDef):
            if not node.name.startswith('_') or node.name in ['__init__', '__main__']:
                functions.append(node.name)
    
    print(f"\nFound classes: {classes}")
    print(f"Found functions: {functions}")
    
    # Check for expected classes
    if 'EmbeddingExtractor' not in classes:
        print("✗ Missing EmbeddingExtractor class")
        return False
    print("✓ EmbeddingExtractor class found")
    
    # Check for expected methods
    expected_methods = ['__init__', 'extract_embeddings']
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == 'EmbeddingExtractor':
            method_names = [m.name for m in node.body if isinstance(m, ast.FunctionDef)]
            for method in expected_methods:
                if method in method_names:
                    print(f"✓ Method {method} found")
                else:
                    print(f"✗ Method {method} not found")
                    return False
    
    # Check for main function
    if 'main' not in functions:
        print("✗ Missing main function")
        return False
    print("✓ main function found")
    
    # Check for docstrings
    has_module_docstring = ast.get_docstring(tree) is not None
    if has_module_docstring:
        print("✓ Module has docstring")
    else:
        print("⚠ Module missing docstring")
    
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == 'EmbeddingExtractor':
            class_docstring = ast.get_docstring(node)
            if class_docstring:
                print("✓ EmbeddingExtractor class has docstring")
            else:
                print("⚠ EmbeddingExtractor class missing docstring")
            
            for method_node in node.body:
                if isinstance(method_node, ast.FunctionDef):
                    method_docstring = ast.get_docstring(method_node)
                    if method_docstring:
                        print(f"✓ Method {method_node.name} has docstring")
                    else:
                        print(f"⚠ Method {method_node.name} missing docstring")
    
    return True


def validate_imports(filepath):
    """Validate that all necessary imports are present"""
    print("\nValidating imports...")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    tree = ast.parse(content)
    
    required_imports = {
        'argparse': False,
        'torch': False,
        'json': False,
        'transformers': False
    }
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in required_imports:
                    required_imports[alias.name] = True
        elif isinstance(node, ast.ImportFrom):
            if node.module in required_imports:
                required_imports[node.module] = True
    
    all_present = True
    for module, found in required_imports.items():
        if found:
            print(f"✓ Import {module}: present")
        else:
            print(f"✗ Import {module}: missing")
            all_present = False
    
    return all_present


def validate_command_line_args(filepath):
    """Validate command line argument parsing"""
    print("\nValidating command line arguments...")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    required_args = {
        '--model_path': False,
        '--prompt': False,
        '--layer': False,
        '--max_new_tokens': False,
        '--output_file': False,
        '--device': False
    }
    
    for arg in required_args:
        if arg in content:
            required_args[arg] = True
            print(f"✓ Argument {arg}: defined")
        else:
            print(f"✗ Argument {arg}: not found")
    
    return all(required_args.values())


def check_file_size(filepath):
    """Check file size"""
    size = os.path.getsize(filepath)
    print(f"\nFile size: {size} bytes ({size/1024:.2f} KB)")
    return True


def check_documentation(filepath):
    """Check for comprehensive documentation"""
    print("\nChecking documentation...")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    doc_items = {
        'Usage examples': 'Usage:' in content or 'Example:' in content,
        'Argument descriptions': 'Arguments:' in content or 'Args:' in content,
        'Return descriptions': 'Returns:' in content or 'Return:' in content,
    }
    
    for item, present in doc_items.items():
        if present:
            print(f"✓ {item}: present")
        else:
            print(f"⚠ {item}: may be missing")
    
    return True


def main():
    """Run all validations"""
    print("="*80)
    print("EXTRACT_EMBEDDINGS.PY VALIDATION")
    print("="*80)
    
    filepath = "extract_embeddings.py"
    
    if not os.path.exists(filepath):
        print(f"✗ File {filepath} not found!")
        return 1
    
    print(f"✓ File {filepath} exists")
    
    results = []
    
    # Run validations
    results.append(("File Structure", validate_file_structure(filepath)))
    results.append(("Imports", validate_imports(filepath)))
    results.append(("Command Line Args", validate_command_line_args(filepath)))
    results.append(("File Size", check_file_size(filepath)))
    results.append(("Documentation", check_documentation(filepath)))
    
    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    for check_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{check_name}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("\n" + "="*80)
    if all_passed:
        print("✓ ALL VALIDATIONS PASSED")
        print("="*80)
        print("\nThe file is ready for use!")
        print("To test with actual models, install dependencies:")
        print("  pip install torch transformers")
        return 0
    else:
        print("✗ SOME VALIDATIONS FAILED")
        print("="*80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
