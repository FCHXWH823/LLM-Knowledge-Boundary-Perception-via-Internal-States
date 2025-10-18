# Quick Start Guide - Embedding Extraction

This is a quick reference guide to get started with the embedding extraction utility.

## 📦 Installation

```bash
pip install torch transformers
```

## 🚀 Quick Start

### Command Line Usage

Extract embeddings from the middle layer:
```bash
python extract_embeddings.py \
    --model_path gpt2 \
    --prompt "What is AI?" \
    --layer mid \
    --max_new_tokens 20
```

Save to file:
```bash
python extract_embeddings.py \
    --model_path gpt2 \
    --prompt "Explain machine learning" \
    --layer mid \
    --max_new_tokens 30 \
    --output_file my_embeddings.json
```

### Python API Usage

```python
from extract_embeddings import EmbeddingExtractor

# Initialize
extractor = EmbeddingExtractor(model_path="gpt2")

# Extract embeddings
result = extractor.extract_embeddings(
    prompt="Hello world",
    layer_spec="mid",
    max_new_tokens=20
)

# Access results
print(f"Generated: {result['generated_text']}")
print(f"Last input token: {result['last_input_token_text']}")
print(f"Number of layers: {len(result['layers'])}")

# Access embeddings
last_input_emb = result['last_input_token_embedding']
gen_token_embs = result['generated_tokens_embeddings']
```

## 📝 Key Arguments

| Argument | Description | Example Values |
|----------|-------------|----------------|
| `--model_path` | Model path or HF model ID | `gpt2`, `/path/to/model` |
| `--prompt` | Input text | `"What is AI?"` |
| `--layer` | Layer to extract from | `mid`, `last`, `all`, `16` |
| `--max_new_tokens` | Max tokens to generate | `20`, `50` |
| `--output_file` | Output JSON file | `embeddings.json` |
| `--device` | Device to use | `cuda`, `cpu` |

## 📊 Output Structure

```json
{
  "input_prompt": "What is AI?",
  "last_input_token_text": "?",
  "last_input_token_embedding": {
    "layer_6": [0.123, -0.456, ...]
  },
  "generated_text": "Artificial Intelligence is...",
  "generated_tokens_embeddings": [
    {
      "token_text": "Artificial",
      "embeddings": {
        "layer_6": [0.234, -0.567, ...]
      }
    },
    ...
  ],
  "layers": [6],
  "hidden_size": 768
}
```

## 🎯 Common Use Cases

### 1. Extract from specific layer
```bash
python extract_embeddings.py --model_path gpt2 --prompt "Test" --layer 8
```

### 2. Extract from all layers
```bash
python extract_embeddings.py --model_path gpt2 --prompt "Test" --layer all
```

### 3. Force CPU usage
```bash
python extract_embeddings.py --model_path gpt2 --prompt "Test" --device cpu
```

### 4. Longer generation
```bash
python extract_embeddings.py --model_path gpt2 --prompt "Test" --max_new_tokens 100
```

## 🔍 Accessing Embeddings in Python

```python
from extract_embeddings import EmbeddingExtractor
import json

# Extract
extractor = EmbeddingExtractor("gpt2")
result = extractor.extract_embeddings("Hello", "mid", 10)

# Save
with open("out.json", "w") as f:
    json.dump(result, f, indent=2)

# Process
for token_data in result['generated_tokens_embeddings']:
    token = token_data['token_text']
    # Get first layer's embedding
    layer_key = list(token_data['embeddings'].keys())[0]
    embedding = token_data['embeddings'][layer_key]
    print(f"{token}: {len(embedding)} dimensions")
```

## 📚 Additional Resources

- **Full Documentation**: [EXTRACT_EMBEDDINGS_README.md](EXTRACT_EMBEDDINGS_README.md)
- **Example Code**: [example_usage.py](example_usage.py)
- **Tests**: [test_extract_embeddings.py](test_extract_embeddings.py)

## ⚡ Pro Tips

1. **Use specific layers instead of 'all'** to save memory with large models
2. **Start with small max_new_tokens** when testing
3. **Use CPU for small models** if GPU memory is limited
4. **The embeddings are in float format** (converted from float16 for JSON compatibility)
5. **Check the 'layers' field** in output to see which layer indices were extracted

## 🐛 Troubleshooting

**Out of Memory?**
- Use `--device cpu`
- Reduce `--max_new_tokens`
- Use specific layer instead of `--layer all`

**Model not found?**
- Verify model path exists
- Check internet connection for HuggingFace models
- Ensure you have authentication for gated models

**Slow generation?**
- Expected on CPU for large models
- Try a smaller model like `gpt2` for testing
- Use GPU if available

## 💡 Examples

See [example_usage.py](example_usage.py) for 6 complete examples including:
- Basic extraction
- Multiple layers
- Saving to file
- Analyzing embeddings
- Comparing layers
- Batch processing

Run examples:
```bash
python example_usage.py
```
