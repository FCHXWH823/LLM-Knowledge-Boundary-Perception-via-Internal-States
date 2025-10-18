"""
Embedding Extractor for LLM Hidden States

This script extracts embeddings (hidden states) from LLM outputs:
1. The embedding of the last token of the input prompt
2. The embeddings of each token in the LLM response
3. From a specified layer

Usage:
    python extract_embeddings.py \
        --model_path /path/to/model \
        --prompt "What is the capital of France?" \
        --layer mid \
        --max_new_tokens 50

Arguments:
    --model_path: Path to the pretrained model
    --prompt: Input prompt text
    --layer: Which layer to extract from. Options: 'all', 'last', 'mid', or a specific layer number (e.g., '16')
    --max_new_tokens: Maximum number of tokens to generate (default: 50)
    --output_file: Optional path to save the embeddings as JSON (default: None, prints to stdout)
    --device: Device to use for inference (default: 'cuda' if available, else 'cpu')

Example:
    python extract_embeddings.py \
        --model_path "meta-llama/Llama-2-7b-chat-hf" \
        --prompt "What is machine learning?" \
        --layer mid \
        --max_new_tokens 30 \
        --output_file embeddings.json
"""

import argparse
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Union, Optional
import sys


class EmbeddingExtractor:
    """
    Extract embeddings from LLM outputs for the last input token and all response tokens.
    """
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        Initialize the embedding extractor.
        
        Args:
            model_path: Path to the pretrained model
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
        """
        self.model_path = model_path
        
        # Determine device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Loading model from {model_path}...")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
            device_map=None
        )
        self.model.to(self.device)
        self.model.eval()
        
        # Set padding token
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        print(f"Model loaded successfully on {self.device}")
        print(f"Model has {len(self.model.config.to_dict().get('num_hidden_layers', 32))} layers")
    
    def _get_layer_indices(self, layer_spec: str, num_layers: int) -> List[int]:
        """
        Convert layer specification to list of layer indices.
        
        Args:
            layer_spec: 'all', 'last', 'mid', or a specific layer number
            num_layers: Total number of layers in the model
            
        Returns:
            List of layer indices (0-indexed)
        """
        if layer_spec == 'all':
            return list(range(num_layers))
        elif layer_spec == 'last':
            return [num_layers - 1]
        elif layer_spec == 'mid':
            return [num_layers // 2]
        else:
            try:
                layer_idx = int(layer_spec)
                if 0 <= layer_idx < num_layers:
                    return [layer_idx]
                else:
                    raise ValueError(f"Layer index {layer_idx} out of range [0, {num_layers-1}]")
            except ValueError:
                raise ValueError(f"Invalid layer specification: {layer_spec}. Use 'all', 'last', 'mid', or a number.")
    
    def extract_embeddings(
        self,
        prompt: str,
        layer_spec: str = 'mid',
        max_new_tokens: int = 50
    ) -> Dict:
        """
        Extract embeddings from the last input token and all response tokens.
        
        Args:
            prompt: Input text prompt
            layer_spec: Which layer(s) to extract from ('all', 'last', 'mid', or layer number)
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            Dictionary containing:
                - input_prompt: The input prompt text
                - input_tokens: List of input token IDs
                - input_text_tokens: List of input tokens as text
                - last_input_token_embedding: Embedding of the last input token (per layer)
                - generated_text: The generated response text
                - generated_tokens: List of generated token IDs
                - generated_text_tokens: List of generated tokens as text
                - generated_tokens_embeddings: Embeddings for each generated token (per layer)
                - layers: List of layer indices that were extracted
        """
        print(f"\nProcessing prompt: {prompt}")
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        input_ids = inputs['input_ids']
        input_len = input_ids.shape[1]
        
        print(f"Input length: {input_len} tokens")
        
        # Generate with hidden states
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                output_hidden_states=True,
                return_dict_in_generate=True,
                do_sample=False,  # Greedy decoding
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Extract generated sequence
        generated_ids = outputs.sequences[0]  # Get first (and only) sequence
        generated_tokens_only = generated_ids[input_len:]  # Exclude input tokens
        
        # Decode texts
        input_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=False)
        generated_text = self.tokenizer.decode(generated_tokens_only, skip_special_tokens=True)
        
        print(f"Generated {len(generated_tokens_only)} tokens")
        print(f"Generated text: {generated_text}")
        
        # Get number of layers from hidden states
        # outputs.hidden_states is a tuple of tuples: (generation_step, layer)
        # Each layer is (batch_size, sequence_length, hidden_size)
        num_layers = len(outputs.hidden_states[0])  # Number of layers including embedding layer
        
        # Determine which layers to extract
        layer_indices = self._get_layer_indices(layer_spec, num_layers)
        print(f"Extracting from layer(s): {layer_indices}")
        
        # Extract last input token embedding
        # The first element of hidden_states corresponds to the input forward pass
        # We need the hidden states at the last input position
        last_input_token_embeddings = {}
        if len(outputs.hidden_states) > 0:
            for layer_idx in layer_indices:
                # outputs.hidden_states[0] contains hidden states from the initial forward pass
                # Shape: (batch_size, input_len, hidden_size)
                hidden_state = outputs.hidden_states[0][layer_idx]
                last_token_embedding = hidden_state[0, -1, :]  # Last position of input
                last_input_token_embeddings[f"layer_{layer_idx}"] = last_token_embedding.cpu().float().tolist()
        
        # Extract embeddings for each generated token
        generated_tokens_embeddings = []
        
        for gen_step, token_id in enumerate(generated_tokens_only):
            token_embeddings = {}
            # gen_step corresponds to the index in outputs.hidden_states
            # Note: hidden_states[0] is from input, hidden_states[1:] are from generation
            if gen_step + 1 < len(outputs.hidden_states):
                for layer_idx in layer_indices:
                    # Get hidden state for this generation step
                    # Shape: (batch_size, 1, hidden_size) for generation steps
                    hidden_state = outputs.hidden_states[gen_step + 1][layer_idx]
                    token_embedding = hidden_state[0, -1, :]  # Last (and only new) position
                    token_embeddings[f"layer_{layer_idx}"] = token_embedding.cpu().float().tolist()
            
            generated_tokens_embeddings.append({
                'token_id': int(token_id),
                'token_text': self.tokenizer.decode([token_id]),
                'embeddings': token_embeddings
            })
        
        # Prepare result
        result = {
            'input_prompt': prompt,
            'input_tokens': input_ids[0].cpu().tolist(),
            'input_text_tokens': self.tokenizer.convert_ids_to_tokens(input_ids[0]),
            'last_input_token_id': int(input_ids[0, -1]),
            'last_input_token_text': self.tokenizer.decode([input_ids[0, -1]]),
            'last_input_token_embedding': last_input_token_embeddings,
            'generated_text': generated_text,
            'generated_tokens': generated_tokens_only.cpu().tolist(),
            'generated_text_tokens': self.tokenizer.convert_ids_to_tokens(generated_tokens_only),
            'generated_tokens_embeddings': generated_tokens_embeddings,
            'layers': layer_indices,
            'layer_spec': layer_spec,
            'num_layers': num_layers,
            'hidden_size': outputs.hidden_states[0][0].shape[-1]
        }
        
        return result


def main():
    parser = argparse.ArgumentParser(
        description='Extract embeddings from LLM prompt and response',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to the pretrained model'
    )
    
    parser.add_argument(
        '--prompt',
        type=str,
        required=True,
        help='Input prompt text'
    )
    
    parser.add_argument(
        '--layer',
        type=str,
        default='mid',
        help="Layer(s) to extract embeddings from. Options: 'all', 'last', 'mid', or a specific layer number (default: 'mid')"
    )
    
    parser.add_argument(
        '--max_new_tokens',
        type=int,
        default=50,
        help='Maximum number of tokens to generate (default: 50)'
    )
    
    parser.add_argument(
        '--output_file',
        type=str,
        default=None,
        help='Path to save the embeddings as JSON. If not provided, prints summary to stdout'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help="Device to use for inference ('cuda' or 'cpu'). If not provided, auto-detects"
    )
    
    args = parser.parse_args()
    
    # Create extractor
    extractor = EmbeddingExtractor(args.model_path, device=args.device)
    
    # Extract embeddings
    result = extractor.extract_embeddings(
        prompt=args.prompt,
        layer_spec=args.layer,
        max_new_tokens=args.max_new_tokens
    )
    
    # Output results
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Embeddings saved to {args.output_file}")
    else:
        # Print summary instead of full embeddings
        print("\n" + "="*80)
        print("EXTRACTION SUMMARY")
        print("="*80)
        print(f"Model: {args.model_path}")
        print(f"Layers extracted: {result['layers']}")
        print(f"Hidden size: {result['hidden_size']}")
        print(f"\nInput prompt: {result['input_prompt']}")
        print(f"Input tokens count: {len(result['input_tokens'])}")
        print(f"Last input token: {result['last_input_token_text']} (ID: {result['last_input_token_id']})")
        print(f"\nGenerated text: {result['generated_text']}")
        print(f"Generated tokens count: {len(result['generated_tokens'])}")
        print(f"\nEmbedding dimensions per layer: {result['hidden_size']}")
        print(f"\nLast input token embedding shape: {len(result['last_input_token_embedding'])} layer(s)")
        print(f"Generated tokens embeddings: {len(result['generated_tokens_embeddings'])} token(s)")
        
        print("\n" + "="*80)
        print("To save full embeddings to a file, use --output_file option")
        print("="*80)


if __name__ == '__main__':
    main()
