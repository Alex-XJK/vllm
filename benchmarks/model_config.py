"""
Try to load a model and print its configuration.
"""
from transformers import AutoModel
from vllm.utils import FlexibleArgumentParser

"""
Here is an sample output:

```json
LlamaConfig {
  "_name_or_path": "meta-llama/Llama-3.1-8B",
  "architectures": [
    "LlamaForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 128000,
  "eos_token_id": 128001,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 14336,
  "max_position_embeddings": 131072,
  "mlp_bias": false,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 8,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": {
    "factor": 8.0,
    "high_freq_factor": 4.0,
    "low_freq_factor": 1.0,
    "original_max_position_embeddings": 8192,
    "rope_type": "llama3"
  },
  "rope_theta": 500000.0,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.45.2",
  "use_cache": true,
  "vocab_size": 128256
}
```
"""

if __name__ == '__main__':
    parser = FlexibleArgumentParser(
        description='Load a model and compute its configuration.')
    parser.add_argument(
        '--model',
        type=str,
        default="meta-llama/Llama-3.1-8B",
        help='Name or path of the huggingface model to use.')
    args = parser.parse_args()
    model = AutoModel.from_pretrained(args.model)
    print(model.config)
