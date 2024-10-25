"""
Try to load a model and print its configuration.
"""
import math
import torch
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

class MC_Reader:
    """
    MC_Reader: Model Configuration Reader
    Load a model and interpret its configuration.
    """
    def __init__(self, model_name: str):
        self.model = AutoModel.from_pretrained(model_name)
        self.config = self.model.config
        self.prec_bytes = self._precision_in_bytes()
        self.token_size = self._compute_token_size()

    def _precision_in_bytes(self):
        """
        Return the precision in bytes.
        """
        if self.config.torch_dtype == "bfloat16" or self.config.torch_dtype == torch.bfloat16:
            return 2
        elif self.config.torch_dtype == "float16" or self.config.torch_dtype == torch.float16:
            return 2
        elif self.config.torch_dtype == "float32" or self.config.torch_dtype == torch.float32:
            return 4
        else:
            self.print_manual_prompt()
            s = int(input("Please enter the precision in bytes: "))
            return s

    def _compute_token_size(self):
        """
        We follow the formula:
        KV-Cache per token = 2 * (num_layers) * (num_heads * dim_head) *  precision_in_bytes
        from the article:
        https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/
        """
        try:
          num_layers = self.config.num_hidden_layers
          num_heads = self.config.num_key_value_heads
          dim_head = self.config.head_dim
          self.token_size = 2 * num_layers * num_heads * dim_head * self.prec_bytes
        except:
          self.print_manual_prompt()
          print(f"If either of B or C item not known, pleae enter 0:")
          num_layers = int(input("A1: Please enter the number of layers: "))
          num_heads = int(input("B1: Please enter the number of heads: "))
          dim_head = int(input("B2: Please enter the dimension of head: "))
          hidden_size = int(input("C1: Please enter the hidden size: "))
          if hidden_size > 0 or num_heads == 0 or dim_head == 0:
              self.token_size = 2 * num_layers * hidden_size * self.prec_bytes
          else:
              self.token_size = 2 * num_layers * num_heads * dim_head * self.prec_bytes
        return self.token_size

    def per_token_size(self) -> int:
        """
        Return the size of onr KV-Cache token in bytes.
        """
        return self.token_size

    def N_token_size_in_GB(self, N: int) -> float:
        """
        Return the size of N KV-Cache tokens in GB.
        """
        return self.per_token_size() * N / 1024**3

    def compute_model_weight(self, weight_repr: str) -> int:
        """
        Compute the model weight.
        """
        suffix = weight_repr[-1].lower()
        number = int(weight_repr[:-1])
        if suffix == 'b':
            return number * 1e9 * self.prec_bytes
        elif suffix == 'm':
            return number * 1e6 * self.prec_bytes
        else:
            self.print_manual_prompt()
            num = int(input("Please enter the weight of the model value: "))
            return num * self.prec_bytes


    def device_capacity(self, GPU_memory_in_GB: int) -> int:
        """
        Compute the maximum number of tokens that can be stored in the GPU memory.
        """
        return math.floor(GPU_memory_in_GB * 1024**3 / self.per_token_size())

    def print_manual_prompt(self):
        print(f"Sorry, we cannot find the model information automatically.")
        print(f"Here is all the model information we know, please enter the parameters manually:")
        print(self.config)

    def __str__(self):
        return self.config


if __name__ == '__main__':
    parser = FlexibleArgumentParser(
        description='Load a model and compute its configuration.')
    parser.add_argument(
        '--model',
        type=str,
        default="meta-llama/Llama-3.1-8B",
        help='Name or path of the huggingface model to use.')
    parser.add_argument(
        '--weight',
        type=str,
        default="8B",
        help='Weight of the model, string should end with "B" or "M".')
    parser.add_argument(
        '--GPU-memory-in-GB',
        type=int,
        default=80,
        help='GPU memory in GB.')
    args = parser.parse_args()

    # Load the model and print its configuration.
    mcr = MC_Reader(args.model)
    print(f"Precision in bytes: {mcr.prec_bytes}")
    print(f"Size of one KV-Cache token: {mcr.per_token_size()} bytes")
    print(f"Size of 1,000 KV-Cache tokens: {mcr.N_token_size_in_GB(1000):.4f} GB")
    print(f"Size of 16,000 KV-Cache tokens: {mcr.N_token_size_in_GB(16000):.4f} GB")
    print(f"Size of 128,000 KV-Cache tokens: {mcr.N_token_size_in_GB(128000):.4f} GB")
    print(f"{'='*50}")
    model_weight_in_GB = mcr.compute_model_weight(args.weight) / 1024**3
    print(f"A {args.weight} Model takes about {model_weight_in_GB:.2f} GB on GPU.")
    remaining_space_in_GB = args.GPU_memory_in_GB - model_weight_in_GB
    print(f"Your {args.GPU_memory_in_GB} GB GPU still remainings {remaining_space_in_GB:.2f} GB after loading the model.")
    print(f"A maximum of {mcr.device_capacity(remaining_space_in_GB)} tokens can be stored in {remaining_space_in_GB:.2f} GB")


