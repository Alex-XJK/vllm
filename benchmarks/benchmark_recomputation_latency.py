"""
Benchmark the re-computation latency of processing a single batch of requests.
Author: Alex
"""
import argparse
import math
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm

from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import DEVICE_OPTIONS, EngineArgs
from vllm.inputs import PromptType
from vllm.utils import FlexibleArgumentParser

CACHE_SIZE_PER_TOKEN = 131072 # Determined by the model
CHUNK_SIZE_LOG_MIN = 8 # Arbitrary
TOKEN_SIZE_LOG_MIN = 8 # Arbitrary, but should be at least chunk size min
TOKEN_SIZE_LOG_MAX = 17  # Determined by number of GPU blocks (~ GPU HBM size).
MAX_MODEL_TOKENS = 65536 # Should have been 131072 but we truncate to 65536 otherwise it throws a CUDA error
NUM_PASSES = 4

@dataclass
class BenchmarkDim:
    max_seq_len: int
    batch_size: int
    chunk_size: int

    def __str__(self):
        return f"max_seq_len={self.max_seq_len}, batch_size={self.batch_size}, chunk_size={self.chunk_size}"


DEBUG_MODE = True
"""
Print a debug message if DEBUG_MODE is enabled.
"""
def debug_print(prompt: str):
    if DEBUG_MODE:
        print(f"DEBUG >> {prompt}")


"""
Generate dummy prompts for the re-computation latency benchmark.
"""
def generate_dummy_prompts(batch_size: int, input_len: int) -> List[PromptType]:
    dummy_prompt_token_ids = np.random.randint(10000, size=(batch_size, input_len))
    return [{"prompt_token_ids": batch} for batch in dummy_prompt_token_ids.tolist()]


"""
Generate benchmark dimensions for the re-computation latency benchmark.
Author: Schwinn
"""
def generate_benchmark_dims() -> List[BenchmarkDim]:
    benchmark_dims = []

    token_count_logs = torch.linspace(start=TOKEN_SIZE_LOG_MIN, end=TOKEN_SIZE_LOG_MAX, steps=TOKEN_SIZE_LOG_MAX-TOKEN_SIZE_LOG_MIN+1, dtype=int).tolist()
    token_count_logs = reversed(token_count_logs)

    for token_count_log in token_count_logs:
        token_count = int(2 ** token_count_log)

        chunk_size_log_max = min(token_count_log, int(math.log2(MAX_MODEL_TOKENS)))
        chunk_size_logs = torch.linspace(start=CHUNK_SIZE_LOG_MIN, end=chunk_size_log_max, steps=chunk_size_log_max-CHUNK_SIZE_LOG_MIN+1, dtype=int).tolist()
        chunk_size_logs = reversed(chunk_size_logs)
        for chunk_size_log in chunk_size_logs:
            chunk_size = int(2 ** chunk_size_log)

            batch_size_log_lo = max(token_count_log - int(math.log2(MAX_MODEL_TOKENS)), 0)
            batch_size_log_hi = token_count_log - chunk_size_log
            batch_sizes = torch.logspace(start=batch_size_log_lo, end=batch_size_log_hi, steps=batch_size_log_hi-batch_size_log_lo+1, base=2, dtype=int).tolist()

            for batch_size in batch_sizes:
                max_seq_len = token_count // batch_size
                benchmark_dims.append(BenchmarkDim(max_seq_len, batch_size, chunk_size))

    # Debug
    debug_print(f"Generated {len(benchmark_dims)} benchmark dimensions.")
    for benchmark_dim in benchmark_dims:
        debug_print(f"{benchmark_dim}")

    return benchmark_dims


def time_run_to_completion(llm: LLM, prompts: List[PromptType], sampling_params: SamplingParams):
    start_time = time.perf_counter()
    llm.generate(prompts,
                 sampling_params=sampling_params,
                 use_tqdm=False)
    end_time = time.perf_counter()
    latency = end_time - start_time
    return latency


def main(args: argparse.Namespace):
    debug_print(f"Running benchmark with args: {args}")

    benchmark_dimensions = generate_benchmark_dims()

    for benchmark_dim in benchmark_dimensions:
        print(f"INFO >> Running benchmark with dimension:")
        print(f"INFO >> {benchmark_dim}")
        print(f"INFO >> ")

        assert benchmark_dim.max_seq_len % benchmark_dim.chunk_size == 0
        num_chunked_prefill_iters = benchmark_dim.max_seq_len // benchmark_dim.chunk_size

        # NOTE(woosuk): If the request cannot be processed in a single batch,
        # the engine will automatically process the request in multiple batches.
        llm = LLM(
            model=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            trust_remote_code=args.trust_remote_code,
            enforce_eager=args.enforce_eager,
            kv_cache_dtype=args.kv_cache_dtype,
            device=args.device,
            ray_workers_use_nsight=args.ray_workers_use_nsight,
            use_v2_block_manager=args.use_v2_block_manager,
            enable_chunked_prefill=args.enable_chunked_prefill,
            download_dir=args.download_dir,
            block_size=args.block_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enable_prefix_caching=args.enable_prefix_caching,
        )

        sampling_params = SamplingParams(
            n=args.n,
            temperature=1.0,
            top_p=1.0,
            ignore_eos=True,
            max_tokens=args.output_len,
        )
        debug_print(f"Sampling params: {sampling_params}")

        # Generate dummy prompts.
        dummy_prompts = generate_dummy_prompts(benchmark_dim.batch_size, benchmark_dim.max_seq_len)

        print(f"INFO >> Warming up...")
        for _ in tqdm(range(args.num_iters_warmup), desc="Warmup iterations"):
            time_run_to_completion(llm, dummy_prompts, sampling_params)

        # Core Benchmark.
        latencies = []
        for _ in tqdm(range(args.num_iters), desc="Profiling iterations"):
            latencies.append(time_run_to_completion(llm, dummy_prompts, sampling_params))

    # Report statistics.
    latencies = np.array(latencies)
    percentages = [10, 25, 50, 75, 90, 99]
    percentiles = np.percentile(latencies, percentages)
    print(f'INFO >> Avg latency: {np.mean(latencies)} seconds')
    for percentage, percentile in zip(percentages, percentiles):
        print(f'INFO >> {percentage}% percentile latency: {percentile} seconds')



if __name__ == '__main__':
    parser = FlexibleArgumentParser(
        description='Benchmark the latency of processing a single batch of '
        'requests till completion.')
    parser.add_argument('--model', type=str, default='facebook/opt-125m')
    parser.add_argument('--tensor-parallel-size', '-tp', type=int, default=1)
    parser.add_argument('--input-len', type=int, default=32)
    parser.add_argument('--output-len', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--n',
                        type=int,
                        default=1,
                        help='Number of generated sequences per prompt.')
    parser.add_argument('--use-beam-search', action='store_true')
    parser.add_argument('--num-iters-warmup',
                        type=int,
                        default=10,
                        help='Number of iterations to run for warmup.')
    parser.add_argument('--num-iters',
                        type=int,
                        default=30,
                        help='Number of iterations to run.')
    parser.add_argument('--trust-remote-code',
                        action='store_true',
                        help='trust remote code from huggingface')
    parser.add_argument('--enforce-eager',
                        action='store_true',
                        help='enforce eager mode and disable CUDA graph')
    parser.add_argument(
        '--kv-cache-dtype',
        type=str,
        choices=['auto', 'fp8', 'fp8_e5m2', 'fp8_e4m3'],
        default="auto",
        help='Data type for kv cache storage. If "auto", will use model '
        'data type. CUDA 11.8+ supports fp8 (=fp8_e4m3) and fp8_e5m2. '
        'ROCm (AMD GPU) supports fp8 (=fp8_e4m3)')
    parser.add_argument("--device",
                        type=str,
                        default="auto",
                        choices=DEVICE_OPTIONS,
                        help='device type for vLLM execution')
    parser.add_argument('--block-size',
                        type=int,
                        default=16,
                        help='block size of key/value cache')
    parser.add_argument(
        '--enable-chunked-prefill',
        action='store_true',
        help='If True, the prefill requests can be chunked based on the '
        'max_num_batched_tokens')
    parser.add_argument("--enable-prefix-caching",
                        action='store_true',
                        help="Enable automatic prefix caching")
    parser.add_argument('--use-v2-block-manager', action='store_true')
    parser.add_argument(
        "--ray-workers-use-nsight",
        action='store_true',
        help="If specified, use nsight to profile ray workers",
    )
    parser.add_argument('--download-dir',
                        type=str,
                        default=None,
                        help='directory to download and load the weights, '
                        'default to the default cache dir of huggingface')
    parser.add_argument('--gpu-memory-utilization',
                        type=float,
                        default=0.9,
                        help='the fraction of GPU memory to be used for '
                        'the model executor, which can range from 0 to 1.'
                        'If unspecified, will use the default value of 0.9.')
    args = parser.parse_args()
    main(args)
