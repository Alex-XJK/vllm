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
import pandas as pd
import torch
from tqdm import tqdm

from vllm import LLM, SamplingParams, LLMEngine
from vllm.config import SchedulerConfig
from vllm.core.scheduler import PreemptionMode
from vllm.engine.arg_utils import DEVICE_OPTIONS, EngineArgs
from vllm.entrypoints.api_server import engine
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


"""
Run the benchmark to completion and return the latency.
Deprecated: Use used by the old benchmarking code.
"""
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
    benchmark_results = []

    for benchmark_dim in benchmark_dimensions:
        print(f"INFO >> Running benchmark with dimension:")
        print(f"INFO >> {benchmark_dim}")
        print(f"INFO >> ")

        assert benchmark_dim.max_seq_len % benchmark_dim.chunk_size == 0
        num_chunked_prefill_iters = benchmark_dim.max_seq_len // benchmark_dim.chunk_size

        # TODO: What is a "chunk_size" in sarathi?
        # scheduler_config = SchedulerConfig(
        #     max_num_seqs=benchmark_dim.batch_size,
        #     max_model_len=benchmark_dim.max_seq_len * benchmark_dim.batch_size,
        #     # chunk_size=benchmark_dim.chunk_size,
        #     max_num_batched_tokens=benchmark_dim.max_seq_len * benchmark_dim.batch_size,
        #     preemption_mode="recompute",
        #     enable_chunked_prefill=True,
        # )

        engine_args = EngineArgs(
            max_num_seqs=benchmark_dim.batch_size,
            max_model_len=benchmark_dim.max_seq_len * benchmark_dim.batch_size,
            max_num_batched_tokens=benchmark_dim.max_seq_len * benchmark_dim.batch_size,
            preemption_mode="recompute",
            enable_chunked_prefill=True,
        )
        my_engine = LLMEngine.from_engine_args(engine_args)

        dummy_prompts = generate_dummy_prompts(2 * benchmark_dim.batch_size, benchmark_dim.max_seq_len)
        sampling_params = SamplingParams(temperature=0, max_tokens=benchmark_dim.max_seq_len)

        print(f"INFO >> Creating {2 * benchmark_dim.batch_size} sequences of length {benchmark_dim.max_seq_len}...")
        for i in tqdm(range(2 * benchmark_dim.batch_size), desc="Adding requests"):
            my_engine.add_request(
                request_id=str(i),
                prompt=dummy_prompts[i],
                params=sampling_params,
            )

        print(f"INFO >> Warming up...")
        for _ in tqdm(range(num_chunked_prefill_iters), desc="Warmup iterations"):
            engine.step()

        print(f"INFO >> Profiling iterations...")
        latencies = []

        total_iters = NUM_PASSES * num_chunked_prefill_iters

        start_all = time.perf_counter_ns()
        start = time.perf_counter_ns()

        for i in range(total_iters):
            outputs = engine.step()

            if i % num_chunked_prefill_iters == num_chunked_prefill_iters - 1:
                end = time.perf_counter_ns()
                latencies.append((end - start) / 1e6)
                print(f"{i:8d}/{total_iters} ::\t Recomputation of whole batch took: {latencies[-1]} ms")
                start = time.perf_counter_ns()

        end_all = time.perf_counter_ns()

        engine.terminate()

        # Report statistics.
        mean_latency_all_div = (end_all - start_all) / (1e6 * NUM_PASSES)

        mean_latency = torch.mean(torch.tensor(latencies)).item()
        std_latency = torch.std(torch.tensor(latencies)).item()
        print(f"Mean latency: {mean_latency} ms, "
              f"std latency: {std_latency} ms, "
              f"mean latency (all divided by num passes): {mean_latency_all_div}"
              )

        engine.terminate()

        benchmark_results.append({
            'chunk_size': benchmark_dim.chunk_size,
            'token_count': benchmark_dim.max_seq_len * benchmark_dim.batch_size,
            'batch_size': benchmark_dim.batch_size,
            'max_seq_len': benchmark_dim.max_seq_len,
            'mean_latency': mean_latency,
            'std_latency': std_latency,
            'mean_latency_all_div': mean_latency_all_div,
            'kv_cache_size': CACHE_SIZE_PER_TOKEN * benchmark_dim.max_seq_len * benchmark_dim.batch_size
        })

    df = pd.DataFrame(benchmark_results)
    df.to_csv("prefill_latency_profiling.csv", index=False)



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
