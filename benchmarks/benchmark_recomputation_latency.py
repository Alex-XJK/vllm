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
from vllm.engine.arg_utils import EngineArgs
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
        #     chunk_size=benchmark_dim.chunk_size,
        #     max_num_batched_tokens=benchmark_dim.max_seq_len * benchmark_dim.batch_size,
        # )

        engine_args = EngineArgs(
            max_num_seqs=benchmark_dim.batch_size,
            # max_model_len=benchmark_dim.max_seq_len * benchmark_dim.batch_size,
            max_num_batched_tokens=benchmark_dim.max_seq_len * benchmark_dim.batch_size,
            preemption_mode=args.preemption_mode,
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
        description='Benchmark the re-computation latency of processing a single batch of requests.')
    parser.add_argument(
        '--preemption-mode',
        type=str,
        choices=['recompute', 'swap'],
        default="recompute",
        help='If \'recompute\', the engine performs preemption by '
             'recomputing; If \'swap\', the engine performs preemption by '
             'block swapping.')

    args = parser.parse_args()
    main(args)
