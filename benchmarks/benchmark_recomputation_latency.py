"""
Benchmark the re-computation latency of processing a single batch of requests.
Author: Alex
"""
import gc
import os
import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from vllm import SamplingParams, LLMEngine, TokensPrompt
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.metrics import LoggingStatLogger, PrometheusStatLogger
from vllm.inputs import PromptType
from vllm.utils import FlexibleArgumentParser

CACHE_SIZE_PER_TOKEN = 131072 # Determined by the model
CHUNK_SIZE_LOG_MIN = 8 # Arbitrary
TOKEN_SIZE_LOG_MIN = 8 # Arbitrary, but should be at least chunk size min
TOKEN_SIZE_LOG_MAX = 17  # Determined by number of GPU blocks (~ GPU HBM size).
MAX_MODEL_TOKENS = 65536 # Should have been 131072 but we truncate to 65536 otherwise it throws a CUDA error

@dataclass
class BenchmarkDim:
    max_seq_len: int
    batch_size: int
    chunk_size: int

    def __str__(self):
        return f"max_seq_len={self.max_seq_len}, batch_size={self.batch_size}, chunk_size={self.chunk_size}"


def generate_benchmark_dims() -> List[BenchmarkDim]:
    """
    Generate benchmark dimensions for the re-computation latency benchmark.
    Author: Schwinn
    """
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

    # Reverse the order of the benchmark dimensions, so that the smallest ones are tested first.
    benchmark_dims.reverse()
    return benchmark_dims


def manual_benchmark_dims(manual_str: str) -> List[BenchmarkDim]:
    """
    Parse the manual benchmark dimensions from the command line.
    The string should be in the format 'max_seq_len,batch_size,chunk_size'.
    """
    def validate_input(input):
        parts = input.split(',')
        if len(parts) != 3:
            return None
        x, y, z = map(int, parts)
        return x, y, z

    benchmark_dims = []
    while validate_input(manual_str) is None:
        manual_str = input("Please enter in the format 'max_seq_len,batch_size,chunk_size': ")

    max_seq_len, batch_size, chunk_size = validate_input(manual_str)
    benchmark_dims.append(BenchmarkDim(max_seq_len, batch_size, chunk_size))
    return benchmark_dims


def config_global(args: argparse.Namespace):
    global CACHE_SIZE_PER_TOKEN
    CACHE_SIZE_PER_TOKEN = args.cache_size_per_token


def main(args: argparse.Namespace):

    config_global(args)

    if args.manual is not None:
        benchmark_dimensions = manual_benchmark_dims(args.manual)
    else:
        benchmark_dimensions = generate_benchmark_dims()

    for benchmark_dim in benchmark_dimensions:

        if benchmark_dim.max_seq_len != benchmark_dim.chunk_size:
            continue

        print(f"INFO >> Running benchmark with dimension:")
        print(f"INFO >> ===== {benchmark_dim} =====")
        print(f"DEBUG >> In this special debugging mode, the benchmark will only run once, with no chunk size optimization.")

        # Try to use the vllm built-in logger.
        alogger = LoggingStatLogger(0.001)
        logger_dict = {"logging": alogger}

        engine_args = EngineArgs(
            model=args.model,
            disable_log_stats=False,
            max_num_seqs=benchmark_dim.batch_size,
            max_num_batched_tokens=benchmark_dim.chunk_size * benchmark_dim.batch_size,
            preemption_mode=args.preemption_mode,
            enable_chunked_prefill=True,
        )
        my_engine = LLMEngine.from_engine_args(engine_args=engine_args, stat_loggers=logger_dict) # For logger

        sampling_params = SamplingParams(temperature=0, max_tokens=benchmark_dim.max_seq_len)

        print(f"INFO >> Creating {2 * benchmark_dim.batch_size} sequences of length {benchmark_dim.max_seq_len}...")
        for i in range(2 * benchmark_dim.batch_size):
            prompt_token_ids = TokensPrompt(prompt_token_ids=list(range(benchmark_dim.max_seq_len)))
            my_engine.add_request(
                request_id=str(i),
                prompt=prompt_token_ids,
                params=sampling_params,
            )

        print(f"INFO >> Warming up...")
        my_engine.step()

        profile_dir = Path(".") / "vllm_benchmark_result" / f"benchmark_re_{benchmark_dim.max_seq_len}_{benchmark_dim.batch_size}_{benchmark_dim.chunk_size}_{time.time()}"

        with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                profile_memory=True,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(str(profile_dir))
        ) as p:

            time_start = time.perf_counter_ns()

            outputs = my_engine.step()

            time_end = time.perf_counter_ns()

        latency = (time_end - time_start) / 1e6

        del my_engine
        gc.collect()
        torch.cuda.empty_cache()

        print(f"Latency: {latency} ms")


if __name__ == '__main__':
    parser = FlexibleArgumentParser(
        description='Benchmark the re-computation latency of processing a single batch of requests.')
    parser.add_argument(
        '--model',
        type=str,
        default="meta-llama/Llama-3.1-8B",
        help='Name or path of the huggingface model to use.')
    parser.add_argument(
        '--preemption-mode',
        type=str,
        choices=['recompute', 'swap'],
        default="recompute",
        help='If \'recompute\', the engine performs preemption by '
             'recomputing; If \'swap\', the engine performs preemption by '
             'block swapping.')
    parser.add_argument(
        '--cache-size-per-token',
        type=int,
        default=CACHE_SIZE_PER_TOKEN,
        help='Size of the cache per token in bytes. Determined by the model.')
    parser.add_argument(
        '--manual',
        type=str,
        default=None,
        help='Manual mode, if specified, will use the specified benchmark dimensions.'
             'Otherwise, will generate benchmark dimensions automatically.'
             'The format should be "max_seq_len,batch_size,chunk_size".')
    args = parser.parse_args()
    main(args)
