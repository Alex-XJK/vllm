"""
Benchmark the re-computation latency of processing a single batch of requests.
Author: Alex
"""
import argparse
import gc
import math
import os
import pandas as pd
import time
import torch

from dataclasses import dataclass
from tqdm import tqdm
from typing import List

from vllm import SamplingParams, LLMEngine, TokensPrompt
from vllm.engine.arg_utils import EngineArgs
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


def warmup_device(args: argparse.Namespace, num_warmup: int = 5):
    """
    Warm up the device by allocating and deallocating memory.
    """
    if not torch.cuda.is_available():
        print(f"ERROR >> CUDA is not available.")
        return

    device_name = torch.cuda.get_device_name()
    print(f"INFO >> Warming up {device_name}...")
    for _ in tqdm(range(num_warmup), desc="Warming up CUDA"):
        torch.ones(1).cuda()
        torch.cuda.empty_cache()

    for i in tqdm(range(num_warmup), desc="Warming up LLMEngine"):
        warmup_engine = LLMEngine.from_engine_args(
            engine_args=EngineArgs(model=args.model),
        )
        warmup_engine.add_request(
            request_id=str(i),
            prompt=TokensPrompt(prompt_token_ids=list(range(256))),
            params=SamplingParams(temperature=0, max_tokens=256),
        )
        warmup_engine.step()
        warmup_engine.abort_request(str(i))
        del warmup_engine
        torch.cuda.empty_cache()

    gc.collect()
    print(f"INFO >> Warmup completed.")


def main(args: argparse.Namespace):

    config_global(args)

    if args.manual is not None:
        benchmark_dimensions = manual_benchmark_dims(args.manual)
    else:
        benchmark_dimensions = generate_benchmark_dims()

    warmup_device(args)

    pid = os.getpid()
    csv_path = f"prefill_{pid}.csv"

    with open(csv_path, mode='a', newline='') as f:

        for benchmark_dim in benchmark_dimensions:

            if benchmark_dim.max_seq_len * benchmark_dim.batch_size > MAX_MODEL_TOKENS:
                print(f"WARN >> Skipping {benchmark_dim} due to exceeding the maximum token limit.")
                continue

            assert benchmark_dim.max_seq_len % benchmark_dim.chunk_size == 0
            num_chunked_prefill_iters = benchmark_dim.max_seq_len // benchmark_dim.chunk_size

            print(f"INFO >> Running benchmark with dimension:")
            print(f"INFO >> ===== {benchmark_dim} =====")
            print(f"INFO >> ===== Chunked Prefill Iterations: {num_chunked_prefill_iters} =====")

            engine_args = EngineArgs(
                model=args.model,
                disable_log_stats=False,
                max_num_seqs=benchmark_dim.batch_size,
                max_num_batched_tokens=benchmark_dim.chunk_size * benchmark_dim.batch_size,
                preemption_mode=args.preemption_mode,
                enable_chunked_prefill=True,
            )
            my_engine = LLMEngine.from_engine_args(engine_args=engine_args)

            sampling_params = SamplingParams(temperature=0, max_tokens=benchmark_dim.max_seq_len)

            print(f"INFO >> Creating {benchmark_dim.batch_size} sequences of length {benchmark_dim.max_seq_len}...")
            time_p0_s = time.perf_counter_ns()
            for i in range(benchmark_dim.batch_size):
                prompt_token_ids = TokensPrompt(prompt_token_ids=list(range(benchmark_dim.max_seq_len)))
                my_engine.add_request(
                    request_id=str(i),
                    prompt=prompt_token_ids,
                    params=sampling_params,
                )
            time_p0_e = time.perf_counter_ns()

            print(f"INFO >> Running 1st Step * {num_chunked_prefill_iters}...")
            time_p1_s = time.perf_counter_ns()
            for i in range(num_chunked_prefill_iters):
                my_engine.step()
            time_p1_e = time.perf_counter_ns()

            print(f"INFO >> Running 2nd Step...")
            time_p2_s = time.perf_counter_ns()
            outputs = my_engine.step()
            time_p2_e = time.perf_counter_ns()

            print(f"INFO >> {len(outputs)} outputs received.")

            del my_engine
            gc.collect()
            torch.cuda.empty_cache()

            p0_time = (time_p0_e - time_p0_s) / 1e9
            p1_time = (time_p1_e - time_p1_s) / 1e9
            p2_time = (time_p2_e - time_p2_s) / 1e9

            print(f"+==================== Benchmark completed ====================")
            print(f"|===== Dimension: {benchmark_dim}")
            print(f"|===== Latency P0: {p0_time:.4f} sec (add_request)")
            print(f"|===== Latency P1: {p1_time:.4f} sec (prefill)")
            print(f"|===== Latency P2: {p2_time:.4f} sec (1st decode step)")
            print(f"+=============================================================")

            benchmark_result = {
                'max_seq_len': benchmark_dim.max_seq_len,
                'batch_size': benchmark_dim.batch_size,
                'chunk_size': benchmark_dim.chunk_size,
                'p0_time_sec': p0_time,
                'p1_time_sec': p1_time,
                'p2_time_sec': p2_time,
            }
            df = pd.DataFrame([benchmark_result])
            if f.tell() == 0:
                df.to_csv(f, index=False)
            else:
                df.to_csv(f, header=False, index=False)
            f.flush()


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
