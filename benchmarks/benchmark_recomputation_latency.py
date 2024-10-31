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
import sys
from tqdm import tqdm

from vllm import SamplingParams, LLMEngine, TokensPrompt
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.metrics import LoggingStatLogger, PrometheusStatLogger
from vllm.inputs import PromptType
from vllm.outputs import RequestOutput
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


def stat_memory_now() -> tuple:
    """
    Get the current CUDA memory usage of allocated and reserved memory in bytes.
    """
    return torch.cuda.memory_allocated(), torch.cuda.memory_reserved()


def bytes_to_gb(bytes: int) -> float:
    return bytes / 1024 / 1024 / 1024


def parse_output(output: RequestOutput) -> float:
    """
    Parse the output of a request.
    """
    metrics = output.metrics
    arrival_time = metrics.arrival_time
    first_scheduled_time = metrics.first_scheduled_time if metrics.first_scheduled_time is not None else 0
    first_token_time = metrics.first_token_time if metrics.first_token_time is not None else 0
    ftt_fst = first_token_time - first_scheduled_time
    ftt_fst = ftt_fst if ftt_fst > 0 else 0
    ftt_arr = first_token_time - arrival_time
    ftt_arr = ftt_arr if ftt_arr > 0 else 0
    print(f"INFO >> TTFT: {ftt_arr}")
    print(f"INFO >> First token time - First scheduled time: {ftt_fst}")
    return ftt_fst


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

            assert benchmark_dim.max_seq_len % benchmark_dim.chunk_size == 0
            num_chunked_prefill_iters = benchmark_dim.max_seq_len // benchmark_dim.chunk_size

            print(f"INFO >> Running benchmark with dimension:")
            print(f"INFO >> ===== {benchmark_dim} =====")

            alogger = LoggingStatLogger(0.01)
            logger_dict = {"logging": alogger}

            mem_aloc_init, mem_resv_init = stat_memory_now()

            engine_args = EngineArgs(
                model=args.model,
                disable_log_stats=False,
                max_num_seqs=benchmark_dim.batch_size,
                max_num_batched_tokens=benchmark_dim.chunk_size * benchmark_dim.batch_size,
                preemption_mode=args.preemption_mode,
                enable_chunked_prefill=True,
            )
            my_engine = LLMEngine.from_engine_args(engine_args=engine_args, stat_loggers=logger_dict)

            sampling_params = SamplingParams(temperature=0, max_tokens=benchmark_dim.max_seq_len)

            mem_aloc_built, mem_resv_built = stat_memory_now()

            print(f"INFO >> Creating {2 * benchmark_dim.batch_size} sequences of length {benchmark_dim.max_seq_len}...")
            time_p0_s = time.perf_counter_ns()
            for i in range(2 * benchmark_dim.batch_size):
                prompt_token_ids = TokensPrompt(prompt_token_ids=list(range(benchmark_dim.max_seq_len)))
                my_engine.add_request(
                    request_id=str(i),
                    prompt=prompt_token_ids,
                    params=sampling_params,
                )
            time_p0_e = time.perf_counter_ns()

            mem_aloc_filled, mem_resv_filled = stat_memory_now()

            print(f"INFO >> Warming up...")
            time_p1_s = time.perf_counter_ns()
            my_engine.step()
            time_p1_e = time.perf_counter_ns()

            print(f"INFO >> Running step...")
            time_p2_s = time.perf_counter_ns()
            outputs = my_engine.step()
            time_p2_e = time.perf_counter_ns()

            mem_aloc_step, mem_resv_step = stat_memory_now()

            print(f"INFO >> {len(outputs)} outputs received.")
            prefill_t = parse_output(outputs[0])

            del my_engine
            gc.collect()
            torch.cuda.empty_cache()

            mem_aloc_clean, mem_resv_clean = stat_memory_now()

            p0_time = (time_p0_e - time_p0_s) / 1e9
            p1_time = (time_p1_e - time_p1_s) / 1e9
            p2_time = (time_p2_e - time_p2_s) / 1e9

            print(f"+==================== Benchmark completed ====================")
            print(f"|===== Dimension: {benchmark_dim}")
            print(f"|===== Latency (timer): p0 {p0_time:.6f} s, p1 {p1_time:.6f} s, p2 {p2_time:.6f} s")
            print(f"|===== Latency (computed): {prefill_t:.6f} s")
            print(f"|===== Memory Usage:")
            print(f"|===== + {'-' * 10} + {'Aloc':>5} + {'Resv':>5} +")
            print(f"|===== | {'Init':<10} | {bytes_to_gb(mem_aloc_init):>5.2f} | {bytes_to_gb(mem_resv_init):>5.2f} |")
            print(f"|===== | {'Built':<10} | {bytes_to_gb(mem_aloc_built):>5.2f} | {bytes_to_gb(mem_resv_built):>5.2f} |")
            print(f"|===== | {'Filled':<10} | {bytes_to_gb(mem_aloc_filled):>5.2f} | {bytes_to_gb(mem_resv_filled):>5.2f} |")
            print(f"|===== | {'Step':<10} | {bytes_to_gb(mem_aloc_step):>5.2f} | {bytes_to_gb(mem_resv_step):>5.2f} |")
            print(f"|===== | {'Clean':<10} | {bytes_to_gb(mem_aloc_clean):>5.2f} | {bytes_to_gb(mem_resv_clean):>5.2f} |")
            print(f"|===== + {'-' * 10} + {'-' * 5} + {'GB':>5} +")
            print(f"+=============================================================")

            benchmark_result = {
                'max_seq_len': benchmark_dim.max_seq_len,
                'batch_size': benchmark_dim.batch_size,
                'chunk_size': benchmark_dim.chunk_size,
                'computed_prefill_sec': prefill_t,
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
