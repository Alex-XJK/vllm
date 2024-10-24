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

from prometheus_client import start_http_server

from vllm import SamplingParams, LLMEngine, TokensPrompt
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.metrics import LoggingStatLogger, PrometheusStatLogger
from vllm.engine.multiprocessing.client import logger
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
def debug_print(prompt: str):
    """
    Print a debug message if DEBUG_MODE is enabled.
    """
    if DEBUG_MODE:
        print(f"DEBUG >> {prompt}")


def generate_dummy_prompts(batch_size: int, input_len: int) -> List[PromptType]:
    """
    Generate dummy prompts for the re-computation latency benchmark.
    """
    dummy_prompt_token_ids = np.random.randint(10000, size=(batch_size, input_len))
    return [{"prompt_token_ids": batch} for batch in dummy_prompt_token_ids.tolist()]


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

    # Debug
    debug_print(f"Generated {len(benchmark_dims)} benchmark dimensions.")
    for benchmark_dim in benchmark_dims:
        debug_print(f"{benchmark_dim}")

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



def mainloop_traditional(engine: LLMEngine, repetitions: int, num_iters: int) -> tuple:
    """
    Main loop for the benchmarking.
    Use Schwinn's originnal coding logic.
    """
    latencies = []

    total_iters = repetitions * num_iters

    start_all = time.perf_counter_ns()
    start = time.perf_counter_ns()

    for i in range(total_iters):
        outputs = engine.step()

        if i % num_iters == num_iters - 1:
            end = time.perf_counter_ns()
            latencies.append((end - start) / 1e6)
            print(f"{i + 1:8d}/{total_iters} ::\t Recomputation of whole batch took: {latencies[-1]} ms")
            start = time.perf_counter_ns()

    end_all = time.perf_counter_ns()
    mean_latency_all_div = (end_all - start_all) / (1e6 * repetitions)

    return latencies, mean_latency_all_div


def mainloop_beautify(engine: LLMEngine, repetitions: int, num_iters: int) -> tuple:
    """
    Main loop for the benchmarking.
    Use my own logic and tqdm for progress bar. (may add extra IO overhead)
    """
    latencies = []

    start_all = time.perf_counter_ns()

    for i in range(repetitions):
        start = time.perf_counter_ns()

        # Critical part
        for _ in tqdm(range(num_iters), desc=f"Pass {i+1}/{repetitions}"):
            outputs = engine.step()

        end = time.perf_counter_ns()
        print(f"INFO >> Pass {i+1}/{repetitions} :: Recomputation of whole batch took: {(end - start) / 1e6} ms")
        latencies.append((end - start) / 1e6)

    end_all = time.perf_counter_ns()
    mean_latency_all_div = (end_all - start_all) / (1e6 * repetitions)

    return latencies, mean_latency_all_div


def mainloop_profiling(engine: LLMEngine, num_iters: int, dim: BenchmarkDim) -> tuple:
    """
    Main loop for the benchmarking.
    Use profiler to profile the CUDA usage as well.
    """
    profile_dir = Path(".") / "vllm_benchmark_result" / f"benchmark_re_{dim.max_seq_len}_{dim.batch_size}_{dim.chunk_size}_{time.time()}"

    print(f"INFO >> Profiling enabled. Run only ONE pass.")

    latencies = []
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        profile_memory=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(profile_dir))
        ) as p:

        start_all = time.perf_counter_ns()
        start = time.perf_counter_ns()

        # Critical part
        for _ in range(num_iters):
            outputs = engine.step()

        end = time.perf_counter_ns()
        end_all = time.perf_counter_ns()

    print(f"INFO >> Pass 1/1 :: Recomputation of whole batch took: {(end - start) / 1e6} ms")
    print(f"INFO >> Profiling results will be saved to '{profile_dir}'...")

    latencies.append((end - start) / 1e6)

    mean_latency_all_div = (end_all - start_all) / 1e6

    return latencies, mean_latency_all_div


def config_global(args: argparse.Namespace):
    global CACHE_SIZE_PER_TOKEN
    CACHE_SIZE_PER_TOKEN = args.cache_size_per_token
    global NUM_PASSES
    NUM_PASSES = args.repetitions


def main(args: argparse.Namespace):
    debug_print(f"Running benchmark with args: {args}")

    config_global(args)

    benchmark_dimensions = []
    if args.manual is not None:
        benchmark_dimensions = manual_benchmark_dims(args.manual)
    else:
        benchmark_dimensions = generate_benchmark_dims()
    benchmark_results = []

    csv_path = args.file_out
    if csv_path is None:
        pid = os.getpid()
        csv_path = f"prefill_latency_profiling_{pid}.csv"

    with open(csv_path, mode='a', newline='') as f:
        print(f"INFO >> Writing intermediate results to '{csv_path}'...")

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

            # Try to use the vllm built-in logger.
            isProfiling = args.run_mode == "profiling"
            alogger = LoggingStatLogger(0.001)
            plogger = PrometheusStatLogger(0.01, dict(model_name=args.model), benchmark_dim.max_seq_len)
            loggerDict = {"logging": alogger, "prometheus": plogger}

            engine_args = EngineArgs(
                model=args.model,
                disable_log_stats=not isProfiling, # For logger
                max_num_seqs=benchmark_dim.batch_size,
                max_num_batched_tokens=benchmark_dim.chunk_size * benchmark_dim.batch_size,
                preemption_mode=args.preemption_mode,
                enable_chunked_prefill=True,
            )
            my_engine = LLMEngine.from_engine_args(engine_args=engine_args, stat_loggers=loggerDict) # For logger

            # dummy_prompts = generate_dummy_prompts(2 * benchmark_dim.batch_size, benchmark_dim.max_seq_len)
            sampling_params = SamplingParams(temperature=0, max_tokens=benchmark_dim.max_seq_len)

            print(f"INFO >> Creating {2 * benchmark_dim.batch_size} sequences of length {benchmark_dim.max_seq_len}...")
            print(f"INFO >> ")
            for i in tqdm(range(2 * benchmark_dim.batch_size), desc="Adding requests"):
                prompt_token_ids = TokensPrompt(prompt_token_ids=list(range(benchmark_dim.max_seq_len)))
                my_engine.add_request(
                    request_id=str(i),
                    prompt=prompt_token_ids,
                    params=sampling_params,
                )

            print(f"INFO >> Warming up...")
            print(f"INFO >> ")
            for _ in tqdm(range(num_chunked_prefill_iters), desc="Warmup iterations"):
                my_engine.step()

            print(f"INFO >> Profiling iterations... will run {NUM_PASSES} * {num_chunked_prefill_iters} iterations.")
            print(f"INFO >> ")

            if args.run_mode == "beautify":
                latencies, mean_latency_all_div = mainloop_beautify(my_engine, NUM_PASSES, num_chunked_prefill_iters)
            elif args.run_mode == "profiling":
                latencies, mean_latency_all_div = mainloop_profiling(my_engine, num_chunked_prefill_iters, benchmark_dim)
            else:
                latencies, mean_latency_all_div = mainloop_traditional(my_engine, NUM_PASSES, num_chunked_prefill_iters)


            # LLMEngine seems doesn't have a terminate method as it does in Sarathi
            # Terminate the engine and clean up CUDA memory.
            # Otherwise, the CUDA memory usage will keep increasing, and out-of-memory soon in 2nd iteration.
            del my_engine
            gc.collect()
            torch.cuda.empty_cache()

            mean_latency = torch.mean(torch.tensor(latencies)).item()
            std_latency = torch.std(torch.tensor(latencies)).item()
            print(f"Mean latency: {mean_latency} ms, "
                f"std latency: {std_latency} ms, "
                f"mean latency (all divided by num passes): {mean_latency_all_div}"
                )

            benchmark_result = {
                'chunk_size': benchmark_dim.chunk_size,
                'token_count': benchmark_dim.max_seq_len * benchmark_dim.batch_size,
                'batch_size': benchmark_dim.batch_size,
                'max_seq_len': benchmark_dim.max_seq_len,
                'mean_latency': mean_latency,
                'std_latency': std_latency,
                'mean_latency_all_div': mean_latency_all_div,
                'kv_cache_size': CACHE_SIZE_PER_TOKEN * benchmark_dim.max_seq_len * benchmark_dim.batch_size
            }
            benchmark_results.append(benchmark_result)

            # Save the intermediate results to a CSV file.
            df = pd.DataFrame([benchmark_result])
            if f.tell() == 0:
                df.to_csv(f, index=False)
            else:
                df.to_csv(f, header=False, index=False)
            f.flush()

    df = pd.DataFrame(benchmark_results)
    print(f"INFO >> Writing final results to 'prefill_latency_profiling.csv'...")
    df.to_csv("prefill_latency_profiling.csv", index=False)


if __name__ == '__main__':
    # For Prometheus server, refresh frequency is not very satisfactory.
    start_http_server(8000)

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
        '--run-mode',
        type=str,
        choices=['traditional', 'beautify', 'profiling'],
        default="beautify",
        help='If \'traditional\', the benchmark will run with Schwinn\'s original logic; '
                'If \'beautify\', the benchmark will run with progress bar and my own logic.'
                'If \'profiling\', the benchmark will run with profiler enabled.')
    parser.add_argument(
        '--repetitions',
        type=int,
        default=NUM_PASSES,
        help='Number of repetitions for the benchmark.')
    parser.add_argument(
        '--manual',
        type=str,
        default=None,
        help='Manual mode, if specified, will use the specified benchmark dimensions.'
             'Otherwise, will generate benchmark dimensions automatically.'
             'The format should be "max_seq_len,batch_size,chunk_size".')
    parser.add_argument(
        '--file-out',
        type=str,
        default=None,
        help='Output file for the benchmark, if not specified, will generate one.')
    args = parser.parse_args()
    main(args)
