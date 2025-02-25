"""
Benchmark the re-computation latency of processing a single batch of requests with adaptive chunk sizes.
Modified from original to support adaptive chunking.
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
from typing import List, Dict, Tuple

from vllm import SamplingParams, LLMEngine, TokensPrompt
from vllm.engine.arg_utils import EngineArgs
from vllm.utils import FlexibleArgumentParser

CACHE_SIZE_PER_TOKEN = 131072  # Determined by the model
TOKEN_SIZE_LOG_MIN = 8  # Min sequence length (2^8 = 256)
TOKEN_SIZE_LOG_MAX = 17  # Max sequence length (2^17)
MAX_MODEL_TOKENS = 65536
MIN_CHUNK_SIZE = 256  # Minimum initial chunk size

def nearest_power_of_two(x):
    """Round a number to the nearest power of 2."""
    return 2 ** round(math.log2(x))

def generate_chunk_sizes(seq_len: int, initial_chunk_size: float) -> Tuple[List[int], bool]:
    """
    Generate a series of chunk sizes that sum to seq_len.
    
    Parameters:
        seq_len (int): The total sequence length
        initial_chunk_size (float): Initial chunk size (will be rounded to nearest power of 2)
    
    Returns:
        Tuple[List[int], bool]: List of chunk sizes that sum to seq_len and a boolean indicating if 
                               the final chunk is valid (>= MIN_CHUNK_SIZE)
    """
    # Round initial chunk size to nearest power of 2
    a1 = nearest_power_of_two(initial_chunk_size)
    a1 = min(max(MIN_CHUNK_SIZE, a1), seq_len // 2)  # Enforce bounds
    
    chunks = [a1]
    current_sum = a1
    
    while current_sum < seq_len:
        S_prev = current_sum
        # Solve quadratic equation: a_i^2 + (2*S_prev - 1)*a_i - a1*(a1-1) = 0
        B = 2 * S_prev - 1
        discriminant = B**2 + 4 * a1 * (a1 - 1)
        a_next = (-B + math.sqrt(discriminant)) / 2
        
        # Round to nearest power of 2
        a_next = nearest_power_of_two(a_next)
        
        # Check if this is the last chunk needed
        remaining_tokens = seq_len - current_sum
        if a_next >= remaining_tokens:
            # This will be our final chunk
            if remaining_tokens < MIN_CHUNK_SIZE:
                return chunks, False
            chunks.append(remaining_tokens)
            return chunks, True
            
        chunks.append(a_next)
        current_sum += a_next
    
    # If we reach here, the last chunk was exactly the remaining tokens
    return chunks, chunks[-1] >= MIN_CHUNK_SIZE

@dataclass
class BenchmarkDim:
    max_seq_len: int
    batch_size: int
    initial_chunk_size: int

    def __str__(self):
        return f"max_seq_len={self.max_seq_len}, batch_size={self.batch_size}, initial_chunk_size={self.initial_chunk_size}"

def generate_benchmark_dims() -> List[BenchmarkDim]:
    """
    Generate benchmark dimensions for the adaptive chunk size benchmark.
    Only includes dimensions where the final chunk size will be >= MIN_CHUNK_SIZE.
    """
    benchmark_dims = []
    
    token_count_logs = torch.linspace(start=TOKEN_SIZE_LOG_MIN, end=TOKEN_SIZE_LOG_MAX, 
                                    steps=TOKEN_SIZE_LOG_MAX-TOKEN_SIZE_LOG_MIN+1, dtype=int).tolist()
    
    for token_count_log in reversed(token_count_logs):
        seq_len = int(2 ** token_count_log)
        if seq_len > MAX_MODEL_TOKENS:
            continue
            
        # Generate different batch sizes
        max_total_tokens = min(MAX_MODEL_TOKENS, seq_len)
        max_batch_size = max_total_tokens // seq_len
        batch_sizes = [1]  # Start with batch size 1
        if max_batch_size > 1:
            batch_sizes.extend(
                torch.logspace(start=0, end=math.log2(max_batch_size), 
                             steps=min(5, int(math.log2(max_batch_size)) + 1), 
                             base=2, dtype=int).tolist()
            )
            batch_sizes = sorted(list(set(batch_sizes)))  # Remove duplicates
        print(f"Max total tokens: {max_total_tokens}, max batch size: {max_batch_size}, batch_sizes: {batch_sizes}")
        
        # Generate different initial chunk sizes
        min_chunk = MIN_CHUNK_SIZE
        max_chunk = seq_len // 2
        if min_chunk <= max_chunk:
            chunk_size_logs = range(int(math.log2(min_chunk)), int(math.log2(max_chunk)) + 1)
            for chunk_size_log in chunk_size_logs:
                initial_chunk_size = 2 ** chunk_size_log
                
                # Check if this initial chunk size produces valid chunk sizes
                chunks, is_valid = generate_chunk_sizes(seq_len, initial_chunk_size)
                if is_valid:
                    for batch_size in batch_sizes:
                        benchmark_dims.append(
                            BenchmarkDim(seq_len, batch_size, initial_chunk_size)
                        )
    
    # Sort by total computation size for progressive testing
    benchmark_dims.sort(key=lambda x: x.max_seq_len * x.batch_size)
    print(benchmark_dims)
    return benchmark_dims

def manual_benchmark_dims(manual_str: str) -> List[BenchmarkDim]:
    """Parse manual benchmark dimensions from command line."""
    def validate_input(input):
        parts = input.split(',')
        if len(parts) != 3:
            return None
        x, y, z = map(int, parts)
        return x, y, z

    while validate_input(manual_str) is None:
        manual_str = input("Please enter in the format 'max_seq_len,batch_size,initial_chunk_size': ")

    max_seq_len, batch_size, initial_chunk_size = validate_input(manual_str)
    return [BenchmarkDim(max_seq_len, batch_size, initial_chunk_size)]

def warmup_device(args: argparse.Namespace, num_warmup: int = 5):
    """Warm up the device by running some initial operations."""
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
            engine_args=EngineArgs(model=args.model, load_format="dummy"),
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
    if args.manual is not None:
        benchmark_dimensions = manual_benchmark_dims(args.manual)
    else:
        benchmark_dimensions = generate_benchmark_dims()

    warmup_device(args)

    pid = os.getpid()
    csv_path = f"adaptive_prefill_{pid}.csv"
    chunk_csv_path = f"adaptive_chunk_latency_{pid}.csv"

    with open(csv_path, mode='a', newline='') as f, open(chunk_csv_path, mode='a', newline='') as chunk_f:
        for benchmark_dim in benchmark_dimensions:
            if benchmark_dim.max_seq_len * benchmark_dim.batch_size > MAX_MODEL_TOKENS:
                print(f"WARN >> Skipping {benchmark_dim} due to exceeding the maximum token limit.")
                continue

            # Generate adaptive chunk sizes
            chunk_sizes, is_valid = generate_chunk_sizes(benchmark_dim.max_seq_len, benchmark_dim.initial_chunk_size)
            if not is_valid:
                print(f"WARN >> Skipping {benchmark_dim} due to invalid final chunk size")
                continue
            num_chunks = len(chunk_sizes)

            print(f"INFO >> Running benchmark with dimension:")
            print(f"INFO >> ===== {benchmark_dim} =====")
            print(f"INFO >> ===== Number of chunks: {num_chunks} =====")
            print(f"INFO >> ===== Chunk sizes: {chunk_sizes} =====")

            engine_args = EngineArgs(
                model=args.model,
                load_format="dummy",
                disable_log_stats=False,
                max_num_seqs=benchmark_dim.batch_size,
                max_num_batched_tokens=max(chunk_sizes) * benchmark_dim.batch_size,
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

            print(f"INFO >> Running chunked prefill steps...")
            torch.cuda.synchronize()
            time_p1_s = time.perf_counter_ns()

            start_events = []
            end_events = []
            for chunk_idx, chunk_size in enumerate(chunk_sizes):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_events.append(start_event)
                end_events.append(end_event)

                start_event.record()
                my_engine.step()
                end_event.record()

            torch.cuda.synchronize()
            time_p1_e = time.perf_counter_ns()

            chunk_latencies = []
            for start_event, end_event in zip(start_events, end_events):
                chunk_latencies.append(start_event.elapsed_time(end_event) * 1e-3)  # Convert to seconds

            print(f"INFO >> Running final step...")
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
            print(f"|===== Chunk sizes: {chunk_sizes}")
            print(f"|===== Latency P0: {p0_time:.4f} sec (add_request)")
            print(f"|===== Latency P1: {p1_time:.4f} sec (prefill)")
            print(f"|===== Latency P2: {p2_time:.4f} sec (1st decode step)")
            print(f"|===== Per-chunk latencies: min={min(chunk_latencies):.4f}, max={max(chunk_latencies):.4f}, avg={sum(chunk_latencies)/len(chunk_latencies):.4f}")
            print(f"+=============================================================")

            # Save overall benchmark results
            benchmark_result = {
                'max_seq_len': benchmark_dim.max_seq_len,
                'batch_size': benchmark_dim.batch_size,
                'initial_chunk_size': benchmark_dim.initial_chunk_size,
                'num_chunks': num_chunks,
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

            # Save per-chunk latencies
            chunk_results = []
            for i, (chunk_size, latency) in enumerate(zip(chunk_sizes, chunk_latencies)):
                chunk_result = {
                    'max_seq_len': benchmark_dim.max_seq_len,
                    'batch_size': benchmark_dim.batch_size,
                    'initial_chunk_size': benchmark_dim.initial_chunk_size,
                    'chunk_index': i,
                    'chunk_size': chunk_size,
                    'chunk_latency_sec': latency
                }
                chunk_results.append(chunk_result)

            chunk_df = pd.DataFrame(chunk_results)
            if chunk_f.tell() == 0:
                chunk_df.to_csv(chunk_f, index=False)
            else:
                chunk_df.to_csv(chunk_f, header=False, index=False)
            chunk_f.flush()

if __name__ == '__main__':
    parser = FlexibleArgumentParser(
        description='Benchmark the re-computation latency with adaptive chunk sizes.')
    parser.add_argument(
        '--model',
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help='Name or path of the huggingface model to use.')
    parser.add_argument(
        '--preemption-mode',
        type=str,
        choices=['recompute', 'swap'],
        default="recompute",
        help='Preemption mode: recompute or swap.')
    parser.add_argument(
        '--cache-size-per-token',
        type=int,
        default=CACHE_SIZE_PER_TOKEN,
        help='Size of the cache per token in bytes.')
    parser.add_argument(
        '--manual',
        type=str,
        default=None,
        help='Manual benchmark dimensions in format "max_seq_len,batch_size,initial_chunk_size".')
    args = parser.parse_args()
    main(args)
