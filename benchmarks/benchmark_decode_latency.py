import argparse
import gc
import math
import os
import numpy as np
import pandas as pd
import time
import torch
import matplotlib.pyplot as plt

from dataclasses import dataclass
from tqdm import tqdm
from typing import List, Optional

from vllm import SamplingParams, LLMEngine, TokensPrompt
from vllm.engine.arg_utils import EngineArgs
from vllm.utils import FlexibleArgumentParser


MIN_TOKENS = 1024
MAX_TOKENS = 65536
MIN_SEQ_LEN = 512


@dataclass
class BenchmarkDim:
    max_seq_len: int
    batch_size: int

    def __str__(self):
        return f"max_seq_len={self.max_seq_len}, batch_size={self.batch_size}"


def generate_benchmark_dims() -> List[BenchmarkDim]:
    benchmark_dims = []
    tokens = MIN_TOKENS

    while tokens <= MAX_TOKENS:
        max_seq_len = MIN_SEQ_LEN
        while max_seq_len <= tokens:
            if tokens % max_seq_len == 0:
                batch_size = tokens // max_seq_len
                benchmark_dims.append(
                    BenchmarkDim(max_seq_len=max_seq_len, batch_size=batch_size)
                )
            max_seq_len *= 2  # Increment max_seq_len by powers of 2
        tokens *= 2  # Increment tokens by powers of 2

    return benchmark_dims


def manual_benchmark_dims(manual_str: str) -> List[BenchmarkDim]:
    """
    Parse the manual benchmark dimensions from the command line.
    The string should be in the format 'max_seq_len,batch_size'.
    """

    def validate_input(input):
        parts = input.split(",")
        if len(parts) != 2:
            return None
        x, y = map(int, parts)
        return x, y

    benchmark_dims = []
    while validate_input(manual_str) is None:
        manual_str = input(
            "Please enter in the format 'max_seq_len,batch_size': "
        )

    max_seq_len, batch_size = validate_input(manual_str)
    benchmark_dims.append(BenchmarkDim(max_seq_len, batch_size))
    return benchmark_dims


def warmup_device(args: argparse.Namespace, num_warmup: int = 1):
    """
    Warm up the device by allocating and deallocating memory.
    """
    if not torch.cuda.is_available():
        print("ERROR >> CUDA is not available.")
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
    print("INFO >> Warmup completed.")


def main(args: argparse.Namespace):
    if args.manual is not None:
        benchmark_dimensions = manual_benchmark_dims(args.manual)
    else:
        benchmark_dimensions = generate_benchmark_dims()

    warmup_device(args)

    for benchmark_dim in benchmark_dimensions:
        if benchmark_dim.max_seq_len * benchmark_dim.batch_size > MAX_TOKENS:
            print(
                f"WARN >> Skipping {benchmark_dim} due to exceeding the maximum token limit."
            )
            continue

        print("INFO >> Running benchmark with dimension:")
        print(f"INFO >> ===== {benchmark_dim} =====")
        print(
            f"INFO >> ===== Number of total tokens: {benchmark_dim.max_seq_len * benchmark_dim.batch_size} ====="
        )

        prefill_len = int(benchmark_dim.max_seq_len * args.prefill_ratio)
        decode_len = int(benchmark_dim.max_seq_len * (1 - args.prefill_ratio))

        engine_args = EngineArgs(
            model=args.model,
            disable_log_stats=False,
            max_num_seqs=benchmark_dim.batch_size,
            enable_chunked_prefill=False,
            max_model_len=benchmark_dim.max_seq_len
            * benchmark_dim.batch_size,  # NOTE: Need this to ensure only one iteration is prefill
        )
        my_engine = LLMEngine.from_engine_args(engine_args=engine_args)

        sampling_params = SamplingParams(
            temperature=0,
            min_tokens=decode_len,
            max_tokens=decode_len,
            ignore_eos=True,
        )

        prompt_token_ids = TokensPrompt(
            prompt_token_ids=list(1 for _ in range(prefill_len))
        )
        print("Warming up inference...")
        my_engine.add_request(
            request_id=str(0),
            prompt=prompt_token_ids,
            params=sampling_params,
        )
        while True:
            request_outputs = my_engine.step()
            if request_outputs and request_outputs[0].finished:
                break

        print(
            f"INFO >> Creating {benchmark_dim.batch_size} sequences of length {benchmark_dim.max_seq_len}..."
        )
        time_p0_s = time.perf_counter_ns()
        for i in range(benchmark_dim.batch_size):
            prompt_token_ids = TokensPrompt(
                prompt_token_ids=list(1 for _ in range(prefill_len))
            )
            my_engine.add_request(
                request_id=str(i + 1),
                prompt=prompt_token_ids,
                params=sampling_params,
            )
        time_p0_e = time.perf_counter_ns()

        print("INFO >> Running prefill...")
        time_p1_s = time.perf_counter_ns()
        my_engine.step()
        time_p1_e = time.perf_counter_ns()
        print("INFO >> Finished prefill!")

        print("INFO >> Running steps...")
        event_timestamps = [time.perf_counter_ns()]
        num_finished = 0
        while num_finished < benchmark_dim.batch_size:
            request_outputs = my_engine.step()
            for request_output in request_outputs:
                if request_output.finished:
                    assert (
                        len(request_output.outputs[0].token_ids) == decode_len
                    )
                    num_finished += 1
            event_timestamps.append(time.perf_counter_ns())
        print("INFO >> Finished decoding!")

        del my_engine
        gc.collect()
        torch.cuda.empty_cache()

        p0_time = (time_p0_e - time_p0_s) / 1e9
        p1_time = (time_p1_e - time_p1_s) / 1e9
        step_latencies = np.diff(np.array(event_timestamps)) / 1e9

        print("+==================== Benchmark completed ====================")
        print(f"ö===== Dimension: {benchmark_dim}")
        print(f"ö===== Latency P0: {p0_time:.4f} sec (add_request)")
        print(f"ö===== Latency P1: {p1_time:.4f} sec (prefill)")
        print(
            f"ö===== Latency P2 (min): {step_latencies.min():.4f} sec, at index {step_latencies.argmin()}"
        )
        print(f"ö===== Latency P2 (mean): {step_latencies.mean():.4f} sec")
        print(
            f"ö===== Latency P2 (max): {step_latencies.max():.4f} sec, at index {step_latencies.argmax()}"
        )
        print(f"ö===== Number of steps: {len(step_latencies)}")
        print("+=============================================================")

        # Create one row per step latency
        benchmark_results = []
        for step_idx, latency in enumerate(step_latencies):
            benchmark_results.append(
                {
                    "max_seq_len": benchmark_dim.max_seq_len,
                    "batch_size": benchmark_dim.batch_size,
                    "step": step_idx,
                    "latency": latency,
                }
            )

        path_stem = f"decode_msl_{benchmark_dim.max_seq_len}_bs_{benchmark_dim.batch_size}_pl_{prefill_len}.csv"

        with open(path_stem + ".csv", mode="a", newline="") as f:
            df = pd.DataFrame(benchmark_results)
            if f.tell() == 0:
                df.to_csv(f, index=False)
            else:
                df.to_csv(f, header=False, index=False)
            f.flush()

        # Generate and save the plot
        plt.figure(figsize=(10, 6))
        plt.plot(df["step"], df["latency"], "-o")
        plt.xlabel("Step")
        plt.ylabel("Latency (seconds)")
        plt.title(
            f"Decoding Latency, max_seq_len={benchmark_dim.max_seq_len}, batch_size={benchmark_dim.batch_size}, prefill_tokens={prefill_len}"
        )
        plt.grid(True)

        plt.savefig(path_stem + ".png")
        plt.close()  # Close the figure to free memory


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark the re-computation latency of processing a single batch of requests."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B",
        help="Name or path of the huggingface model to use.",
    )
    # parser.add_argument(
    #     '--cache-size-per-token',
    #     help='Size of the cache per token in bytes. Determined by the model.')
    #     type=int,
    #     default=CACHE_SIZE_PER_TOKEN,
    #     help='Size of the cache per token in bytes. Determined by the model.')
    parser.add_argument(
        "--prefill-ratio",
        type=float,
        default=0.5,
        help="The ratio of the sequence length to prefill before decoding.",
    )
    parser.add_argument(
        "--manual",
        type=str,
        default=None,
        help="Manual mode, if specified, will use the specified benchmark dimensions."
        "Otherwise, will generate benchmark dimensions automatically."
        'The format should be "max_seq_len,batch_size".',
    )
    args = parser.parse_args()
    main(args)
