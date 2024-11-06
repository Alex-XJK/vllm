"""
Benchmark the re-computation latency of processing a single batch of requests.
Author: Alex
"""
import gc
import math
import numpy as np
import os
import pandas as pd
import sys
import time
import torch
from dataclasses import asdict, dataclass
from pathlib import Path
from tqdm import tqdm
from typing import List, Optional

from vllm import SamplingParams, LLMEngine, TokensPrompt
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.metrics import LoggingStatLogger, PrometheusStatLogger
from vllm.engine.metrics_types import StatLoggerBase, Stats, SupportsMetricsInfo
from vllm.inputs import PromptType
from vllm.outputs import RequestOutput
from vllm.utils import FlexibleArgumentParser

CACHE_SIZE_PER_TOKEN = 131072 # Determined by the model
CHUNK_SIZE_LOG_MIN = 8 # Arbitrary
TOKEN_SIZE_LOG_MIN = 8 # Arbitrary, but should be at least chunk size min
TOKEN_SIZE_LOG_MAX = 17  # Determined by number of GPU blocks (~ GPU HBM size).
MAX_MODEL_TOKENS = 65536 # Should have been 131072 but we truncate to 65536 otherwise it throws a CUDA error

class KvLogger(StatLoggerBase):

    @dataclass
    class LogEntry:
        time: float
        cpu_usage: float
        gpu_usage: float
        remark: int

    def __init__(self, local_interval: float) -> None:
        super().__init__(local_interval)
        self.records = list[KvLogger.LogEntry]()
        self.remark_counter = 0

    def log(self, stats: Stats) -> None:
        """
        Called by LLMEngine.
        Logs to Stdout every self.local_interval seconds.
        """
        if stats.now > self.last_local_log + self.local_interval:
            print(f"KvLogger >> GPU KV Cache: {stats.gpu_cache_usage_sys * 100:.6f}")
            self.last_local_log = stats.now
            self.records.append(
                KvLogger.LogEntry(
                    time=stats.now,
                    cpu_usage=stats.cpu_cache_usage_sys,
                    gpu_usage=stats.gpu_cache_usage_sys,
                    remark=self.remark_counter,
                )
            )

    def get_records(self) -> list:
        return self.records

    def get_gpu_kvcache_usage(self) -> list:
        return [record.gpu_usage for record in self.records]

    def set_counter(self, counter: int) -> None:
        self.remark_counter = counter

    def info(self, type: str, obj: SupportsMetricsInfo) -> None:
        raise NotImplementedError


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


def manual_benchmark_dims(manual_str: str) -> BenchmarkDim:
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

    while validate_input(manual_str) is None:
        manual_str = input("Please enter in the format 'max_seq_len,batch_size,chunk_size': ")

    max_seq_len, batch_size, chunk_size = validate_input(manual_str)
    return BenchmarkDim(max_seq_len, batch_size, chunk_size)


def config_global(args: argparse.Namespace):
    global CACHE_SIZE_PER_TOKEN
    CACHE_SIZE_PER_TOKEN = args.cache_size_per_token


def warmup_device(args: argparse.Namespace, num_warmup: int = 2):
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

    benchmark_dim = manual_benchmark_dims(args.manual)

    warmup_device(args)

    pid = os.getpid()
    csv_path = f"kvcache_{pid}.csv"

    with open(csv_path, mode='a', newline='') as f:

        assert benchmark_dim.max_seq_len % benchmark_dim.chunk_size == 0
        num_chunked_prefill_iters = benchmark_dim.max_seq_len // benchmark_dim.chunk_size

        print(f"INFO >> Running benchmark with dimension:")
        print(f"INFO >> ===== {benchmark_dim} =====")

        alogger = KvLogger(0.01)
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
        for i in range(2 * benchmark_dim.batch_size):
            prompt_token_ids = TokensPrompt(prompt_token_ids=list(range(benchmark_dim.max_seq_len)))
            my_engine.add_request(
                request_id=str(i),
                prompt=prompt_token_ids,
                params=sampling_params,
            )

        mem_aloc_filled, mem_resv_filled = stat_memory_now()

        print(f"INFO >> Before everything...")
        my_engine.do_log_stats()
        alogger.set_counter(0)

        print(f"INFO >> Running Prefill...")
        time_p1_s = time.perf_counter_ns()
        for i in range(num_chunked_prefill_iters):
            my_engine.step()
            my_engine.do_log_stats()
        time_p1_e = time.perf_counter_ns()

        print(f"INFO >> After Prefill...")
        my_engine.do_log_stats()
        alogger.set_counter(1)

        print(f"INFO >> Running Decode...")
        for i in range(args.decode_steps):
            outputs = my_engine.step()
            my_engine.do_log_stats()

        print(f"INFO >> After 2nd Step...")
        my_engine.do_log_stats()

        mem_aloc_step, mem_resv_step = stat_memory_now()

        print(f"INFO >> {len(outputs)} outputs received.")
        total_output_tokens = sum(len(co.token_ids) for output in outputs for co in output.outputs)

        del my_engine
        gc.collect()
        torch.cuda.empty_cache()

        mem_aloc_clean, mem_resv_clean = stat_memory_now()

        p1_time = (time_p1_e - time_p1_s) / 1e9

        print(f"+==================== Benchmark completed ====================")
        print(f"|===== Dimension: {benchmark_dim}")
        print(f"|===== Latency: {p1_time:.6f} sec")
        print(f"|===== Total Output Tokens: {total_output_tokens}")
        print(f"|===== Memory Usage:")
        print(f"|===== + {'-' * 10} + {'Aloc':>5} + {'Resv':>5} +")
        print(f"|===== | {'Init':<10} | {bytes_to_gb(mem_aloc_init):>5.2f} | {bytes_to_gb(mem_resv_init):>5.2f} |")
        print(f"|===== | {'Built':<10} | {bytes_to_gb(mem_aloc_built):>5.2f} | {bytes_to_gb(mem_resv_built):>5.2f} |")
        print(f"|===== | {'Filled':<10} | {bytes_to_gb(mem_aloc_filled):>5.2f} | {bytes_to_gb(mem_resv_filled):>5.2f} |")
        print(f"|===== | {'Step':<10} | {bytes_to_gb(mem_aloc_step):>5.2f} | {bytes_to_gb(mem_resv_step):>5.2f} |")
        print(f"|===== | {'Clean':<10} | {bytes_to_gb(mem_aloc_clean):>5.2f} | {bytes_to_gb(mem_resv_clean):>5.2f} |")
        print(f"|===== + {'-' * 10} + {'-' * 5} + {'GB':>5} +")
        print(f"+=============================================================")

        data = alogger.get_records()
        df = pd.DataFrame([asdict(obj) for obj in data])
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
        '--decode-steps',
        type=int,
        default=10,
        help='Number of decoding steps to run.')
    parser.add_argument(
        '--manual',
        type=str,
        default="16384,1,16384",
        help='Manual mode, if specified, will use the specified benchmark dimensions.'
             'Otherwise, will generate benchmark dimensions automatically.'
             'The format should be "max_seq_len,batch_size,chunk_size".')
    args = parser.parse_args()
    main(args)
