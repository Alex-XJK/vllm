import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns


def load_profiling_data(profile_json_path: str) -> list:
    """
    Load the profiling data from a JSON file.
    """
    with open(profile_json_path, 'r') as f:
        profiling_data = json.load(f)
        # Access trace events
        trace_events = profiling_data.get('traceEvents', [])
    return trace_events


def get_cpu_data(trace_events: list) -> pd.DataFrame:
    """
    Extract CPU data from the trace events.
    """
    cpu_events = [
        event for event in trace_events
        if event.get('cat') == 'cpu_op' and event.get('ph') == 'X'
    ]
    print(f"Found {len(cpu_events)} CPU events.")

    cpu_df = pd.DataFrame(cpu_events)
    return cpu_df


def process_cpu_data(cpu_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the CPU data.
    """
    # Extract necessary fields
    cpu_df['name'] = cpu_df['name']
    cpu_df['duration_ns'] = cpu_df['dur']  # Duration in nanoseconds
    cpu_df['duration_ms'] = cpu_df['duration_ns'] / 1e6  # Convert to milliseconds

    # Aggregate total duration per operation
    cpu_agg = cpu_df.groupby('name')['duration_ms'].sum().reset_index()

    # Sort operations by total duration
    cpu_agg = cpu_agg.sort_values(by='duration_ms', ascending=False)

    return cpu_agg


def get_cuda_data(trace_events: list) -> pd.DataFrame:
    """
    Extract CUDA data from the trace events.
    """
    cuda_events = [
        event for event in trace_events
        if event.get('cat') == 'cuda_runtime' and event.get('ph') == 'X'
    ]
    print(f"Found {len(cuda_events)} CUDA events.")

    cuda_df = pd.DataFrame(cuda_events)
    return cuda_df


def process_cuda_data(cuda_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the CUDA data.
    """
    # Extract necessary fields
    cuda_df['name'] = cuda_df['name']
    cuda_df['duration_ns'] = cuda_df['dur']  # Duration in nanoseconds
    cuda_df['duration_ms'] = cuda_df['duration_ns'] / 1e6  # Convert to milliseconds

    # Aggregate total duration per operation
    cuda_agg = cuda_df.groupby('name')['duration_ms'].sum().reset_index()

    # Sort operations by total duration
    cuda_agg = cuda_agg.sort_values(by='duration_ms', ascending=False)

    return cuda_agg


def get_memory_data(trace_events: list) -> pd.DataFrame:
    """
    Extract memory data from the trace events.
    """
    memory_events = [
        event for event in trace_events
        if event.get('name') == '[memory]' and event.get('ph') in ['M', 'I', 'i']
    ]
    print(f"Found {len(memory_events)} memory events.")

    memory_df = pd.DataFrame(memory_events)
    return memory_df


def process_memory_data(memory_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the memory data.
    """
    # Extract necessary fields
    # Assuming 'args' contains memory information
    memory_df['name'] = memory_df['name']
    memory_df['timestamp_ns'] = memory_df['ts']
    memory_df['memory_allocated'] = memory_df['args'].apply(lambda x: x.get('Total Allocated', 0))
    memory_df['memory_reserved'] = memory_df['args'].apply(lambda x: x.get('Total Reserved', 0))

    # Convert timestamp to milliseconds
    memory_df['timestamp_ms'] = memory_df['timestamp_ns'] / 1e6

    # Sort by timestamp
    memory_df = memory_df.sort_values(by='timestamp_ms')

    return memory_df


def plot_all(cpu: pd.DataFrame, cuda: pd.DataFrame, memory: pd.DataFrame, output_path: str) -> None:
    """
    Plot the CPU, CUDA, and memory data.
    """
    # Create a grid layout
    fig = plt.figure(constrained_layout=True, figsize=(18, 12))
    gs = gridspec.GridSpec(3, 1, figure=fig)

    # CPU Time Plot
    ax0 = fig.add_subplot(gs[0, 0])
    sns.barplot(x='duration_ms', y='name', data=cpu.head(20), palette='viridis', ax=ax0)
    ax0.set_xlabel('Total CPU Time (ms)')
    ax0.set_ylabel('Operation')
    ax0.set_title('Top 20 CPU Operations by Total Time')

    # CUDA Time Plot
    ax1 = fig.add_subplot(gs[1, 0])
    sns.barplot(x='duration_ms', y='name', data=cuda.head(20), palette='magma', ax=ax1)
    ax1.set_xlabel('Total CUDA Time (ms)')
    ax1.set_ylabel('CUDA Operation')
    ax1.set_title('Top 20 CUDA Operations by Total Time')

    # Memory Usage Plot
    ax2 = fig.add_subplot(gs[2, 0])
    ax2.plot(memory['timestamp_ms'], memory['memory_allocated'], label='Memory Allocated', color='blue')
    ax2.plot(memory['timestamp_ms'], memory['memory_reserved'], label='Memory Reserved', color='orange')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Memory (bytes)')
    ax2.set_title('Memory Usage Over Time')
    ax2.legend()

    # Save the plot as a PNG image
    plt.savefig(output_path, dpi=400)

    plt.show()


def main(args):
    # Load the profiling data
    trace_events = load_profiling_data(args.dir)

    # Get CPU data
    cpu_df = get_cpu_data(trace_events)
    cpu_agg = process_cpu_data(cpu_df)

    # Get CUDA data
    cuda_df = get_cuda_data(trace_events)
    cuda_agg = process_cuda_data(cuda_df)

    # Get memory data
    memory_df = get_memory_data(trace_events)
    memory_df = process_memory_data(memory_df)

    # Plot the data
    plot_all(cpu_agg, cuda_agg, memory_df, args.output)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize the profiler results.')
    parser.add_argument('--dir', 
                        type=str, 
                        default='vllm_benchmark_result/benchmark_re_256_1_256_1729658647.8585253/32874d54c30d68_1564.1729658647932508105.pt.trace.json', 
                        help='Path to the profiler results.')
    parser.add_argument('--output', 
                        type=str, 
                        default='profiler_results.png', 
                        help='Path to save the visualization.')
    args = parser.parse_args()
    main(args)

