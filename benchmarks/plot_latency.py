import argparse
import pandas as pd
import matplotlib.pyplot as plt

def plot_latency(args):
    data = pd.read_csv(args.csv)
    data['x_axis'] = data['max_seq_len'] * data['batch_size']

    plt.figure(figsize=(10, 6))

    plt.plot(data['x_axis'], data['latency_sec'], label='Latency (sec)', marker='o')

    plt.plot(data['x_axis'], data['computed_prefill_sec'], label='Computed Prefill (sec)', marker='x')

    plt.xlabel('Max Sequence Length * Batch Size')
    plt.ylabel('Time (sec)')
    plt.title('Latency and Computed Prefill Time vs Max Sequence Length * Batch Size')
    plt.legend()

    plt.savefig(f"{args.out}_latency.png")


def plot_periodic_latency(args):
    data = pd.read_csv(args.csv)
    data['x_axis'] = data['max_seq_len'] * data['batch_size']

    unique_data = data.drop_duplicates(subset="x_axis")

    plt.figure(figsize=(10, 6))

    plt.bar(unique_data['x_axis'], unique_data['1st_period_sec'], label='1st Period (sec)', width=2000, alpha=0.7)
    plt.bar(unique_data['x_axis'], unique_data['2nd_period_sec'], bottom=unique_data['1st_period_sec'], label='2nd Period (sec)', width=2000, alpha=0.7)

    plt.xlabel('Max Sequence Length * Batch Size')
    plt.ylabel('Latency (sec)')
    plt.title('Latency Composition: 1st Period and 2nd Period')
    plt.legend()

    plt.savefig(f"{args.out}_periods.png")


def plot_gpu_usage(args):
    data = pd.read_csv(args.csv)
    data['time'] = pd.to_datetime(data['time'])
    data['time_diff'] = (data['time'] - data['time'].iloc[0]).dt.total_seconds()

    plt.figure(figsize=(10, 6))

    plt.plot(data.index, data['gpu_usage'], color='gray', linestyle='-', label='GPU Usage')

    # Unique remarks for color coding
    unique_remarks = data['remark'].unique()
    colors = plt.cm.tab10(range(len(unique_remarks)))  # Generate distinct colors for each remark

    for i, remark in enumerate(unique_remarks):
        filtered_data = data[data['remark'] == remark]
        plt.plot(filtered_data.index, filtered_data['gpu_usage'], label=f'Remark: {remark}', color=colors[i])
        plt.scatter(filtered_data.index, filtered_data['gpu_usage'], color=colors[i], s=50,
                    label=f'Data Points ({remark})')

    # Labels and legend
    plt.xlabel('Step')
    plt.ylabel('GPU Usage')
    plt.title('GPU KV-cache Usage Over Time')
    plt.legend()

    plt.savefig(f"{args.out}_kvcache.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize the latency.')
    parser.add_argument('--csv',
                        type=str,
                        help='Path to the latency results.')
    parser.add_argument('--out',
                        type=str,
                        default='latency_results',
                        help='Output file label.')
    args = parser.parse_args()
    # plot_latency(args)
    # plot_periodic_latency(args)
    plot_gpu_usage(args)
