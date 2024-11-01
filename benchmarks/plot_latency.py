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
    plot_latency(args)
    plot_periodic_latency(args)
