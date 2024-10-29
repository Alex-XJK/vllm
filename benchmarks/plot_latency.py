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

    plt.show()
    plt.savefig(args.out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize the latency.')
    parser.add_argument('--csv',
                        type=str,
                        help='Path to the latency results.')
    parser.add_argument('--out',
                        type=str,
                        default='latency_results.png',
                        help='Path to save the visualization.')
    args = parser.parse_args()
    plot_latency(args)
