import pandas as pd
import matplotlib.pyplot as plt
import argparse
import numpy as np


def plot_metrics(csv_path):
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Drop data where values required below are missing.
    df = df.dropna(subset=["model_arch", "max_batch_steps_per_second"])

    df["min_seconds_per_batch"] = 1 / df["max_batch_steps_per_second"]

    # Filter out any experiments with non-integer batch sizes (indicating that
    # they might still be running and not all metrics have been written).
    df["batch_size"] = pd.to_numeric(df["batch_size"], errors="coerce")
    df = df.dropna(subset=["batch_size"]).reset_index(drop=True)

    df = df[
        [
            "model_arch",
            "protocol",
            "backprop_scaling_factor",
            "batch_size",
            "eager_mode",
            "min_seconds_per_batch",
        ]
    ]

    df = df.groupby(
        [
            "model_arch",
            "protocol",
            "backprop_scaling_factor",
            "batch_size",
            "eager_mode",
        ]
    ).agg(
        {
            "min_seconds_per_batch": "min",
        }
    ).reset_index()

    # Sort to ensure ascending order in plot
    df = df.sort_values("model_arch")
    df = df.sort_values("protocol")
    # df = df.sort_values("backprop_scaling_factor")
    df = df.sort_values("batch_size")

    plt.style.use("grayscale")  # Set colormap to grayscale
    # Create a figure with two subplots side by side
    fig, ax1 = plt.subplots(1, 1, figsize=(6, 4), width_ratios=[1])

    # Get unique protocols
    model_archs = df["model_arch"].unique()
    protocols = df["protocol"].unique()
    scaling_factors = df["backprop_scaling_factor"].unique()
    batch_sizes = df["batch_size"].unique()

    # Recalculate speedup by comparing equivalent configurations with different eager modes
    speedup_results = []

    # Group by everything except eager_mode to find matching pairs
    groups = df.groupby(['model_arch', 'protocol', 'backprop_scaling_factor', 'batch_size'])

    for name, group in groups:
        if len(group) == 2:  # We need both eager=0 and eager=1 for comparison
            eager_time = group[group['eager_mode'] == 1]['min_seconds_per_batch'].values[0]
            deferred_time = group[group['eager_mode'] == 0]['min_seconds_per_batch'].values[0]

            # Calculate speedups
            percentage_speedup = (eager_time - deferred_time) / eager_time * 100
            absolute_speedup = eager_time - deferred_time

            # Create a result row with the configuration and speedup metrics
            result = {
                'model_arch': name[0],
                'protocol': name[1], 
                'backprop_scaling_factor': name[2],
                'batch_size': name[3],
                'speedup_percentage': percentage_speedup,
                'absolute_speedup': absolute_speedup
            }
            speedup_results.append(result)

    # Create a new dataframe with the speedup results
    speedup_df = pd.DataFrame(speedup_results)

    # Sort the results for plotting
    speedup_df = speedup_df.sort_values(['model_arch', 'protocol', 'batch_size'])


    # Get unique values for grouping
    unique_models = speedup_df['model_arch'].unique()
    unique_protocols = speedup_df['protocol'].unique()

    # Create a color map for protocols
    protocol_colors = plt.cm.Greys(np.linspace(0.3, 0.8, len(unique_protocols)))
    protocol_color_map = {protocol: protocol_colors[i] for i, protocol in enumerate(unique_protocols)}

    # Calculate positions for grouped bars
    group_spacing = 0.8  # Space between model architecture groups
    bar_spacing = 0.30   # Space between protocol bars within a group
    bar_sz = 0.2     # Height of each bar
    group_positions = []
    bar_positions = []
    labels = []

    # Calculate positions and create labels
    current_pos = 0
    for model in unique_models:
        model_data = speedup_df[speedup_df['model_arch'] == model]
        model_protocols = model_data['protocol'].unique()
    
        # Store the group's starting position
        group_start = current_pos
    
        # Calculate positions for each protocol within this model group
        for protocol in unique_protocols:
            protocol_data = model_data[model_data['protocol'] == protocol]
        
            for _, row in protocol_data.iterrows():
                bar_positions.append(current_pos)
                label = f"{model}, bs={int(row['batch_size'])}"
                if row['backprop_scaling_factor'] != 1.0:
                    label += f", sf={row['backprop_scaling_factor']}"
                labels.append(label)
                current_pos += bar_spacing
    
        # Add space after each group
        current_pos += group_spacing - bar_spacing
        group_positions.append((group_start + current_pos - group_spacing) / 2)

    # Plot percentage speedup on the left side (ax1) - flipped axis
    for protocol in unique_protocols:
        protocol_indices = [i for i, row in enumerate(speedup_df.itertuples()) 
                        if row.protocol == protocol]
        protocol_positions = [bar_positions[i] for i in protocol_indices]
        protocol_data = speedup_df[speedup_df['protocol'] == protocol]
    
        ax1.bar(protocol_positions, protocol_data['speedup_percentage'], 
                color=protocol_color_map[protocol],
                width=bar_sz,
                label=protocol)

    # Set y-ticks and labels
    ax1.set_xticks(group_positions)
    ax1.set_xticklabels(unique_models)
    ax1.set_xlabel("Model Architecture")

    # Add value annotations to the bars
    for protocol in unique_protocols:
        protocol_data = speedup_df[speedup_df['protocol'] == protocol]
        protocol_indices = [i for i, row in enumerate(speedup_df.itertuples()) 
                        if row.protocol == protocol]
        protocol_positions = [bar_positions[i] for i in protocol_indices]
    
        for pos, ypos, val in zip(protocol_positions, protocol_data['speedup_percentage'], protocol_data['absolute_speedup']):
            ax1.text(pos, ypos + 2, f"{val:.1f}s", va='center', ha='center', fontsize=7)


    # Set titles and labels
    fig.suptitle(f"Deferred Execution Speed Up versus Eager")

    # Customize the accuracy plot
    ax1.set_ylabel("Relative Speed Up (%)")

    # ax1.legend(loc='upper left')
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.2)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the plot
    output_filename = f"deferred_speedup.png"
    plt.savefig(output_filename, bbox_inches="tight")
    print(f"Plot saved as {output_filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot speedup of eager vs deferred mode for all protocols and models."
    )
    parser.add_argument(
        "--csv",
        default="metrics.csv",
        help="Path to the CSV file (default: metrics.csv)",
    )

    args = parser.parse_args()

    plot_metrics(args.csv)


if __name__ == "__main__":
    main()
