import pandas as pd
import matplotlib.pyplot as plt
import argparse
import numpy as np


def plot_metrics_by_protocol(csv_path, model_arch, epsilon, min_y):
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Drop data where values required below are missing.
    df = df.dropna(subset=["dp_epsilon", "epoch_categorical_accuracy", "max_batch_steps_per_second"])

    # Filter for specified model architecture
    df = df[df["model_arch"] == model_arch]

    # Filter out any experiments with non-integer batch sizes or backprop
    # scaling factors (indicating that they might still be running and those
    # metrics have not yet been written).
    df["batch_size"] = pd.to_numeric(df["batch_size"], errors="coerce")
    df = df.dropna(subset=["batch_size"]).reset_index(drop=True)

    # Round epsilon to 2 decimal places.
    df["dp_epsilon"] = df["dp_epsilon"].astype(float).round(2)

    # Filter for specified epsilon
    df = df[df["dp_epsilon"] == epsilon]

    # Convert batch size and backprop scaling factor to integers
    df["batch_size"] = df["batch_size"].astype(int)
    df["backprop_scaling_factor"] = df["backprop_scaling_factor"].astype(int)

    # Sort by epsilon to ensure ascending order in plot
    df = df.sort_values("dp_epsilon")

    print(df)

    plt.style.use("grayscale")  # Set colormap to grayscale
    # Create a figure with two subplots side by side
    fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))

    # Get unique protocols
    protocols = df["protocol"].unique()
    scaling_factors = df["backprop_scaling_factor"].unique()
    batch_sizes = df["batch_size"].unique()

    # Plot lines for each subplot
    for protocol in protocols:
        for scaling_factor in scaling_factors:
            for batch_size in batch_sizes:
                experiment_accuracy = df[
                    (df["protocol"] == protocol)
                    & (df["backprop_scaling_factor"] == scaling_factor)
                    & (df["batch_size"] == batch_size)
                ]

                if experiment_accuracy.empty:
                    continue

                if protocol == "DP-SGD":
                    # DP-SGD does backprop in cleartext, no scaling factor.
                    label = f"{protocol} sf: inf, bs: {batch_size}"
                else:
                    label = f"{protocol} sf: {scaling_factor}, bs: {batch_size}"

                # Accuracy plot
                acc_line = ax1.plot(
                    range(len(experiment_accuracy["epoch_categorical_accuracy"])),
                    experiment_accuracy["epoch_categorical_accuracy"],
                    marker="o",
                    label=label,
                )

    fig.suptitle(f"Accuracy vs Epoch, Epsilon: {epsilon}, Model: {model_arch}")

    # Customize the accuracy plot
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation Accuracy")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.7)

    # Set the minimum y-axis value if specified
    if min_y:
        ax1.set_ylim(bottom=min_y)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the plot
    output_filename = f"epoch_vs_acc_epsilon{epsilon}_model{model_arch}.png"
    plt.savefig(output_filename, bbox_inches="tight")
    print(f"Plot saved as {output_filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot metrics vs epsilon for different protocols"
    )
    parser.add_argument(
        "--model_arch", required=True, help="Model architecture to plot"
    )
    parser.add_argument(
        "--epsilon", type=float, help="Epsilon value to plot"
    )
    parser.add_argument(
        "--min_y", type=float, help="Minimum y-axis value for the plot"
    )
    parser.add_argument(
        "--csv",
        default="metrics.csv",
        help="Path to the CSV file (default: metrics.csv)",
    )

    args = parser.parse_args()

    plot_metrics_by_protocol(args.csv, args.model_arch, args.epsilon, args.min_y)


if __name__ == "__main__":
    main()
