import pandas as pd
import matplotlib.pyplot as plt
import argparse
import numpy as np


def plot_metrics_by_protocol(csv_path, model_arch, min_y):
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Filter for specified model architecture
    df = df[df["model_arch"] == model_arch]

    # Drop data where values required below are missing.
    df = df.dropna(subset=["dp_epsilon", "epoch_categorical_accuracy", "max_batch_steps_per_second"])

    # Filter out any experiments with non-integer batch sizes or backprop
    # scaling factors (indicating that they might still be running and those
    # metrics have not yet been written).
    df["batch_size"] = pd.to_numeric(df["batch_size"], errors="coerce")
    df = df.dropna(subset=["batch_size"]).reset_index(drop=True)

    # Round epsilon to 2 decimal places.
    df["dp_epsilon"] = df["dp_epsilon"].astype(float).round(2)

    # Convert batch size and backprop scaling factor to integers
    df["batch_size"] = df["batch_size"].astype(int)
    df["backprop_scaling_factor"] = df["backprop_scaling_factor"].astype(float)

    # Sort by epsilon to ensure ascending order in plot
    df = df.sort_values("dp_epsilon")

    # Prepare the accuracy data for plotting
    # First get max accuracy for each experiment
    mean_accuracy_df = (
        df.groupby(
            [
                "protocol",
                "backprop_scaling_factor",
                "batch_size",
                "dp_epsilon",
                "experiment_name", # Max per experiment_name
            ]
        )["epoch_categorical_accuracy"]
        .max()
        .reset_index()
    )
    print(mean_accuracy_df)
    # print(mean_accuracy_df)
    # Then average (or max) these maximums across experiments
    mean_accuracy_df = (
        mean_accuracy_df.groupby(
            [
                "protocol",
                "backprop_scaling_factor",
                "batch_size",
                "dp_epsilon",
                # over experiment_names
            ]
        )["epoch_categorical_accuracy"]
        # .mean()
        .max()
        .reset_index()
    )

    # Prepare the batch time data for plotting.
    # Do not include the "no-privacy" data in the batch time plot as it does not
    # run the noise protocol.
    batch_time_df = df[df["dp_epsilon"] != np.inf]
    batch_time_df = (
        batch_time_df.groupby(
            [
                "protocol",
                "backprop_scaling_factor",
                "batch_size",
                # Max over all dp_epsilons
                # Max over all experiment_names
            ]
        )["max_batch_steps_per_second"]
        .max()
        .reset_index()
    )

    # Prepare the "no-privacy" data.
    no_privacy_df = mean_accuracy_df[mean_accuracy_df["dp_epsilon"] == np.inf]
    no_privacy_df = (
        no_privacy_df.groupby(
            [
                "protocol",
                "backprop_scaling_factor",
                "batch_size",
            ]
        )["epoch_categorical_accuracy"]
        .max()
        .reset_index()
    )

    # Filter out infinity dp_epsilons from mean_accuracy_df
    mean_accuracy_df = mean_accuracy_df[mean_accuracy_df["dp_epsilon"] != np.inf]


    plt.style.use("grayscale")  # Set colormap to grayscale
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.7, 3.5), width_ratios=[5, 0.3])

    # Get unique protocols
    protocols = mean_accuracy_df["protocol"].unique()
    scaling_factors = mean_accuracy_df["backprop_scaling_factor"].unique()
    batch_sizes = mean_accuracy_df["batch_size"].unique()

    # Plot lines for each subplot
    for protocol in protocols:
        for scaling_factor in scaling_factors:
            for batch_size in batch_sizes:
                experiment_accuracy = mean_accuracy_df[
                    (mean_accuracy_df["protocol"] == protocol)
                    & (mean_accuracy_df["backprop_scaling_factor"] == scaling_factor)
                    & (mean_accuracy_df["batch_size"] == batch_size)
                ]
                experiment_batch_time = batch_time_df[
                    (batch_time_df["protocol"] == protocol)
                    & (batch_time_df["backprop_scaling_factor"] == scaling_factor)
                    & (batch_time_df["batch_size"] == batch_size)
                ]
                no_privacy_accuracy = no_privacy_df[
                    (no_privacy_df["protocol"] == protocol)
                    & (no_privacy_df["backprop_scaling_factor"] == scaling_factor)
                    & (no_privacy_df["batch_size"] == batch_size)
                ]

                if experiment_accuracy.empty or experiment_batch_time.empty:
                    continue

                if protocol == "DP-SGD":
                    # DP-SGD does backprop in cleartext, no scaling factor.
                    label = f"{protocol} sf: inf, bs: {batch_size}"
                else:
                    label = f"{protocol} sf: {scaling_factor}, bs: {batch_size}"

                # Accuracy plot
                acc_line = ax1.plot(
                    experiment_accuracy["dp_epsilon"],
                    experiment_accuracy["epoch_categorical_accuracy"],
                    marker="o",
                    label=label,
                )

                # Batch steps plot
                ax2.plot(
                    0 * experiment_batch_time["max_batch_steps_per_second"],
                    1 / experiment_batch_time["max_batch_steps_per_second"],
                    marker="o",
                    label=label,
                )

                if no_privacy_accuracy.empty:
                    continue

                ax1.plot(
                    [min(experiment_accuracy["dp_epsilon"]), max(experiment_accuracy["dp_epsilon"])],
                    [no_privacy_accuracy["epoch_categorical_accuracy"], no_privacy_accuracy["epoch_categorical_accuracy"]],
                    linestyle="--",
                    color=acc_line[0].get_color(),
                )

    fig.suptitle(f"Accuracy vs Epsilon and Time, Model: {model_arch}")

    # Customize the accuracy plot
    ax1.set_xlabel("Epsilon")
    ax1.set_ylabel("Maximum Categorical Accuracy")
    # ax1.legend()
    # put the legend on the bottom right corner
    ax1.legend(loc="lower right", fontsize="small")
    ax1.grid(True, linestyle="--", alpha=0.7)

    # Set the minimum y-axis value if specified
    if min_y:
        ax1.set_ylim(bottom=min_y)

    # Customize the batch steps plot
    ax2.set_ylabel("Seconds per Batch (Excluding IO)")
    # ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.7)
    ax2.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    ax2.xaxis.grid(False)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the plot
    output_filename = f"epsilon_vs_metrics_{model_arch}.png"
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
        "--min_y", type=float, help="Minimum y-axis value for the plot"
    )
    parser.add_argument(
        "--csv",
        default="metrics.csv",
        help="Path to the CSV file (default: metrics.csv)",
    )

    args = parser.parse_args()

    plot_metrics_by_protocol(args.csv, args.model_arch, args.min_y)


if __name__ == "__main__":
    main()
