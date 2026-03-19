import pandas as pd
import matplotlib.pyplot as plt
import argparse
import numpy as np


def print_metadata(csv_path, model_arch):
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Filter for specified model architecture
    df = df[df["model_arch"] == model_arch]

    # Drop data where values required below are missing.
    df = df.dropna(
        subset=[
            "dp_epsilon",
            "epoch_categorical_accuracy",
            "max_batch_steps_per_second",
            "mb_sent_per_batch",
            "mb_recv_per_batch",
        ]
    )

    # Filter out any experiments with missing data indicating that they might
    # still be running and those metrics have not yet been written).
    df["mb_sent_per_batch"] = pd.to_numeric(df["mb_sent_per_batch"], errors="coerce")
    df = df.dropna(subset=["mb_sent_per_batch"]).reset_index(drop=True)

    df["min_seconds_per_batch"] = 1 / df["max_batch_steps_per_second"]

    # Aggregate across the same experiment parameters.
    df = df[["model_arch", "protocol", "backprop_scaling_factor", "batch_size", "min_seconds_per_batch", "mb_sent_per_batch", "mb_recv_per_batch"]]
    df = (
        df.groupby(
            [
                "model_arch",
                "protocol",
                "backprop_scaling_factor",
                "batch_size",
            ]
        ).agg(
            {
                "min_seconds_per_batch": "min",
                "mb_sent_per_batch": "mean",
                "mb_recv_per_batch": "mean",
            }
        )
    )

    # df["max_batch_seconds_per_step"] = 1 / df["max_batch_steps_per_second"]
    # Add seconds per step column
    # df["seconds_per_step"] = df["max_batch_steps_per_second"] ** -1
    # df["seconds_per_step"] = 1 / df["max_batch_steps_per_second"]

    print(f"Data for model: {model_arch}")
    print(df)



def main():
    parser = argparse.ArgumentParser(
        description="Plot metrics vs epsilon for different protocols"
    )
    parser.add_argument(
        "--model_arch", required=True, help="Model architecture to plot"
    )
    parser.add_argument(
        "--csv",
        default="metrics.csv",
        help="Path to the CSV file (default: metrics.csv)",
    )

    args = parser.parse_args()

    print_metadata(args.csv, args.model_arch)


if __name__ == "__main__":
    main()
