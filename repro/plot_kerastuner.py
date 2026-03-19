import keras_tuner as kt
import matplotlib.pyplot as plt
import argparse
import numpy as np
import os
import json
import tensorflow as tf  # Import TensorFlow

def is_pareto_optimal(costs):
    """
    Finds the Pareto-optimal points (same as before).
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
            is_efficient[i] = True
    return is_efficient


def load_and_plot_results(directory, title_suffix, output_filename):
    """
    Loads Keras Tuner results from a directory and plots the Pareto frontier.
    This version reads trial data from individual trial directories.
    """
    project_dir = os.path.join(directory)
    if not os.path.isdir(project_dir):
        print(f"Error: Project directory not found at {project_dir}")
        return

    accuracies = []
    training_times = []
    trial_ids = []

    # Iterate through all subdirectories (potential trials)
    for trial_id in os.listdir(project_dir):
        trial_dir = os.path.join(project_dir, trial_id)
        if not os.path.isdir(trial_dir):
            continue  # Skip files, only process directories

        # Construct path to trial.json
        trial_json_path = os.path.join(trial_dir, 'trial.json')
        if not os.path.exists(trial_json_path):
            print(f"Warning: Skipping {trial_id} - no trial.json found.")
            continue

        # Load trial.json
        try:
            with open(trial_json_path, 'r') as f:
                trial_data = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Skipping {trial_id} - invalid trial.json.")
            continue

        # Extract metrics (robustly)
        if 'metrics' in trial_data and 'metrics' in trial_data['metrics'] and trial_data['metrics']['metrics']:
            metrics = trial_data['metrics']['metrics']

            if 'val_categorical_accuracy' in metrics and metrics['val_categorical_accuracy']['observations']:
                accuracy = metrics['val_categorical_accuracy']['observations'][0]['value'][0]
            else:
                accuracy = None

            if 'time' in metrics and metrics['time']['observations']:
                training_time = metrics['time']['observations'][0]['value'][0]
            else:
                training_time = None

            if accuracy is not None and training_time is not None:
                accuracies.append(accuracy)
                training_times.append(training_time)
                trial_ids.append(trial_id)
        else:
            print(f"Warning: Skipping trial {trial_id} due to missing metrics.")

    if not accuracies:
        print("Error: No valid trial data found.")
        return

    costs = np.array([[-a, t] for a, t in zip(accuracies, training_times)])
    pareto_mask = is_pareto_optimal(costs)
    pareto_accuracies = np.array(accuracies)[pareto_mask]
    pareto_times = np.array(training_times)[pareto_mask]
    pareto_trial_ids = np.array(trial_ids)[pareto_mask]

    sorted_indices = np.argsort(pareto_accuracies)
    pareto_accuracies = pareto_accuracies[sorted_indices]
    pareto_times = pareto_times[sorted_indices]
    pareto_trial_ids = pareto_trial_ids[sorted_indices]

    plt.figure(figsize=(6, 4))
    plt.scatter(training_times, accuracies, label='All Trials', color='gray', alpha=0.5)
    plt.plot(pareto_times, pareto_accuracies, label='Pareto Frontier', color='black', marker='o', linestyle='-')
    plt.xlabel('Training Time per Batch (seconds)')
    plt.ylabel('Validation Accuracy')
    plt.title(f"Pareto Frontier of Training Time vs. Accuracy\n{title_suffix}")
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig(output_filename)



    print("Pareto-optimal points (Trial ID, Accuracy, Training Time):")
    for trial_id, acc, time in zip(pareto_trial_ids, pareto_accuracies, pareto_times):
        print(f"  {trial_id}, {acc:.4f}, {time:.2f} seconds")

    # chosen_trial_id = input("Enter the Trial ID (or press Enter to skip): ")
    # if chosen_trial_id:
    #     model_path = os.path.join(directory, chosen_trial_id, 'best_model')
    #     if os.path.exists(model_path):
    #         try:
    #             loaded_model = tf.keras.models.load_model(model_path)
    #             print(f"Model from Trial ID {chosen_trial_id} loaded.")
    #             _, (x_test, y_test) = tf.keras.datasets.mnist.load_data()  # Or your data loading
    #             x_test = x_test.astype('float32') / 255.0
    #             loss, accuracy = loaded_model.evaluate(x_test, y_test, verbose=0)
    #             print(f"  Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
    #         except Exception as e:
    #             print(f"Error loading model: {e}")
    #     else:
    #         print(f"Error: Model not found at {model_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot pareto frontier from keras tuner."
    )
    parser.add_argument(
        "--kt_directory",
        default="kerastuner/myproj",
        help="Path to the keras tuner project directory (default: kerastuner/myproj)",
    )
    parser.add_argument(
        "--title_suffix",
        default="Model X: Protocol Y",
        help="The second line of the title on the plot",
    )
    parser.add_argument(
        "--output_filename",
        default="pareto_frontier.pdf",
        help="Name of the output filename",
    )
    args = parser.parse_args()

    load_and_plot_results(args.kt_directory, args.title_suffix, args.output_filename)


if __name__ == "__main__":
    main()
