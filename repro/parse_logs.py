import os
import glob
import pandas as pd
import tensorflow as tf
from collections import defaultdict
from tensorboard.backend.event_processing import event_accumulator
from tensorboard.plugins.hparams.plugin_data_pb2 import HParamsPluginData

def parse_event_file(exp_path, exp_name):
    # Initialize data for this experiment
    exp_data = defaultdict(dict)
    exp_data['experiment_name'] = exp_name

    # Parse the hyperparameters.
    meta_paths = tf.io.gfile.glob(os.path.join(exp_path, "events.out.tfevents.*"))
    if len(meta_paths) == 1:
        ea = event_accumulator.EventAccumulator(meta_paths[0], size_guidance={'tensors': 0})
        ea.Reload()
        hpdata = ea._plugin_to_tag_to_content["hparams"]["_hparams_/session_start_info"]
        hparam_data = HParamsPluginData.FromString(hpdata).session_start_info.hparams
        hparam_dict = {key: hparam_data[key].ListFields()[0][1] for key in hparam_data.keys()}
        exp_data.update(hparam_dict)
    else:
        print(f"Warning: No hyperparameters found for {exp_name}")

    val_path = os.path.join(exp_path, 'validation')
    for event_file in tf.compat.v1.train.summary_iterator(
        tf.io.gfile.glob(os.path.join(val_path, "events.out.tfevents.*"))[0]
    ):
        for value in event_file.summary.value:
            if value.tag == 'noise_multiplier':
                exp_data['noise_multiplier'] = tf.make_ndarray(value.tensor).item()
            if value.tag == 'learning_rate':
                exp_data['learning_rate'] = tf.make_ndarray(value.tensor).item()
            if value.tag == 'party':
                exp_data['party'] = tf.make_ndarray(value.tensor).item().decode('utf-8')
            if value.tag == 'gpu_enabled':
                exp_data['gpu_enabled'] = tf.make_ndarray(value.tensor).item()
            if value.tag == 'num_gpus':
                exp_data['num_gpus'] = tf.make_ndarray(value.tensor).item()
            if value.tag == 'target_delta':
                exp_data['target_delta'] = tf.make_ndarray(value.tensor).item()
            if value.tag == 'training_num_samples':
                exp_data['training_num_samples'] = tf.make_ndarray(value.tensor).item()
            if value.tag == 'planned_epochs':
                exp_data['planned_epochs'] = tf.make_ndarray(value.tensor).item()
            if value.tag == 'backprop_cleartext_sz':
                exp_data['backprop_cleartext_sz'] = tf.make_ndarray(value.tensor).item()
            if value.tag == 'backprop_scaling_factor':
                exp_data['backprop_scaling_factor'] = tf.make_ndarray(value.tensor).item()
            if value.tag == 'backprop_noise_offset':
                exp_data['backprop_noise_offset'] = tf.make_ndarray(value.tensor).item()
            if value.tag == 'noise_cleartext_sz':
                exp_data['noise_cleartext_sz'] = tf.make_ndarray(value.tensor).item()
            if value.tag == 'noise_noise_offset':
                exp_data['noise_noise_offset'] = tf.make_ndarray(value.tensor).item()
            if value.tag == 'eager_mode':
                exp_data['eager_mode'] = tf.make_ndarray(value.tensor).item()
            if value.tag == 'check_overflow_INSECURE':
                exp_data['check_overflow_INSECURE'] = tf.make_ndarray(value.tensor).item()
            if value.tag == 'disable_he_backprop_INSECURE':
                exp_data['disable_he_backprop_INSECURE'] = tf.make_ndarray(value.tensor).item()
            if value.tag == 'disable_masking_INSECURE':
                exp_data['disable_masking_INSECURE'] = tf.make_ndarray(value.tensor).item()
            if value.tag == 'simple_noise_INSECURE':
                exp_data['simple_noise_INSECURE'] = tf.make_ndarray(value.tensor).item()
            if value.tag == 'batch_size':
                exp_data['batch_size'] = tf.make_ndarray(value.tensor).item()
            if value.tag == 'bytes_sent':
                exp_data['bytes_sent'] = tf.make_ndarray(value.tensor).item()
            if value.tag == 'bytes_recv':
                exp_data['bytes_recv'] = tf.make_ndarray(value.tensor).item()
            if value.tag == 'dp_epsilon':
                exp_data['dp_epsilon'] = tf.make_ndarray(value.tensor).item()
            if value.tag == 'epoch_categorical_accuracy':
                if exp_data.get('epoch_categorical_accuracy') is None:
                    exp_data['epoch_categorical_accuracy'] = tf.make_ndarray(value.tensor).item()
                else:
                    exp_data['epoch_categorical_accuracy'] = max(
                        exp_data['epoch_categorical_accuracy'],
                        tf.make_ndarray(value.tensor).item()
                    )


    # Check if epoch_categorical_accuracy was found
    if 'epoch_categorical_accuracy' not in exp_data:
        print(f"Warning: No epoch_categorical_accuracy found for {exp_name}. Did training fail?")

    # Get the smallest batch time from the training logs
    exp_data['max_batch_steps_per_second'] = 0
    training_path = os.path.join(exp_path, 'train')
    for event_file in tf.compat.v1.train.summary_iterator(
        tf.io.gfile.glob(os.path.join(training_path, "events.out.tfevents.*"))[0]
    ):
        for value in event_file.summary.value:
            if value.tag == 'batch_steps_per_second':
                exp_data['max_batch_steps_per_second'] = max(tf.make_ndarray(value.tensor).item(), exp_data['max_batch_steps_per_second'])

    # Set the model architecture
    if 'conv' in exp_name and not 'dog' in exp_name:
        exp_data['model_arch'] = 'B'
    elif 'binary' in exp_name:
        exp_data['model_arch'] = 'C'
    elif 'imdb' in exp_name:
        exp_data['model_arch'] = 'D'
    elif 'dog' in exp_name:
        exp_data['model_arch'] = 'E'
    elif 'bert' in exp_name and 'finetune' not in exp_name:
        exp_data['model_arch'] = 'F'
    elif 'bert-finetune' in exp_name:
        exp_data['model_arch'] = 'G'
    else:
        exp_data['model_arch'] = 'A'
    
    # Set the protocol
    if 'dpsgd' in exp_name:
        if exp_data['disable_he_backprop_INSECURE'] and exp_data['disable_masking_INSECURE'] and exp_data['simple_noise_INSECURE']:
            exp_data['protocol'] = 'DP-SGD'
        elif not exp_data['disable_he_backprop_INSECURE'] and not exp_data['disable_masking_INSECURE']:  # simple noise is True for epsilon inf (set inside model)
            exp_data['protocol'] = 'HE-DP-SGD'
        else:
            exp_data['protocol'] = 'UNKNOWN DP-SGD'
    elif 'post-scale' in exp_name:
        if not exp_data['disable_he_backprop_INSECURE'] and not exp_data['disable_masking_INSECURE']:  # simple noise is True for epsilon inf (set inside model)
            exp_data['protocol'] = 'PostScale'
        else:
            exp_data['protocol'] = 'UNKNOWN PostScale'

    # Compute the per-batch network IO
    if type(exp_data['bytes_sent']) is float and type(exp_data['bytes_recv']) is float:
        BYTES_PER_MB = 1024 * 1024
        steps_per_epoch = exp_data['training_num_samples'] // exp_data['batch_size']
        exp_data['mb_sent_per_batch'] = exp_data['bytes_sent'] / exp_data['planned_epochs'] / steps_per_epoch / BYTES_PER_MB
        exp_data['mb_recv_per_batch'] = exp_data['bytes_recv'] / exp_data['planned_epochs'] / steps_per_epoch / BYTES_PER_MB

    return exp_data

def parse_tf_events(log_dirs):
    """
    Parse TensorFlow event logs to extract epsilon and accuracy metrics.
    
    Args:
        log_dir (str): Root directory containing experiment folders
    
    Returns:
        list: List of dictionaries containing parsed data
    """
    results = []
    
    # Iterate through experiment directories
    for log_dir in log_dirs:
        for exp_dir in os.listdir(log_dir):
            exp_path = os.path.join(log_dir, exp_dir)


            # Paths when using keras tuner and without are different.
            val_path = os.path.join(exp_path, "validation")
            train_path = os.path.join(exp_path, "train")
            if os.path.isdir(val_path) and os.path.isdir(train_path):
                # When not using keras tuner, validation directory is inside
                # the experiment directory.
                exp_data = parse_event_file(exp_path, exp_dir)
                results.append(exp_data)
                continue
            else:
                # When using keras tuner, we need to iterate through each
                # trial and run directory to find the validation directory.
                for trial_dir in os.listdir(exp_path):
                    trial_path = os.path.join(exp_path, trial_dir)
                    if not os.path.isdir(trial_path):
                        continue

                    for run_dir in os.listdir(trial_path):
                        run_path = os.path.join(trial_path, run_dir)
                        if not os.path.isdir(run_path):
                            continue

                        # Make sure both paths exist.
                        val_path = os.path.join(run_path, "validation")
                        train_path = os.path.join(run_path, "train")
                        if os.path.isdir(val_path) and os.path.isdir(train_path):
                            # Initialize data for this experiment
                            exp_data = parse_event_file(run_path, exp_dir)
                            results.append(exp_data)
    
    return results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Parse TensorFlow event logs to CSV')
    parser.add_argument('log_dir', nargs='+', help='Top level directory(ies) containing TensorFlow event files (usually tflogs).')
    parser.add_argument('--output', '-o', default='metrics.csv',
                      help='Output CSV file path (default: metrics.csv)')
    args = parser.parse_args()

    # Set the path to your TensorFlow logs directory
    logs_dir = args.log_dir
    
    # Parse the events
    parsed_data = parse_tf_events(logs_dir)
    
    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(parsed_data)
    print(df)
    
    # Sort by experiment name and timestamp
    df = df.sort_values(['experiment_name'])
    
    # Save to CSV
    df.to_csv(args.output, index=False)
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()