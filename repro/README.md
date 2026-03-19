# tf-shell Experimental Results

Results in the paper from commit 17f673d.

Collected tflogs from running tf-shell on g2-standard-90 (8x NVIDIA L4s) and
c3d-standard-90.

The convolutional models were run on the g2-standard-90, and the dense models
were run on the c3d-standard-90.

First the accuracy tests are run with --party b, then distributed tests (with
small num epochs) are run with --party f and --party l on both machines.
The distributed tests are just to get the NetIO and time per batch.

## Errata

The dpsgd conv distributed test needs a modification in
tf_shell_ml/large_tensor.py set the SAFETY_FACTOR to 0.6.

Without this change, tensorflow segfaults with no error message, stack trace,
or core dump (which can be very difficult to debug).

The newer version, hadal-flow (not tf-shell), does not need this.

## Accuracy Tests

Local Runs
```bash
export GPU="--gpu"
export PARTY="--party b"

# Binary dense MNIST
for eps in {0.2,0.3,0.5,1.0,0.0}; do
  python ./mnist-dpsgd-binary.py $PARTY $GPU --epsilon $eps
  python ./mnist-dpsgd-binary.py $PARTY $GPU --epsilon $eps  # run twice for average
  python ./mnist-post-scale-binary.py $PARTY $GPU --epsilon $eps
  python ./mnist-dpsgd-binary.py $PARTY $GPU --epsilon $eps --dp_sgd
done

# Dense MNIST
for eps in {0.2,0.3,0.5,1.0,0.0}; do
  python ./mnist-dpsgd.py $PARTY $GPU --epsilon $eps
  python ./mnist-post-scale.py $PARTY $GPU --epsilon $eps
  python ./mnist-dpsgd.py $PARTY $GPU --epsilon $eps --dp_sgd
done
python ./mnist-dpsgd.py $PARTY $GPU --epsilon 0.0 --backprop_noise_offset 16  # no noise drops one of the moduli because of one fewer mul. rerun with extra nosie offset to fix.

# Convolutional MNIST
for eps in {0.2,0.3,0.5,1.0,0.0}; do
  python ./mnist-post-scale-conv.py $PARTY $GPU --epsilon $eps
  python ./mnist-dpsgd-conv.py $PARTY $GPU --epsilon $eps
  python ./mnist-dpsgd-conv.py $PARTY $GPU --epsilon $eps --dp_sgd
done

# IMDB
for eps in {0.2,0.3,0.5,1.0,0.0}; do
  python ./imdb-dpsgd.py $PARTY $GPU --epsilon $eps
  python ./imdb-post-scale.py $PARTY $GPU --epsilon $eps
  python ./imdb-dpsgd.py $PARTY $GPU --epsilon $eps --dp_sgd
  python ./imdb-dpsgd.py $PARTY $GPU --epsilon $eps --dp_sgd  # re-run for avg
done

# Copy data to local machine.
rsync -chavzP --stats <host>:/home/$USER/hadal-experiments/training/tflogs/ ./tflogs_accuracy/
```

## Distributed Tests

Quick (only 1 epoch) distributed (across two parties) measurements for network
io and timing.
These tests do not train all epochs to save costs.
Model accuracy is not used / important here.

```bash
# On both machines:
export CLUSTER_SPEC='{"tfshellfeatures": ["<host1>"], "tfshelllabels": ["<host2>:2222"]}'
# or for testing with two processes on the same machine instead of two machines:
# export CLUSTER_SPEC='{"tfshellfeatures": ["127.0.0.1:2222"], "tfshelllabels": ["127.0.0.1:2223"]}'

# On the label holding machine:
export PARTY="--party l"

# On the feature holding machine:
export PARTY="--party f"

# Run the following commands on both machines, one at a time.
python ./mnist-post-scale-binary.py --gpu $PARTY --cluster_spec="$CLUSTER_SPEC" --epochs 6
python ./mnist-post-scale.py --gpu $PARTY --cluster_spec="$CLUSTER_SPEC" --epochs 1
python ./mnist-post-scale-conv.py --gpu $PARTY --cluster_spec="$CLUSTER_SPEC" --epochs 1
python ./imdb-post-scale.py $PARTY --cluster_spec="$CLUSTER_SPEC" --epochs 1  # tf-shell v0.1.33 has bug with multiple GPUs, turn off.

python ./mnist-dpsgd-binary.py --gpu $PARTY --cluster_spec="$CLUSTER_SPEC" --epochs 6
python ./mnist-dpsgd.py --gpu $PARTY --cluster_spec="$CLUSTER_SPEC" --epochs 1
python ./mnist-dpsgd-conv.py --gpu $PARTY --cluster_spec="$CLUSTER_SPEC" --epochs 1
python ./imdb-dpsgd.py $PARTY --cluster_spec="$CLUSTER_SPEC" --epochs 1  # tf-shell v0.1.33 has bug with multiple GPUs, turn off.
python ./dog-cat-post-scale-conv.py --gpu $PARTY --cluster_spec="$CLUSTER_SPEC" --epochs 1

# Copy data to local machine.
rsync -chavzP --stats 34.48.60.220:/home/$USER/hadal-experiments/training/tflogs/ ./tflogs_distrib/
```

Measure network bandwidth between the machines.

iperf test

```bash
# Label holding party
iperf -s

# Feature holding party
iperf -c <host2>
```

Result from intra-zone machines on GCP:

```log
------------------------------------------------------------
Server listening on TCP port 5001
TCP window size:  128 KByte (default)
------------------------------------------------------------
[  1] local <host1> port 5001 connected with <host2> port 35816 (icwnd/mss/irtt=13/1408/447)
[ ID] Interval       Transfer     Bandwidth
[  1] 0.0000-10.0012 sec  23.7 GBytes  20.3 Gbits/sec
```

## Eager vs Deferred

```bash
# On the c3d-standard-90 machine.
export GPU=""
for pt in {"","--dp_sgd",}; do
  for em in {"",""--eager_mode}; do
    python ./mnist-post-scale-binary.py $GPU $pt $em --epochs 6
    python ./mnist-post-scale.py        $GPU $pt $em --epochs 1
    # python ./mnist-post-scale-conv.py   $GPU $pt $em --epochs 1  # Run on gpu machine
    python ./imdb-post-scale.py         $GPU $pt $em --epochs 1

    python ./mnist-dpsgd-binary.py      $GPU $pt $em --epochs 6
    python ./mnist-dpsgd.py             $GPU $pt $em --epochs 1
    # python ./mnist-dpsgd-conv.py        $GPU $pt $em --epochs 1  # Run on gpu machine
    python ./imdb-dpsgd.py              $GPU $pt $em --epochs 1
  done
done

# On the l4x8 gpu machine.
export GPU="--gpu"
for pt in {"","--dp_sgd",}; do
  for em in {"",""--eager_mode}; do
    python ./mnist-post-scale-conv.py   $GPU $pt $em --epochs 1
    python ./mnist-dpsgd-conv.py        $GPU $pt $em --epochs 1
  done
done

# Copy data to local machine.
rsync -chavzP --stats <host1>:/home/$USER/hadal-experiments/training/tflogs/ ./tflogs_deferred_speedup/
```

## Keras Tuner

Consider tuning at both epsilon 0.2 and 0.3.

```bash
while true; do sleep 5; python ./imdb-dpsgd.py --epsilon 0.2 --tune; done
while true; do sleep 5; python ./imdb-post-scale.py --epsilon 0.2 --tune; done

# Copy data to local machine.
rsync -chavzP --stats cc@<host1>:/home/cc/hadal-experiments/training/kerastuner/imdb-dpsgd/ ./kerastuner/
rsync -chavzP --stats cc@<host2>:/home/cc/hadal-experiments/training/kerastuner/imdb-post-scale/ ./kerastuner/
```

## Plotting

Plot accuracy vs. epsilon results.

```bash
# Parse the logs
python ./parse_logs.py ./tflogs_accuracy
# Plot accuracy vs epsilon
python ./plot_epsilon_vs_acc.py --model_arch A
python ./plot_epsilon_vs_acc.py --model_arch B
python ./plot_epsilon_vs_acc.py --model_arch C
python ./plot_epsilon_vs_acc.py --model_arch D
# Optional: plot individual experiment epoch vs accuracy
python ./plot_epoch_vs_acc.py --model_arch A --epsilon 0.2
python ./plot_epoch_vs_acc.py --model_arch A --epsilon 0.5
python ./plot_epoch_vs_acc.py --model_arch A --epsilon 1.0
python ./plot_epoch_vs_acc.py --model_arch B --epsilon 1.0
python ./plot_epoch_vs_acc.py --model_arch C --epsilon 1.0
```

Print distributed timing / networking metadata for the tables in the paper.

```bash
# Parse the distributed logs
python ./parse_logs.py ./tflogs_distrib
# Print metadata
python ./print_metadata.py --model_arch A
python ./print_metadata.py --model_arch B
python ./print_metadata.py --model_arch C
```

Plot eager vs deferred timing results.

```bash
# Parse the distributed logs
python ./parse_logs.py ./tflogs_deferred_speedup
# Plot eager vs deferred timing.
python ./plot_eager_speedup.py
```

Plot keras tuner results.

```bash
# No log parsing script. Simply point at the log dir directly.
export KT_BASE="./tf_shell_0.2.0/kerastuner"
python ./plot_kerastuner.py --kt_directory $KT_BASE/mnist-dpsgd/ --title "Model A, Protocol HE-DP-SGD, epsilon 0.2" --output_file "pareto_hypertune_modelA_hedpsgd.png"
python ./plot_kerastuner.py --kt_directory $KT_BASE/mnist-dpsgd-conv/ --title "Model B, Protocol HE-DP-SGD, epsilon 0.2" --output_file "pareto_hypertune_modelB_hedpsgd.png"
python ./plot_kerastuner.py --kt_directory $KT_BASE/mnist-dpsgd-binary/ --title "Model C, Protocol HE-DP-SGD, epsilon 0.2" --output_file "pareto_hypertune_modelC_hedpsgd.png"
python ./plot_kerastuner.py --kt_directory $KT_BASE/imdb-dpsgd/ --title "Model D, Protocol HE-DP-SGD, epsilon 0.2" --output_file "pareto_hypertune_modelD_hedpsgd.png"

python ./plot_kerastuner.py --kt_directory $KT_BASE/mnist-postscale/ --title "Model A, Protocol PostScale, epsilon 0.2" --output_file "pareto_hypertune_modelA_postscale.png"
python ./plot_kerastuner.py --kt_directory $KT_BASE/mnist-postscale-conv/ --title "Model B, Protocol PostScale, epsilon 0.2" --output_file "pareto_hypertune_modelB_postscale.png"
python ./plot_kerastuner.py --kt_directory $KT_BASE/mnist-postscale-binary/ --title "Model C, Protocol PostScale, epsilon 0.2" --output_file "pareto_hypertune_modelC_postscale.png"
python ./plot_kerastuner.py --kt_directory $KT_BASE/imdb-postscale/ --title "Model D, Protocol PostScale, epsilon 0.2" --output_file "pareto_hypertune_modelD_postscale.png"
```

## Choosing Noise Multipliers

Use the convenience script to find the noise multiplier corresponding to a
desired epsilon. The log2_batch_size varies depending on the experiment,
namely if the scaling factor is large or the multiplicative depth of the model
requires a larger ciphertext ring degree (i.e. batch size).

```bash
# MNIST dataset
python ./noise_multiplier_finder.py --target_epochs 10 --log2_batch_size <12 or 13> --samples 60000 --target_delta 1e-5
# Binary MNIST dataset
python ./noise_multiplier_finder.py --target_epochs 10 --log2_batch_size <12 or 13> --samples 11982 --target_delta 1e-4
# IMDB dataset
python ./noise_multiplier_finder.py --target_epochs 10 --log2_batch_size <12 or 13> --samples 15000 --target_delta 1e-5
```

