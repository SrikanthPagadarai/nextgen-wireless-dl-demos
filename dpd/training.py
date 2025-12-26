#!/usr/bin/env python3
"""
Training script for Neural Network DPD using Indirect Learning Architecture.

Usage:
    python training.py --iterations 10000
    python training.py --iterations 5000 --fresh  # Start fresh, ignore checkpoint
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import warnings  # noqa: E402

warnings.filterwarnings("ignore", message=".*complex64.*float32.*")

import tensorflow as tf  # noqa: E402

tf.get_logger().setLevel("ERROR")

# GPU setup - must happen before any TensorFlow operations
gpus = tf.config.list_physical_devices("GPU")
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)
print("GPUs:", tf.config.list_logical_devices("GPU"))

import numpy as np  # noqa: E402
import pickle  # noqa: E402
import argparse  # noqa: E402

from src.system import DPDSystem  # noqa: E402


# CLI
parser = argparse.ArgumentParser(description="Train Neural Network DPD.")
parser.add_argument(
    "--iterations", type=int, default=10000, help="Train for N more iterations"
)
parser.add_argument(
    "--fresh", action="store_true", help="Start fresh (ignore checkpoint)"
)
parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
args = parser.parse_args()

# Training config
BATCH_SIZE = args.batch_size
ACCUMULATION_STEPS = 4
LEARNING_RATE = 1e-3

# Filesystem
os.makedirs("results", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)
ckpt_dir = "checkpoints"

# System
print("Building DPD System...")
system = DPDSystem(
    training=True,
    tx_config_path="src/tx_config.json",
    pa_order=7,
    pa_memory_depth=4,
    dpd_memory_depth=4,
    dpd_num_filters=64,
    dpd_num_layers_per_block=2,
    dpd_num_res_blocks=3,
    rms_input_dbm=0.5,
    pa_sample_rate=122.88e6,
)

# Warm-up (variable creation) - generate signal outside tf.function
print("Warming up model...")
x_warmup = system.generate_signal(BATCH_SIZE)
_ = system.forward_on_signal(x_warmup, training=True)
print(f"Number of trainable variables: {len(system.trainable_variables)}")
print(
    "Total trainable parameters: ",
    f"{sum(tf.size(v).numpy() for v in system.trainable_variables)}",
)

# Estimate PA gain (required for proper indirect learning)
pa_gain = system.estimate_pa_gain()
print(f"Estimated PA gain: {pa_gain:.4f} ({20*np.log10(pa_gain):.2f} dB)")

# Optimizer / checkpoint
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
checkpoint = tf.train.Checkpoint(
    model=system,
    optimizer=optimizer,
)

start_iteration = 0
loss_history = []
latest = tf.train.latest_checkpoint(ckpt_dir)
if not args.fresh and latest:
    checkpoint.restore(latest)
    iter_file = os.path.join(ckpt_dir, "iter.txt")
    loss_file = os.path.join(ckpt_dir, "loss.npy")
    if os.path.exists(iter_file):
        start_iteration = int(open(iter_file).read())
    if os.path.exists(loss_file):
        loss_history = np.load(loss_file).tolist()
    print(f"Resumed from iteration {start_iteration}")

target_iteration = start_iteration + args.iterations
print(f"Training from {start_iteration} to {target_iteration}")


# Train step - takes pre-generated signal, not batch_size
@tf.function(reduce_retracing=True)
def train_step(x):
    with tf.GradientTape() as tape:
        loss = system.forward_on_signal(x, training=True)
    grads = tape.gradient(loss, system.trainable_variables)
    grads = [
        g if g is not None else tf.zeros_like(w)
        for g, w in zip(grads, system.trainable_variables)
    ]
    return loss, grads


# Sanity: accumulation alignment
if start_iteration % ACCUMULATION_STEPS != 0:
    start_iteration = (start_iteration // ACCUMULATION_STEPS) * ACCUMULATION_STEPS
    print(f"Adjusted start_iteration to {start_iteration} for accumulation alignment")

if target_iteration % ACCUMULATION_STEPS != 0:
    target_iteration = (
        (target_iteration // ACCUMULATION_STEPS) + 1
    ) * ACCUMULATION_STEPS
    print(f"Adjusted target_iteration to {target_iteration} for accumulation alignment")

# Training loop
print("\nStarting training...")

# Pre-allocate gradient accumulators with zeros (same shape as trainable variables)
accumulated_grads = [
    tf.Variable(tf.zeros_like(v), trainable=False) for v in system.trainable_variables
]

for i in range(start_iteration, target_iteration):
    # Generate signal OUTSIDE tf.function (in eager mode)
    x = system.generate_signal(BATCH_SIZE)

    # Train step inside tf.function
    loss, grads = train_step(x)

    for acc_g, g in zip(accumulated_grads, grads):
        acc_g.assign_add(g)

    if (i + 1) % ACCUMULATION_STEPS == 0:
        avg_grads = [g / ACCUMULATION_STEPS for g in accumulated_grads]
        optimizer.apply_gradients(zip(avg_grads, system.trainable_variables))
        # Reset accumulators
        for acc_g in accumulated_grads:
            acc_g.assign(tf.zeros_like(acc_g))

    loss_value = float(loss.numpy())
    loss_history.append(loss_value)

    # Progress logging
    print(
        f"\rStep {i + 1}/{target_iteration}  Loss: {loss_value:.6f}",
        end="",
        flush=True,
    )

print("\n\nTraining complete.")

# Save state
checkpoint.save(os.path.join(ckpt_dir, "ckpt"))
open(os.path.join(ckpt_dir, "iter.txt"), "w").write(str(target_iteration))
np.save(os.path.join(ckpt_dir, "loss.npy"), loss_history)
np.save(
    os.path.join("results", "loss.npy"),
    np.array(loss_history, dtype=np.float32),
)

with open(os.path.join("results", "nn-dpd-weights"), "wb") as f:
    pickle.dump(system.get_weights(), f)
print("Saved checkpoints, loss history, and weights.")

# Plot loss history
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.semilogy(loss_history)
    plt.title("NN DPD Training Loss (Indirect Learning)")
    plt.xlabel("Iteration")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.savefig(os.path.join("results", "training_loss.png"), dpi=150)
    plt.close()
    print("Saved training loss plot to results/training_loss.png")
except ImportError:
    print("matplotlib not available, skipping loss plot")
