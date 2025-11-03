import tensorflow as tf
import numpy as np
import sionna as sn
import pickle
from src.system import System
from pathlib import Path
import matplotlib
matplotlib.use("Agg")

BATCH_SIZE = 32
EBN0_DB_MIN = -3
EBN0_DB_MAX = 7

# Evaluation: instantiate fresh inference model, load weights, plot BER
eval_system = System(training=False, use_neural_rx=True)

# Build eval model & load weights
_ = eval_system(tf.constant(1, tf.int32), tf.fill([1], tf.constant(10.0, tf.float32)))
with open('weights-ofdm-neuralrx', 'rb') as f:
    weights = pickle.load(f)
    eval_system.set_weights(weights)

# ---- Minimal wrapper for PlotBER: fix kwarg name to 'ebno_db' ----
@tf.function(
    reduce_retracing=True,
    input_signature=[
        tf.TensorSpec([], tf.int32),    # scalar batch_size
        tf.TensorSpec([], tf.float32),  # scalar ebno_db
    ],
)
def mc_fun(batch_size, ebno_db):
    ebno_vec = tf.fill([batch_size], ebno_db)  # expand to shape (BATCH_SIZE,)
    return eval_system(batch_size, ebno_vec)   # reuse vector-SNR path

# Compute BER
outdir = Path("results")
outdir.mkdir(parents=True, exist_ok=True)
ber_plots = sn.phy.utils.PlotBER("Advanced neural receiver")
ber, bler = ber_plots.simulate(
    mc_fun,
    ebno_dbs=np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 1),
    batch_size=BATCH_SIZE,
    max_mc_iter=2,
    num_target_block_errors=100,
    target_bler=1e-2,
    soft_estimates=True,
    show_fig=False,
)