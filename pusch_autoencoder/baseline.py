import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sionna.phy.utils import PlotBER
from sionna.phy.channel import CIRDataset
from cir_generator import CIRGenerator

from config import Config
from system import PUSCHLinkE2E
from cir import build_channel_model

_cfg = Config()
batch_size = _cfg.batch_size
num_bs = _cfg.num_bs
num_ue = _cfg.num_ue
num_bs_ant = _cfg.num_bs_ant
num_ue_ant = _cfg.num_ue_ant
num_time_steps = _cfg.num_time_steps

# Build channel model
# channel_model = build_channel_model()

# CIRDataset construction
# Load all CIRs ('a' and 'tau') from TFRecord files into tensors
cir_dir = os.path.join(os.path.dirname(__file__), "cir_tfrecords")
cir_files = tf.io.gfile.glob(os.path.join(cir_dir, "*.tfrecord"))

if not cir_files:
    raise ValueError(f"No TFRecord files found in {cir_dir}")

# Assumes that 'a' and 'tau' were written using tf.io.serialize_tensor
# and that 'a' is complex64 and 'tau' is float32.
feature_description = {
    "a": tf.io.FixedLenFeature([], tf.string),
    "tau": tf.io.FixedLenFeature([], tf.string),
    "a_shape": tf.io.VarLenFeature(tf.int64),
    "tau_shape": tf.io.VarLenFeature(tf.int64),
}

def _parse_example(example_proto):
    parsed = tf.io.parse_single_example(example_proto, feature_description)

    # Deserialize tensors
    a = tf.io.parse_tensor(parsed["a"], out_type=tf.complex64)
    tau = tf.io.parse_tensor(parsed["tau"], out_type=tf.float32)

    # Read shape metadata (sparse -> dense)
    a_shape = tf.sparse.to_dense(parsed["a_shape"])
    tau_shape = tf.sparse.to_dense(parsed["tau_shape"])

    # Ensure correct shapes (mirroring cir.py)
    a = tf.reshape(a, a_shape)
    tau = tf.reshape(tau, tau_shape)

    return a, tau


ds = tf.data.TFRecordDataset(cir_files)
ds = ds.map(_parse_example)

all_a = []
all_tau = []
for a, tau in ds:
    # Each record is expected to have shapes like:
    # a:  (N_i, 1, 16, 1, 4, 13, 14)
    # tau:(N_i, 1, 1, 13)
    all_a.append(a)
    all_tau.append(tau)

all_a = tf.concat(all_a, axis=0)
all_tau = tf.concat(all_tau, axis=0)
all_a = tf.expand_dims(all_a, axis=1)
all_tau = tf.expand_dims(all_tau, axis=1)
print("(in init) all_a.shape: ", all_a.shape)
print("(in init) all_tau.shape: ", all_tau.shape)

cir_generator = CIRGenerator(all_a, all_tau, num_ue)
channel_model = CIRDataset(
    cir_generator,
    batch_size,
    num_bs,
    num_bs_ant,
    num_ue,
    num_ue_ant,
    13,
    num_time_steps
)

# Quick functional check
ebno_db = 10.
e2e_model = PUSCHLinkE2E(channel_model, perfect_csi=False)
# e2e_model = PUSCHLinkE2E(perfect_csi=False)

# We can draw samples from the end-2-end link-level simulations
b, b_hat = e2e_model(batch_size, ebno_db)

# SNR sweep
ebno_db = np.arange(-3, 18, 2)

# create the BER/BLER simulator
ber_plot = PlotBER("Site-Specific MU-MIMO 5G NR PUSCH")

# compute BER/BLER results
ber_list, bler_list = [], []
for perf_csi in [True, False]:
    e2e_model = PUSCHLinkE2E(channel_model, perfect_csi=perf_csi)
    # e2e_model = PUSCHLinkE2E(perfect_csi=perf_csi)

    ber_i, bler_i = ber_plot.simulate(
        e2e_model,
        ebno_dbs=ebno_db,
        max_mc_iter=50,
        num_target_block_errors=200,
        batch_size=batch_size,
        soft_estimates=False,
        show_fig=False,
        add_bler=True,
    )

    # Ensure NumPy arrays
    ber_list.append(ber_i.numpy() if hasattr(ber_i, "numpy") else np.asarray(ber_i))
    bler_list.append(bler_i.numpy() if hasattr(bler_i, "numpy") else np.asarray(bler_i))

# Stack to arrays with shape [2, len(ebno_db)]
ber = np.stack(ber_list, axis=0)
bler = np.stack(bler_list, axis=0)

# Plot BLER only (two curves: Perf./Imperf. CSI)
os.makedirs("results", exist_ok=True)
plt.figure()
for idx, csi_label in enumerate(["Perfect CSI", "Imperfect CSI"]):
    plt.semilogy(ebno_db, bler[idx], marker="o", linestyle="-", label=f"LMMSE {csi_label}")

plt.xlabel("Eb/N0 [dB]")
plt.ylabel("BLER")
plt.title("PUSCH - BLER vs Eb/N0")
plt.grid(True, which="both")
plt.legend()

outfile = os.path.join(
    "results",
    f"bler_plot_bs{batch_size}_ue{num_ue}_ant{num_bs_ant}x{num_ue_ant}.png"
)
plt.savefig(outfile, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved BLER plot to {outfile}")
