import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sionna.phy.utils import PlotBER

from config import Config
from system import PUSCHLinkE2E
from cir_manager import CIRManager

# ============================================================================
# TensorFlow and GPU Configuration
# ============================================================================
def setup_tensorflow():
    """Configure TensorFlow and GPU settings."""
    # Set GPU device if not already specified
    if os.getenv("CUDA_VISIBLE_DEVICES") is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # Suppress TensorFlow info/warning logs
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.get_logger().setLevel('ERROR')
    
    # Configure GPU memory growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")

# Run setup on import
setup_tensorflow()

# ============================================================================
# Configuration and Channel Model Setup
# ============================================================================

# Get configuration
_cfg = Config()
batch_size = _cfg.batch_size
num_bs = _cfg.num_bs
num_ue = _cfg.num_ue
num_bs_ant = _cfg.num_bs_ant
num_ue_ant = _cfg.num_ue_ant
num_time_steps = _cfg.num_time_steps

# Build channel model
cir_manager = CIRManager()
channel_model = cir_manager.build_channel_model()

# ============================================================================
# Quick Functional Check
# ============================================================================

# Quick functional check
ebno_db = 10.
e2e_model = PUSCHLinkE2E(channel_model, perfect_csi=False)

# We can draw samples from the end-2-end link-level simulations
b, b_hat = e2e_model(batch_size, ebno_db)

# ============================================================================
# BER/BLER Simulation
# ============================================================================

# SNR sweep
ebno_db = np.arange(-3, 18, 2)

# create the BER/BLER simulator
ber_plot = PlotBER("Site-Specific MU-MIMO 5G NR PUSCH")

# compute BER/BLER results
ber_list, bler_list = [], []
for perf_csi in [True, False]:
    e2e_model = PUSCHLinkE2E(channel_model, perfect_csi=perf_csi)

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

# ============================================================================
# Results Plotting and Saving
# ============================================================================

# Plot BLER
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