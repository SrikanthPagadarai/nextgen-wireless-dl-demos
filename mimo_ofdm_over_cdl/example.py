"""
Single-iteration simulation using shared CSI:
Tx -> Channel -> Rx
"""

import tensorflow as tf
from sionna.phy.utils import ebnodb2no, compute_ber
from src import Config, CSI, Tx, Channel, Rx

# 1. Choose direction & CSI behavior
cfg = Config(direction="downlink", perfect_csi=True)
B = tf.constant(8, dtype=tf.int32)
EbNo_dB = tf.constant(10.0)

# 2. Build shared CSI (h_freq created inside)
csi = CSI(cfg, batch_size=B)

# 3. Create modules that share the same CSI
tx = Tx(cfg, csi)
ch = Channel(cfg, csi)
rx = Rx(cfg, csi)

# 4. Transmitter stage
tx_out = tx(B)  # uses csi.h_freq internally for precoding if downlink

# 5. Compute noise variance
no = ebnodb2no(EbNo_dB, cfg.num_bits_per_symbol, cfg.coderate, cfg.rg)
no = tf.constant(0.0, tf.float32)

# 6. Channel stage (applies same h_freq as CSI)
y_out = ch(B, tx_out["x_rg_tx"], no)

# 7. Receiver stage
rx_out = rx(B, y_out["y"], no, g=tx_out["g"])
ber = compute_ber(tx_out['b'], rx_out['b_hat'])

# 8. Display shapes for verification
print("\n===== SIMULATION SUMMARY =====")
print(f"Direction: {cfg.direction}, Perfect CSI: {cfg.perfect_csi}")
print(f"h_freq shape: {csi.h_freq.shape}")
# print(f"g shape: {tx_out["g"].shape}")
print(f"Tx output keys: {list(tx_out.keys())}")
print(f"Channel output keys: {list(y_out.keys())}")
print(f"Rx output keys: {list(rx_out.keys())}")

print("\nSample tensor shapes:")
print(f"y : {y_out['y'].shape}")
print(f"b_hat : {rx_out['b_hat'].shape}")
print("BER: {}".format(ber))
