# system.py
import tensorflow as tf
from typing import Dict, Any
from sionna.phy import Block
from sionna.phy.utils import ebnodb2no, compute_ber

from .config import Config, BitsPerSym, CDLModel, Direction
from .csi import CSI
from .tx import Tx
from .channel import Channel
from .rx import Rx


class System(Block):
    """
    Tx -> Channel -> Rx wrapped as a Sionna Block.

    Convention:
      - call(batch_size, ebno_db) : only these two inputs (Sionna-style).
      - use_neural_rx is set at __init__ and kept as a class parameter.
    """
    def __init__(self,
                 *,
                 batch_size: int = 32,           # default; you can override per-call
                 direction: Direction = "uplink",
                 perfect_csi: bool = False,
                 cdl_model: CDLModel = "D",
                 delay_spread: float = 300e-9,
                 carrier_frequency: float = 2.6e9,
                 speed: float = 0.0,
                 num_bits_per_symbol: BitsPerSym = BitsPerSym.QPSK,
                 use_neural_rx: bool = False,     # ← moved here
                 name: str = "system"):
        super().__init__(name=name)

        self._default_batch_size = tf.constant(batch_size, dtype=tf.int32)
        self._use_neural_rx = bool(use_neural_rx)

        # Build config & submodules (batch size is provided at call-time)
        self._cfg = Config(direction=direction,
                           perfect_csi=perfect_csi,
                           cdl_model=cdl_model,
                           delay_spread=delay_spread,
                           carrier_frequency=carrier_frequency,
                           speed=speed,
                           num_bits_per_symbol=num_bits_per_symbol)

        # CSI/Tx/Channel/Rx do not need batch size at construction
        self._csi = CSI(self._cfg, batch_size)
        self._tx = Tx(self._cfg, self._csi)
        self._ch = Channel(self._cfg, self._csi)
        self._rx = Rx(self._cfg, self._csi)
        # self._neural_rx = NeuralRx(...)  # hook up when available

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(), dtype=tf.int32),   # batch_size
            tf.TensorSpec(shape=(), dtype=tf.float32), # ebno_db in dB
        ],
        reduce_retracing=True,
    )
    def call(self,
             batch_size: tf.Tensor,
             ebno_db: tf.Tensor) -> Dict[str, Any]:
        """One forward pass for given (batch_size, ebno_db)."""

        # If someone passes a non-positive batch, fall back to default
        B = tf.where(batch_size > 0, batch_size, self._default_batch_size)

        # Noise power
        no = ebnodb2no(ebno_db,
                       self._cfg.num_bits_per_symbol,
                       self._cfg.coderate,
                       self._csi.rg)

        # Tx
        tx_out = self._tx(B)

        # Channel
        y_out = self._ch(B, tx_out["x_rg_tx"], no)

        # Rx (toggle via class parameter)
        if self._use_neural_rx:
            # When a neural receiver is wired in, route here.
            rx_out = self._rx(B, y_out["y"], no, g=tx_out["g"])  # placeholder
        else:
            rx_out = self._rx(B, y_out["y"], no, g=tx_out["g"])

        b, b_hat = tx_out["b"], rx_out["b_hat"]

        return b, b_hat


if __name__ == "__main__":
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    tf.get_logger().setLevel("ERROR")

    system = System(
        batch_size=8,
        direction="uplink",
        perfect_csi=False,
        cdl_model="D",
        delay_spread=300e-9,
        carrier_frequency=2.6e9,
        speed=0.0,
        use_neural_rx=False,
        name="system",
    )

    b, b_hat = system(tf.constant(8, tf.int32), tf.constant(40.0, tf.float32))
    ber = compute_ber(b, b_hat)
    tf.print("BER:", ber)
