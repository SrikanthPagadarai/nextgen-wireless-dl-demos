"""DPD System for training and inference.

Wraps signal generation, upsampling, DPD, and PA into a single differentiable system.
Returns the indirect learning loss for training.
"""

import json
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.layers import Layer
from sionna.phy.ofdm import OFDMDemodulator

from .nn_dpd import NeuralNetworkDPD
from .power_amplifier import PowerAmplifier
from .interpolator import Interpolator
from .tx import Tx
from .utilities import normalize_to_rms


class DPDSystem(Layer):
    """
    Complete DPD system for training and inference.

    Wraps signal generation (Sionna Tx), upsampling, DPD, and PA.
    Returns indirect learning loss: ||DPD(PA_output/G) - predistorter_output||²

    The indirect learning approach:
    1. Generate baseband signal x
    2. Upsample to PA sample rate
    3. Apply predistorter: u = DPD(x)
    4. Pass through PA: y = PA(u)
    5. Normalize by PA gain: y_norm = y / G
    6. Train postdistorter: loss = ||DPD(y_norm) - u||²

    Args:
        training: Whether in training mode
        tx_config_path: Path to transmitter configuration JSON
        pa_order: PA polynomial order (default: 7)
        pa_memory_depth: PA memory depth (default: 4)
        dpd_memory_depth: DPD memory depth (default: 4)
        dpd_num_filters: DPD hidden layer size (default: 64)
        dpd_num_layers_per_block: Layers per residual block (default: 2)
        dpd_num_res_blocks: Number of residual blocks (default: 3)
        rms_input_dbm: Target input RMS power in dBm (default: 0.5)
        pa_sample_rate: PA sample rate in Hz (default: 122.88e6)
    """

    def __init__(
        self,
        training: bool,
        tx_config_path: str = "src/tx_config.json",
        pa_order: int = 7,
        pa_memory_depth: int = 4,
        dpd_memory_depth: int = 4,
        dpd_num_filters: int = 64,
        dpd_num_layers_per_block: int = 2,
        dpd_num_res_blocks: int = 3,
        rms_input_dbm: float = 0.5,
        pa_sample_rate: float = 122.88e6,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._training = training
        self._tx_config_path = tx_config_path
        self._rms_input_dbm = rms_input_dbm
        self._pa_sample_rate = pa_sample_rate

        # Load configuration
        self._tx_config = json.loads(Path(tx_config_path).read_text())

        # Compute signal sample rate from FFT size and subcarrier spacing
        fft_size = float(self._tx_config["rg"]["fft_size"])
        subcarrier_spacing = float(self._tx_config["rg"]["subcarrier_spacing"])
        self._signal_fs = fft_size * subcarrier_spacing

        # Build transmitter once (not inside tf.function)
        self._tx = Tx(tx_config_path)

        # Interpolator for upsampling
        self._interpolator = Interpolator(
            input_rate=self._signal_fs,
            output_rate=self._pa_sample_rate,
        )

        # Power Amplifier
        self._pa = PowerAmplifier(order=pa_order, memory_depth=pa_memory_depth)

        # Neural Network DPD
        self._dpd = NeuralNetworkDPD(
            memory_depth=dpd_memory_depth,
            num_filters=dpd_num_filters,
            num_layers_per_block=dpd_num_layers_per_block,
            num_res_blocks=dpd_num_res_blocks,
        )

        # Loss function with scaling for better gradient flow
        self._loss_fn = tf.keras.losses.MeanSquaredError()
        self._loss_scale = 1000.0  # Scale up loss for better monitoring

        # PA gain (estimated during first forward pass)
        self._pa_gain = tf.Variable(1.0, trainable=False, dtype=tf.float32)
        self._pa_gain_initialized = False

        # Demodulation setup for constellation plotting
        self._fft_size = int(self._tx_config["rg"]["fft_size"])
        self._cp_length = int(self._tx_config["rg"]["cyclic_prefix_length"])
        self._num_ofdm_symbols = int(self._tx_config["rg"]["num_ofdm_symbols"])
        self._num_guard_lower = int(self._tx_config["rg"]["num_guard_carriers"][0])
        self._num_guard_upper = int(self._tx_config["rg"]["num_guard_carriers"][1])
        self._dc_null = bool(self._tx_config["rg"]["dc_null"])

        # Sionna demodulator
        self._ofdm_demod = OFDMDemodulator(
            fft_size=self._fft_size,
            l_min=0,
            cyclic_prefix_length=self._cp_length,
        )

        # Subcarrier indices for data extraction
        self._lower_start = self._num_guard_lower
        self._lower_end = self._fft_size // 2
        self._upper_start = self._fft_size // 2 + (1 if self._dc_null else 0)
        self._upper_end = self._fft_size - self._num_guard_upper

    @property
    def dpd(self):
        """Access the DPD layer."""
        return self._dpd

    @property
    def pa(self):
        """Access the PA layer."""
        return self._pa

    @property
    def interpolator(self):
        """Access the interpolator."""
        return self._interpolator

    @property
    def tx_config(self):
        """Access transmitter configuration."""
        return self._tx_config

    @property
    def signal_fs(self):
        """Signal sample rate."""
        return self._signal_fs

    @property
    def pa_sample_rate(self):
        """PA sample rate."""
        return self._pa_sample_rate

    @property
    def pa_gain(self):
        """Estimated PA gain."""
        return self._pa_gain

    def estimate_pa_gain(self, num_samples=10000):
        """
        Estimate PA small-signal gain by measuring input/output power ratio.

        This should be called once before training to determine G.

        Args:
            num_samples: Number of samples to use for estimation

        Returns:
            Estimated gain (linear scale)
        """
        # Generate a test signal with low amplitude (linear region)
        test_input = tf.complex(
            tf.random.normal([num_samples], stddev=0.1),
            tf.random.normal([num_samples], stddev=0.1),
        )

        # Pass through PA
        test_output = self._pa(test_input)

        # Compute gain as sqrt(output_power / input_power)
        input_power = tf.reduce_mean(tf.abs(test_input) ** 2)
        output_power = tf.reduce_mean(tf.abs(test_output) ** 2)
        gain = tf.sqrt(output_power / (input_power + 1e-12))

        self._pa_gain.assign(gain)
        self._pa_gain_initialized = True

        return float(gain.numpy())

    def generate_signal(self, batch_size, return_extras=False):
        """
        Generate a batch of baseband signals.

        Args:
            batch_size: Number of signals to generate (Python int or tf.Tensor)
            return_extras: If True, return additional info for constellation plotting

        Returns:
            If return_extras=False: tx_upsampled [B, num_samples]
            If return_extras=True: dict with tx_upsampled, tx_baseband, x_rg, fd_symbols
        """
        # Call pre-built transmitter
        tx_out = self._tx(batch_size)
        x_time = tx_out["x_time"]  # [B, 1, 1, num_samples]
        x_rg = tx_out["x_rg"]  # [B, 1, 1, num_symbols, fft_size]

        # Remove singleton dimensions: [B, num_samples]
        tx = tf.squeeze(x_time, axis=(1, 2))

        # Keep baseband copy for constellation sync (flattened)
        tx_baseband = tf.reshape(x_time, [-1])

        # Normalize to target RMS
        tx_normalized, _ = normalize_to_rms(tx, self._rms_input_dbm)

        # Upsample to PA rate
        tx_upsampled, _ = self._interpolator(tx_normalized)

        if not return_extras:
            return tx_upsampled

        # Extract frequency-domain symbols for constellation comparison
        # x_rg shape: [B, 1, 1, num_symbols, fft_size] -> [num_symbols, fft_size]
        x_rg_squeezed = tf.squeeze(
            x_rg[0], axis=(0, 1)
        )  # First batch, [num_sym, fft_size]

        # Extract data subcarriers
        fd_lower = tf.transpose(x_rg_squeezed[:, self._lower_start : self._lower_end])
        fd_upper = tf.transpose(x_rg_squeezed[:, self._upper_start : self._upper_end])
        fd_symbols = tf.concat(
            [fd_lower, fd_upper], axis=0
        )  # [num_subcarriers, num_symbols]

        return {
            "tx_upsampled": tx_upsampled,
            "tx_baseband": tx_baseband,
            "x_rg": x_rg,
            "fd_symbols": fd_symbols,
        }

    def call(self, batch_size, training=None):
        """
        Forward pass through the DPD system.

        In training mode, returns the indirect learning loss.
        In inference mode, returns dict with PA outputs (with and without DPD).

        Args:
            batch_size: Batch size (scalar int32 tensor or Python int)
            training: Override training mode if specified

        Returns:
            Training: scalar loss value
            Inference: dict with 'pa_input', 'pa_output_no_dpd', 'pa_output_with_dpd'
        """
        is_training = training if training is not None else self._training

        # Generate and upsample signal
        x = self.generate_signal(batch_size)

        if is_training:
            return self._training_forward(x)
        else:
            return self._inference_forward(x)

    def forward_on_signal(self, x, training=None):
        """
        Forward pass on pre-generated signal (for use with tf.function).

        Signal generation with Sionna cannot run inside tf.function,
        so this method accepts a pre-generated signal.

        Args:
            x: [B, num_samples] complex tensor at PA sample rate
            training: Override training mode if specified

        Returns:
            Training: scalar loss value
            Inference: dict with 'pa_input', 'pa_output_no_dpd', 'pa_output_with_dpd'
        """
        is_training = training if training is not None else self._training

        if is_training:
            return self._training_forward(x)
        else:
            return self._inference_forward(x)

    def _normalize_to_unit_power(self, x):
        """Normalize signal to unit power, return (normalized, scale_factor)."""
        power = tf.reduce_mean(tf.abs(x) ** 2)
        scale = tf.sqrt(power + 1e-12)
        return x / tf.cast(scale, x.dtype), scale

    def _training_forward(self, x):
        """
        Training forward pass with indirect learning.

        Args:
            x: [B, num_samples] input signal at PA rate

        Returns:
            loss: scalar MSE loss
        """
        # Normalize input to unit power for better NN conditioning
        x_norm, x_scale = self._normalize_to_unit_power(x)

        # Step 1: Apply predistorter (stop gradient - this is our target)
        u_norm = self._dpd(x_norm, training=False)
        u_target = tf.stop_gradient(u_norm)

        # Scale back for PA (PA expects original power level)
        u = u_norm * tf.cast(x_scale, u_norm.dtype)

        # Step 2: Pass through PA
        y = self._pa(u)

        # Step 3: Compensate for PA gain
        y_compensated = y / tf.cast(self._pa_gain, y.dtype)

        # Normalize PA output to unit power for postdistorter
        y_norm, _ = self._normalize_to_unit_power(y_compensated)

        # Step 4: Apply postdistorter (this is what we're training)
        u_hat_norm = self._dpd(y_norm, training=True)

        # Step 5: Compute loss in normalized domain
        # Split into real/imag for MSE computation
        u_target_ri = tf.stack(
            [tf.math.real(u_target), tf.math.imag(u_target)], axis=-1
        )
        u_hat_ri = tf.stack(
            [tf.math.real(u_hat_norm), tf.math.imag(u_hat_norm)], axis=-1
        )

        loss = self._loss_fn(u_target_ri, u_hat_ri) * self._loss_scale

        return loss

    def _inference_forward(self, x):
        """
        Inference forward pass.

        Args:
            x: [B, num_samples] input signal at PA rate

        Returns:
            dict with PA input and outputs
        """
        # PA output without DPD
        pa_output_no_dpd = self._pa(x)

        # Normalize for DPD, apply DPD, scale back
        x_norm, x_scale = self._normalize_to_unit_power(x)
        x_predistorted_norm = self._dpd(x_norm, training=False)
        x_predistorted = x_predistorted_norm * tf.cast(
            x_scale, x_predistorted_norm.dtype
        )

        # Pass through PA
        pa_output_with_dpd = self._pa(x_predistorted)

        return {
            "pa_input": x,
            "pa_output_no_dpd": pa_output_no_dpd,
            "pa_output_with_dpd": pa_output_with_dpd,
            "predistorted": x_predistorted,
        }

    def demod(self, signal):
        """
        Demodulate OFDM signal to extract frequency-domain symbols.

        Args:
            signal: [num_samples] complex tensor at baseband sample rate

        Returns:
            [num_subcarriers, num_symbols] complex tensor
        """
        if not isinstance(signal, tf.Tensor):
            signal = tf.constant(signal, dtype=tf.complex64)

        # Reshape for Sionna demodulator: [batch, rx, tx, samples]
        signal_4d = tf.reshape(signal, [1, 1, 1, -1])

        # Demodulate
        rg = self._ofdm_demod(signal_4d)[0, 0, 0, :, :]  # [num_symbols, fft_size]

        # Extract data subcarriers
        fd_lower = tf.transpose(rg[:, self._lower_start : self._lower_end])
        fd_upper = tf.transpose(rg[:, self._upper_start : self._upper_end])

        return tf.concat([fd_lower, fd_upper], axis=0)

    def get_config(self):
        """Return layer configuration for serialization."""
        config = super().get_config()
        config.update(
            {
                "training": self._training,
                "tx_config_path": self._tx_config_path,
                "rms_input_dbm": self._rms_input_dbm,
                "pa_sample_rate": self._pa_sample_rate,
            }
        )
        return config
