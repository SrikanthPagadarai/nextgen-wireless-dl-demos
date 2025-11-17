import tensorflow as tf
from tensorflow.keras.layers import Layer, ConvLSTM2D
from .config import Config

class ConvLSTMResBlock(Layer):
    """
    Residual block built from a ConvLSTM2D layer.

    Input / output shape: [B, T, H, W, C]
    where T is the time dimension (OFDM symbols).
    """

    def __init__(
        self,
        filters: int,
        kernel_size=(3, 3),
        name: str = None,
    ):
        super().__init__(name=name)
        self.filters = int(filters)
        self.kernel_size = kernel_size

        self._convlstm = ConvLSTM2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            padding="same",
            return_sequences=True,
            name=None if name is None else name + "_convlstm",
        )
        self._proj = None  # optional projection for channel mismatch

    def build(self, input_shape):
        # input_shape: [B, T, H, W, C]
        in_ch = int(input_shape[-1])
        if in_ch != self.filters:
            # 1x1 ConvLSTM-like projection to match channels
            self._proj = ConvLSTM2D(
                filters=self.filters,
                kernel_size=(1, 1),
                padding="same",
                return_sequences=True,
                name=None if self.name is None else self.name + "_proj",
            )
        super().build(input_shape)

    def call(self, x, training=None):
        # x: [B, T, H, W, C]
        y = self._convlstm(x, training=training)
        if self._proj is not None:
            skip = self._proj(x, training=training)
        else:
            skip = x
        return y + skip


class PUSCHNeuralDetector(Layer):
    """
    PUSCH Neural MIMO-OFDM Detector using ConvLSTM2D + residual blocks.

    Input shapes:
      y      : [B, num_bs, num_bs_ant, H, W] (complex)
      h_hat  : [B, num_bs, num_bs_ant, num_ue, num_streams_per_ue, H, W] (complex)
      err_var: [1, 1, 1, num_ue, num_streams_per_ue, H, W] (float)
      no     : scalar, TensorShape([]) (float)

    Output:
      llr    : [B, num_ue, num_streams_per_ue,
                num_data_symbols * num_bits_per_symbol]
               where num_data_symbols = H * W
    """

    def __init__(
        self,
        cfg: Config,
        num_conv2d_filters: int = 64,
        num_res_blocks: int = 2,
        kernel_size=(3, 3),
    ):
        super().__init__()
        self._cfg = cfg
        self.num_conv2d_filters = int(num_conv2d_filters)
        self.num_res_blocks = int(num_res_blocks)
        self.kernel_size = kernel_size

        # System parameters
        self._num_bits_per_symbol = int(self._cfg.num_bits_per_symbol)
        self._num_ue = int(self._cfg.num_ue)
        self._num_streams_total = int(self._cfg.resource_grid.num_streams_per_tx)
        self._num_streams_per_ue = self._num_streams_total // self._num_ue

        # Input ConvLSTM2D: maps C_in -> num_conv2d_filters
        self._convlstm_in = ConvLSTM2D(
            filters=self.num_conv2d_filters,
            kernel_size=self.kernel_size,
            padding="same",
            return_sequences=True,
            name="convlstm_in",
        )

        # Residual ConvLSTM blocks (all with num_conv2d_filters channels)
        self._res_blocks = [
            ConvLSTMResBlock(
                filters=self.num_conv2d_filters,
                kernel_size=self.kernel_size,
                name=f"convlstm_resblock_{i}",
            )
            for i in range(self.num_res_blocks)
        ]

        # Output ConvLSTM2D: maps -> num_streams_total * num_bits_per_symbol
        self._convlstm_out = ConvLSTM2D(
            filters=self._num_streams_total * self._num_bits_per_symbol,
            kernel_size=self.kernel_size,
            padding="same",
            return_sequences=True,
            name="convlstm_out",
        )

    def call(
        self,
        y: tf.Tensor,
        h_hat: tf.Tensor,
        err_var: tf.Tensor,
        no: tf.Tensor,
        training=None,
    ) -> tf.Tensor:
        """
        y      : [B, num_bs, num_bs_ant, H, W], complex
        h_hat  : [B, num_bs, num_bs_ant, num_ue, num_streams_per_ue, H, W], complex
        err_var: [1, 1, 1, num_ue, num_streams_per_ue, H, W], float
        no     : scalar, TensorShape([]), float

        Returns:
          llr: [B, num_ue, num_streams_per_ue,
                num_data_symbols * num_bits_per_symbol]
        """
        # Ensure dtypes
        y = tf.cast(y, tf.complex64)
        h_hat = tf.cast(h_hat, tf.complex64)
        err_var = tf.cast(err_var, tf.float32)
        no = tf.cast(no, tf.float32)

        # Dynamic shapes
        B = tf.shape(y)[0]
        num_bs = tf.shape(y)[1]
        num_bs_ant = tf.shape(y)[2]
        num_ofdm_symbols = tf.shape(y)[3]        # H
        num_data_subcarriers = tf.shape(y)[4]    # W

        # ================================
        # 1) y features (per BS, per ant)
        # ================================
        # y_real, y_imag: [B, num_bs, num_bs_ant, H, W]
        y_real = tf.math.real(y)
        y_imag = tf.math.imag(y)
        # [B, num_bs, num_bs_ant, H, W, 2]
        y_stack = tf.stack([y_real, y_imag], axis=-1)
        # -> [B, H, W, num_bs, num_bs_ant, 2]
        y_stack = tf.transpose(y_stack, [0, 3, 4, 1, 2, 5])
        # -> [B, H, W, num_bs * num_bs_ant * 2]
        y_feats = tf.reshape(
            y_stack,
            [
                B,
                num_ofdm_symbols,
                num_data_subcarriers,
                num_bs * num_bs_ant * 2,
            ],
        )

        # ==========================================
        # 2) h_hat features (KEEP BS/ant, UE, stream)
        # ==========================================
        # h_hat: [B, num_bs, num_bs_ant, num_ue, num_streams_per_ue, H, W]
        h_real = tf.math.real(h_hat)
        h_imag = tf.math.imag(h_hat)
        # [B, num_bs, num_bs_ant, num_ue, num_streams_per_ue, H, W, 2]
        h_stack = tf.stack([h_real, h_imag], axis=-1)
        # Move spatial dims H,W in front:
        # -> [B, H, W, num_bs, num_bs_ant, num_ue, num_streams_per_ue, 2]
        h_stack = tf.transpose(h_stack, [0, 5, 6, 1, 2, 3, 4, 7])
        # Flatten all non-spatial dims into channels
        # C_h_hat = num_bs * num_bs_ant * num_ue * num_streams_per_ue * 2
        h_feats = tf.reshape(
            h_stack,
            [
                B,
                num_ofdm_symbols,
                num_data_subcarriers,
                -1,
            ],
        )

        # ===========================
        # 3) err_var features (per UE)
        # ===========================
        # err_var: [1, 1, 1, num_ue, num_streams_per_ue, H, W]
        # Tile over batch dimension:
        err_var_t = tf.tile(
            err_var,
            [B, 1, 1, 1, 1, 1, 1],
        )  # [B, 1, 1, num_ue, num_streams_per_ue, H, W]
        # Remove dummy dims:
        err_var_t = tf.squeeze(err_var_t, axis=[1, 2])  # [B, num_ue, num_streams_per_ue, H, W]
        # -> [B, H, W, num_ue, num_streams_per_ue]
        err_feats = tf.transpose(err_var_t, [0, 3, 4, 1, 2])
        # -> [B, H, W, num_ue * num_streams_per_ue]
        err_feats = tf.reshape(
            err_feats,
            [
                B,
                num_ofdm_symbols,
                num_data_subcarriers,
                self._num_ue * self._num_streams_per_ue,
            ],
        )

        # ==================================
        # 4) noise variance as scalar feature
        # ==================================
        # no: scalar -> [B, H, W, 1]
        no_expanded = tf.reshape(no, [1, 1, 1, 1])
        no_expanded = tf.tile(
            no_expanded,
            [B, num_ofdm_symbols, num_data_subcarriers, 1],
        )

        # ==================================
        # Build feature tensor z: [B, H, W, C_in]
        # ==================================
        # Channels:
        #   C_y      = 2 * num_bs * num_bs_ant
        #   C_h_hat  = 2 * num_bs * num_bs_ant * num_ue * num_streams_per_ue
        #   C_err    =     num_ue * num_streams_per_ue
        #   C_noise  = 1
        z = tf.concat([y_feats, h_feats, err_feats, no_expanded], axis=-1)
        z = tf.cast(z, tf.float32)

        # ==================================
        # ConvLSTM2D stack with residuals
        # ==================================
        # ConvLSTM2D expects [B, T, rows, cols, C]
        #
        # We use:
        #   T    = H (OFDM symbols)
        #   rows = W (subcarriers)
        #   cols = 1 (dummy)
        #   ch   = C_in
        #
        # z: [B, H, W, C_in] -> [B, H, W, 1, C_in]
        z_seq = tf.expand_dims(z, axis=3)

        # Input ConvLSTM2D: [B, H, W, 1, num_conv2d_filters]
        z_seq = self._convlstm_in(z_seq, training=training)

        # Residual ConvLSTM blocks (same shape in/out)
        for block in self._res_blocks:
            z_seq = block(z_seq, training=training)

        # Output ConvLSTM2D:
        # [B, H, W, 1, num_streams_total * num_bits_per_symbol]
        z_seq = self._convlstm_out(z_seq, training=training)

        # Remove dummy spatial dim -> [B, H, W, F]
        z = tf.squeeze(z_seq, axis=3)

        # ==================================
        # Reshape to LLRs
        # ==================================
        num_data_symbols = num_ofdm_symbols * num_data_subcarriers

        F = tf.shape(z)[-1]
        tf.debugging.assert_equal(
            F,
            self._num_streams_total * self._num_bits_per_symbol,
            message="ConvLSTM2D output channels must be num_streams_total * num_bits_per_symbol",
        )

        # z: [B, H, W, num_streams_total * num_bits_per_symbol]
        # -> [B, num_data_symbols, num_streams_total, num_bits_per_symbol]
        z = tf.reshape(
            z,
            [
                B,
                num_data_symbols,
                self._num_streams_total,
                self._num_bits_per_symbol,
            ],
        )

        # -> [B, num_data_symbols, num_ue, num_streams_per_ue, num_bits_per_symbol]
        z = tf.reshape(
            z,
            [
                B,
                num_data_symbols,
                self._num_ue,
                self._num_streams_per_ue,
                self._num_bits_per_symbol,
            ],
        )

        # Reorder to [B, num_ue, num_streams_per_ue, num_data_symbols, bits]
        z = tf.transpose(z, [0, 2, 3, 1, 4])

        # Flatten over (num_data_symbols * bits):
        llr = tf.reshape(
            z,
            [
                B,
                self._num_ue,
                self._num_streams_per_ue,
                num_data_symbols * self._num_bits_per_symbol,
            ],
        )

        return llr
