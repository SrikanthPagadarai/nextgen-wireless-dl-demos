"""Batched Indirect Learning Architecture DPD for Sionna pipeline."""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class IndirectLearningDPD(tf.keras.layers.Layer):
    """
    Indirect Learning Architecture Digital Pre-Distortion for batched signals.

    Inherits from tf.keras.layers.Layer for Sionna compatibility and differentiability.
    Supports batched input [B, num_samples] for Sionna-style processing.
    Uses the same GMP (Generalized Memory Polynomial) model as the original.

    The predistort() method is fully differentiable and can be used in end-to-end
    training pipelines. For LS-based learning, use perform_learning().

    Args:
        order: Polynomial order (must be odd, default: 7)
        memory_depth: Memory depth in samples (default: 3)
        lag_depth: Lag/lead depth for cross-terms (default: 0)
        nIterations: Number of learning iterations (default: 3)
        learning_rate: Learning rate (default: 0.75)
        learning_method: 'newton' or 'ema' (default: 'newton')
        use_even: Include even-order terms (default: False)
        use_conj: Include conjugate terms (default: False)
        use_dc_term: Include DC term (default: False)
    """

    DEFAULT_PARAMS = {
        "order": 7,
        "memory_depth": 4,
        "lag_depth": 0,
        "nIterations": 3,
        "use_conj": False,
        "use_dc_term": False,
        "learning_rate": 0.75,
        "use_even": False,
        "learning_method": "newton",
    }

    def __init__(self, params=None, **kwargs):
        super().__init__(**kwargs)
        p = {**self.DEFAULT_PARAMS, **(params or {})}

        if p["order"] % 2 == 0:
            raise ValueError("Order of the DPD must be odd.")

        self._order, self._memory_depth, self._lag_depth = (
            p["order"],
            p["memory_depth"],
            p["lag_depth"],
        )
        self._nIterations, self._learning_rate = p["nIterations"], p["learning_rate"]
        self._learning_method = p["learning_method"]
        self._use_even, self._use_conj, self._use_dc_term = (
            p["use_even"],
            p["use_conj"],
            p["use_dc_term"],
        )

        if self._use_even:
            assert (
                self._lag_depth == 0
            ), "GMP not yet supported for even terms. Set lag_depth=0"

        self._n_coeffs = self._compute_n_coeffs()
        self._coeffs, self.coeff_history, self.result_history = None, None, None

    def build(self, input_shape):
        """Build layer - create trainable weights."""
        init_real = np.zeros((self._n_coeffs, 1), dtype=np.float32)
        init_real[0, 0] = 1.0
        init_imag = np.zeros((self._n_coeffs, 1), dtype=np.float32)

        self._coeffs_real = self.add_weight(
            name="dpd_coeffs_real",
            shape=(self._n_coeffs, 1),
            initializer=tf.keras.initializers.Constant(init_real),
            trainable=True,
            dtype=tf.float32,
        )
        self._coeffs_imag = self.add_weight(
            name="dpd_coeffs_imag",
            shape=(self._n_coeffs, 1),
            initializer=tf.keras.initializers.Constant(init_imag),
            trainable=True,
            dtype=tf.float32,
        )
        super().build(input_shape)

    @property
    def order(self):
        return self._order

    @property
    def memory_depth(self):
        return self._memory_depth

    @property
    def n_coeffs(self):
        return self._n_coeffs

    @property
    def coeffs(self):
        """Return complex coefficients from real/imag parts."""
        if not self.built:
            raise RuntimeError(
                "Layer not built. Call the layer on input first, or call build()."
            )
        return tf.complex(self._coeffs_real, self._coeffs_imag)

    @coeffs.setter
    def coeffs(self, value):
        """Set coefficients from complex tensor."""
        if not self.built:
            raise RuntimeError(
                "Layer not built. Call the layer on input first, or call build()."
            )
        self._coeffs_real.assign(tf.math.real(value))
        self._coeffs_imag.assign(tf.math.imag(value))

    def _compute_n_coeffs(self):
        """Compute total number of DPD coefficients."""
        n_order = self._order if self._use_even else (self._order + 1) // 2
        n = n_order * self._memory_depth
        if not self._use_even:
            n += 2 * (n_order - 1) * self._memory_depth * self._lag_depth
        if self._use_conj:
            n *= 2
        if self._use_dc_term:
            n += 1
        return n

    def _delay_signal(self, signal, delay):
        """Apply delay to signal by prepending zeros."""
        if delay == 0:
            return signal
        padding = tf.zeros(delay, dtype=signal.dtype)
        return tf.concat([padding, signal[:-delay]], axis=0)

    def _add_memory_columns(self, columns, branch):
        """Add delayed versions of branch for all memory depths."""
        for delay in range(self._memory_depth):
            columns.append(self._delay_signal(branch, delay))

    def setup_basis_matrix(self, x):
        """
        Build GMP basis matrix for 1D input. Fully differentiable.

        Args:
            x: [num_samples] complex tensor
        Returns:
            [num_samples, n_coeffs] complex tensor
        """
        x = tf.cast(tf.reshape(x, [-1]), tf.complex64)
        n_samples = tf.shape(x)[0]
        abs_x = tf.abs(x)
        step = 1 if self._use_even else 2
        columns = []

        # Main memory polynomial branch
        for order in range(1, self._order + 1, step):
            branch = x * tf.cast(tf.pow(abs_x, order - 1), tf.complex64)
            self._add_memory_columns(columns, branch)

        # Lagging cross-terms
        for order in range(3, self._order + 1, step):
            abs_base = tf.pow(abs_x, order - 1)
            for lag in range(1, self._lag_depth + 1):
                lagged_abs = tf.concat(
                    [tf.zeros(lag, dtype=tf.float32), abs_base[:-lag]], axis=0
                )
                branch = x * tf.cast(lagged_abs, tf.complex64)
                self._add_memory_columns(columns, branch)

        # Leading cross-terms
        for order in range(3, self._order + 1, step):
            abs_base = tf.pow(abs_x, order - 1)
            for lead in range(1, self._lag_depth + 1):
                lead_abs = tf.concat(
                    [abs_base[lead:], tf.zeros(lead, dtype=tf.float32)], axis=0
                )
                branch = x * tf.cast(lead_abs, tf.complex64)
                self._add_memory_columns(columns, branch)

        # Conjugate branch
        if self._use_conj:
            for order in range(1, self._order + 1, step):
                branch = tf.math.conj(x) * tf.cast(
                    tf.pow(abs_x, order - 1), tf.complex64
                )
                self._add_memory_columns(columns, branch)

        # DC term
        if self._use_dc_term:
            columns.append(tf.ones(n_samples, dtype=tf.complex64))

        return tf.stack(columns, axis=1)

    def predistort(self, x):
        """
        Apply predistortion to input signal. Fully differentiable.

        Args:
            x: [num_samples] or [B, num_samples] tensor
        Returns:
            Same shape as input - predistorted signal
        """
        if not self.built:
            self.build(x.shape)

        input_shape, input_ndims = tf.shape(x), len(x.shape)
        coeffs = self.coeffs

        if input_ndims == 1:
            X = self.setup_basis_matrix(x)
            return tf.reshape(tf.linalg.matmul(X, coeffs), [-1])
        elif input_ndims == 2:
            batch_size, samples_per_batch = input_shape[0], input_shape[1]
            X = self.setup_basis_matrix(tf.reshape(x, [-1]))
            y_flat = tf.reshape(tf.linalg.matmul(X, coeffs), [-1])
            return tf.reshape(y_flat, [batch_size, samples_per_batch])
        else:
            raise ValueError(f"Input must be 1D or 2D, got shape {x.shape}")

    def call(self, x, training=None):
        """Keras layer call - applies predistortion."""
        return self.predistort(x)

    def _ls_estimation(self, X, y):
        """Regularized least-squares estimation."""
        start = self._memory_depth + self._lag_depth - 1
        end = -self._lag_depth if self._lag_depth > 0 else None
        X_slice, y_slice = X[start:end], tf.reshape(y[start:end], [-1, 1])

        lam = tf.constant(0.001, dtype=tf.float32)
        XH = tf.linalg.adjoint(X_slice)
        XHX = tf.linalg.matmul(XH, X_slice)
        reg = tf.cast(lam * tf.eye(tf.shape(XHX)[0]), dtype=tf.complex64)
        return tf.linalg.solve(XHX + reg, tf.linalg.matmul(XH, y_slice))

    def perform_learning(self, x, pa, verbose=False):
        """
        Perform iterative DPD learning using indirect learning architecture.

        Args:
            x: [num_samples] or [B, num_samples] input tensor (at PA sample rate)
            pa: PowerAmplifier instance
            verbose: Print progress (default: True)
        Returns:
            Dictionary with learning results
        """
        if not self.built:
            self.build(x.shape)

        input_ndims, input_shape = len(x.shape), tf.shape(x)
        batch_size = input_shape[0] if input_ndims == 2 else 1
        samples_per_batch = input_shape[1] if input_ndims == 2 else None
        x_flat = tf.reshape(x, [-1]) if input_ndims == 2 else x

        self.coeff_history = self.coeffs.numpy().copy()
        self.result_history = []

        if verbose:
            print(
                f"Starting DPD learning: {self._nIterations} iterations, "
                f"order={self._order}, memory={self._memory_depth}"
            )

        for iteration in range(self._nIterations):
            u = self.predistort(x_flat)

            # Pass through PA (handle batched PA)
            if input_ndims == 2:
                y = tf.reshape(pa(tf.reshape(u, [batch_size, samples_per_batch])), [-1])
            else:
                y = tf.reshape(pa(tf.expand_dims(u, 0)), [-1])

            Y = self.setup_basis_matrix(y)
            current_coeffs = self.coeffs

            if self._learning_method == "newton":
                new_coeffs = current_coeffs + self._learning_rate * self._ls_estimation(
                    Y, u - self.predistort(y)
                )
            else:  # 'ema'
                new_coeffs = (
                    1 - self._learning_rate
                ) * current_coeffs + self._learning_rate * self._ls_estimation(Y, u)

            self.coeffs = new_coeffs
            self.coeff_history = np.hstack([self.coeff_history, self.coeffs.numpy()])

            if verbose:
                y_power = 10 * np.log10(np.mean(np.abs(y.numpy()) ** 2) + 1e-12)
                print(
                    f"  Iteration {iteration + 1}/{self._nIterations}: "
                    f"PA output power = {y_power:.2f} dB"
                )

        if verbose:
            print("DPD learning complete.")

        return {"coeffs": self.coeffs.numpy(), "coeff_history": self.coeff_history}

    def plot_coeff_history(self, save_path=None):
        """Plot coefficient learning history."""
        if self.coeff_history is None:
            print("No learning history available. Run perform_learning() first.")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(self.coeff_history.shape[1]), np.abs(self.coeff_history.T))
        plt.title("DPD Coefficient Learning History")
        plt.xlabel("Iteration")
        plt.ylabel("|coeffs|")
        plt.grid(True)

        if save_path:
            plt.savefig(save_path, dpi=150)
            plt.close()
            print(f"Saved to {save_path}")
        else:
            plt.show()

    def get_config(self):
        """Return layer configuration for serialization."""
        config = super().get_config()
        config.update(
            {
                "order": self._order,
                "memory_depth": self._memory_depth,
                "lag_depth": self._lag_depth,
                "nIterations": self._nIterations,
                "learning_rate": self._learning_rate,
                "learning_method": self._learning_method,
                "use_even": self._use_even,
                "use_conj": self._use_conj,
                "use_dc_term": self._use_dc_term,
            }
        )
        return config
