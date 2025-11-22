from sionna.phy.nr import PUSCHTransmitter
from sionna.phy.mapping import Mapper, Constellation
import tensorflow as tf

class PUSCHTrainableTransmitter(PUSCHTransmitter):
    def __init__(self, *args, training=False, **kwargs):
        self._training = training
        
        # parent constructor
        super().__init__(*args, **kwargs)
        
        if self._training:
            self._setup_training()
    
    @property
    def trainable_variables(self):
        if self._training:
            return [self._points_r, self._points_i]
        return []

    def _setup_training(self):
        """Setup trainable constellation"""
        # Original QAM constellation used as initialization
        qam_points = Constellation("qam", num_bits_per_symbol=self._num_bits_per_symbol).points

        # Trainable real/imag parts as tf.Variables
        init_r = tf.math.real(qam_points)
        init_i = tf.math.imag(qam_points)

        self._points_r = tf.Variable(
            tf.cast(init_r, self.rdtype),
            trainable=True,
            name="constellation_real"
        )
        self._points_i = tf.Variable(
            tf.cast(init_i, self.rdtype),
            trainable=True,
            name="constellation_imag"
        )

        # custom QAM constellation
        self._constellation = Constellation(
            "custom",
            num_bits_per_symbol=self._num_bits_per_symbol,
            points=tf.complex(self._points_r, self._points_i),
            normalize=True,
            center=True
        )

        # Replace the mapper to use our trainable constellation
        self._mapper = Mapper(constellation=self._constellation)
    
    def call(self, inputs):
        """
        Parameters
        ----------
        inputs : int or [batch_size, num_layers, tb_size], tf.float32
            Either batch_size (if return_bits=True) or bits tensor (if return_bits=False)
        """
        if self._training:
            # Update constellation from trainable weights
            self._constellation.points = tf.complex(self._points_r,
                                                    self._points_i)
        
        ### copy of PUSCHTransmitter.call from here with the additional return of c
        if self._return_bits:
            # inputs defines batch_size
            batch_size = inputs
            b = self._binary_source([batch_size, self._num_tx, self._tb_size])
        else:
            b = inputs

        # Encode transport block
        c = self._tb_encoder(b)

        # Map to constellations
        x_map = self._mapper(c)

        # Map to layers
        x_layer = self._layer_mapper(x_map)

        # Apply resource grid mapping
        x_grid = self._resource_grid_mapper(x_layer)

        # (Optionally) apply PUSCH precoding
        if self._precoding=="codebook":
            x_pre = self._precoder(x_grid)
        else:
            x_pre = x_grid

        # (Optionally) apply OFDM modulation
        if self._output_domain=="time":
            x = self._ofdm_modulator(x_pre)
        else:
            x = x_pre

        if self._return_bits:
            return x, b, c
        else:
            return x