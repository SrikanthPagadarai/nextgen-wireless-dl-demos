import numpy as np
import tensorflow as tf

from sionna.phy.channel import OFDMChannel
from sionna.phy.nr import PUSCHConfig, PUSCHTransmitter, PUSCHReceiver
from sionna.phy.utils import ebnodb2no
from sionna.phy.ofdm import LinearDetector
from sionna.phy.mimo import StreamManagement

from .config import Config
from .pusch_trainable_transmitter import PUSCHTrainableTransmitter
from .pusch_neural_detector import PUSCHNeuralDetector
from .pusch_trainable_receiver import PUSCHTrainableReceiver

class PUSCHLinkE2E(tf.keras.Model):
    def __init__(self, channel_model, perfect_csi, use_autoencoder=False, training=False):
        super().__init__()

        self._training = training

        self._channel_model = channel_model
        self._perfect_csi = perfect_csi
        self._use_autoencoder = use_autoencoder

        # Central config (all hard-coded system parameters live in Config)
        self._cfg = Config()

        # System configuration from Config
        self._num_prb = self._cfg.num_prb
        self._mcs_index = self._cfg.mcs_index
        self._num_layers = self._cfg.num_layers
        self._mcs_table = self._cfg.mcs_table
        self._domain = self._cfg.domain

        # from Config
        self._num_ue_ant = self._cfg.num_ue_ant
        self._num_ue = self._cfg.num_ue
        self._subcarrier_spacing = self._cfg.subcarrier_spacing  # must be the same as used for Path2CIR

        # PUSCHConfig for the first transmitter
        pusch_config = PUSCHConfig()
        pusch_config.carrier.subcarrier_spacing = self._subcarrier_spacing / 1000.0
        pusch_config.carrier.n_size_grid = self._num_prb
        pusch_config.num_antenna_ports = self._num_ue_ant
        pusch_config.num_layers = self._num_layers
        pusch_config.precoding = "codebook"
        pusch_config.tpmi = 1
        pusch_config.dmrs.dmrs_port_set = list(range(self._num_layers))
        pusch_config.dmrs.config_type = 1
        pusch_config.dmrs.length = 1
        pusch_config.dmrs.additional_position = 1
        pusch_config.dmrs.num_cdm_groups_without_data = 2
        pusch_config.tb.mcs_index = self._mcs_index
        pusch_config.tb.mcs_table = self._mcs_table

        # set Config's properties so that Neural-Detector can use them
        self._cfg.pusch_pilot_indices = pusch_config.dmrs_symbol_indices
        self._cfg.pusch_num_subcarriers = pusch_config.num_subcarriers
        self._cfg.pusch_num_symbols_per_slot = pusch_config.carrier.num_symbols_per_slot

        # Create PUSCHConfigs for the other transmitters
        pusch_configs = [pusch_config]
        for i in range(1, self._num_ue):
            pc = pusch_config.clone()
            pc.dmrs.dmrs_port_set = list(range(i * self._num_layers, (i + 1) * self._num_layers))
            pusch_configs.append(pc)

        # Create PUSCHTransmitter
        self._pusch_transmitter = (PUSCHTrainableTransmitter(pusch_configs, output_domain=self._domain, training=self._training)
                                   if self._use_autoencoder 
                                   else PUSCHTransmitter(pusch_configs, output_domain=self._domain))
        self._cfg.resource_grid = self._pusch_transmitter.resource_grid

        # Create PUSCHReceiver
        rx_tx_association = np.ones([1, self._num_ue], bool)
        stream_management = StreamManagement(rx_tx_association,self._num_layers)

        if self._use_autoencoder:
            detector = PUSCHNeuralDetector(self._cfg)
        else:
            detector = LinearDetector(
                equalizer="lmmse",
                output="bit",
                demapping_method="maxlog",
                resource_grid=self._pusch_transmitter.resource_grid,
                stream_management=stream_management,
                constellation_type="qam",
                num_bits_per_symbol=pusch_config.tb.num_bits_per_symbol,
            )        

        if self._use_autoencoder:
            if self._perfect_csi:
                self._pusch_receiver = PUSCHTrainableReceiver(
                    self._pusch_transmitter,
                    mimo_detector=detector,
                    input_domain=self._domain,
                    channel_estimator="perfect",
                )
            else:
                self._pusch_receiver = PUSCHTrainableReceiver(
                    self._pusch_transmitter,
                    mimo_detector=detector,
                    input_domain=self._domain,
                )
        else:
            if self._perfect_csi:
                self._pusch_receiver = PUSCHReceiver(
                    self._pusch_transmitter,
                    mimo_detector=detector,
                    input_domain=self._domain,
                    channel_estimator="perfect",
                )
            else:
                self._pusch_receiver = PUSCHReceiver(
                    self._pusch_transmitter,
                    mimo_detector=detector,
                    input_domain=self._domain,
                )

        # Configure the actual channel
        self._channel = OFDMChannel(
            self._channel_model,
            self._pusch_transmitter.resource_grid,
            normalize_channel=True,
            return_channel=True,
        )

        if self._training:
            self._bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    @property
    def trainable_variables(self):
        vars_ = []
        if hasattr(self, "_pusch_transmitter"):
            vars_ += list(self._pusch_transmitter.trainable_variables)
        if hasattr(self, "_pusch_receiver"):
            vars_ += list(self._pusch_receiver.trainable_variables)
        return vars_
    
    '''
    @tf.function(jit_compile=False)
    def call(self, batch_size, ebno_db):

        if self._use_autoencoder:
            x, b, c = self._pusch_transmitter(batch_size)
        else:
            x, b = self._pusch_transmitter(batch_size)

        no = ebnodb2no(
            ebno_db,
            self._pusch_transmitter._num_bits_per_symbol,
            self._pusch_transmitter._target_coderate,
            self._pusch_transmitter.resource_grid,
        )
        y, h = self._channel(x, no)

        if self._use_autoencoder and self._training:
            if self._perfect_csi:
                llr = self._pusch_receiver(y, no, h)
            else:
                llr = self._pusch_receiver(y, no)            
            loss = self._bce(c, llr)
            return loss
        else:
            if self._perfect_csi:
                b_hat = self._pusch_receiver(y, no, h)
            else:
                b_hat = self._pusch_receiver(y, no)
            return b, b_hat
    '''
    
    '''
    @tf.function(jit_compile=False)
    def call(self, batch_size, ebno_db):

        if self._use_autoencoder:
            x, b, c = self._pusch_transmitter(batch_size)
        else:
            x, b = self._pusch_transmitter(batch_size)

        no = ebnodb2no(
            ebno_db,
            self._pusch_transmitter._num_bits_per_symbol,
            self._pusch_transmitter._target_coderate,
            self._pusch_transmitter.resource_grid,
        )

        y, h = self._channel(x, no)
        y = tf.stop_gradient(y)
        if h is not None:
            h = tf.stop_gradient(h)

        if self._use_autoencoder and self._training:
            if self._perfect_csi:
                llr = self._pusch_receiver(y, no, h)
            else:
                llr = self._pusch_receiver(y, no)
            loss = self._bce(c, llr)
            return loss
        else:
            if self._perfect_csi:
                b_hat = self._pusch_receiver(y, no, h)
            else:
                b_hat = self._pusch_receiver(y, no)
            return b, b_hat
    '''    

    @tf.function(jit_compile=False)
    def call(self, batch_size, ebno_db):

        if self._use_autoencoder:
            x, b, c = self._pusch_transmitter(batch_size)
        else:
            x, b = self._pusch_transmitter(batch_size)

        no = ebnodb2no(
            ebno_db,
            self._pusch_transmitter._num_bits_per_symbol,
            self._pusch_transmitter._target_coderate,
            self._pusch_transmitter.resource_grid,
        )

        @tf.custom_gradient
        def channel_wrapper(x_in, no_in):
            # Forward: real channel
            y_out, h_out = self._channel(x_in, no_in)

            def grad(dy_y, dy_h):
                """
                dy_y: gradient of loss w.r.t. y_out, shape [B, num_bs, num_bs_ant, H, W]
                dy_h: gradient of loss w.r.t. h_out (ignored here)

                We build a surrogate gradient for x_in with the correct shape
                [B, ..., H_x, W_x] by aggregating dy_y and broadcasting.
                """
                x_shape = tf.shape(x_in)  # [B, ..., H_x, W_x]
                B = x_shape[0]
                H_x = x_shape[-2]
                W_x = x_shape[-1]

                # Reduce along the BS/antenna dims of dy_y -> [B, H_y, W_y]
                dy_y_mean = tf.reduce_mean(dy_y, axis=[1, 2])  # [B, H_y, W_y]

                # Reshape to [B, 1, 1, 1, 1, H_x, W_x] and broadcast to x_shape
                dy_y_reshaped = tf.reshape(
                    dy_y_mean,
                    [B, 1, 1, 1, 1, H_x, W_x],
                )
                dx = tf.broadcast_to(dy_y_reshaped, x_shape)

                dno = None  # no gradient w.r.t. noise variance
                return dx, dno

            return (y_out, h_out), grad

        # Use the wrapped channel
        y, h = channel_wrapper(x, no)

        if self._use_autoencoder and self._training:
            if self._perfect_csi:
                llr = self._pusch_receiver(y, no, h)
            else:
                llr = self._pusch_receiver(y, no)
            loss = self._bce(c, llr)
            return loss
        else:
            if self._perfect_csi:
                b_hat = self._pusch_receiver(y, no, h)
            else:
                b_hat = self._pusch_receiver(y, no)
            return b, b_hat

