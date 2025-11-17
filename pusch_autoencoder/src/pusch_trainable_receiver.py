import tensorflow as tf
from sionna.phy.channel import time_to_ofdm_channel
from sionna.phy.nr import PUSCHReceiver

class PUSCHTrainableReceiver(PUSCHReceiver):
    """
    Minor variant of PUSCHReceiver that returns LLRs after the LayerDemapper
    and *before* TB decoding. This lets us define an autoencoder-style
    loss on the bit LLRs.
    """   
    def call(self, y, no, h=None):
        ### copy of PUSCHReceiver.call up to the TBDecoder
        # (Optional) OFDM Demodulation
        if self._input_domain=="time":
            y = self._ofdm_demodulator(y)

        # Channel estimation
        if self._perfect_csi:

            # Transform time-domain to frequency-domain channel
            if self._input_domain=="time":
                h = time_to_ofdm_channel(h, self.resource_grid, self._l_min)


            if self._w is not None:
                # Reshape h to put channel matrix dimensions last
                # [batch size, num_rx, num_tx, num_ofdm_symbols,...
                #  ...fft_size, num_rx_ant, num_tx_ant]
                h = tf.transpose(h, perm=[0,1,3,5,6,2,4])

                # Multiply by precoding matrices to compute effective channels
                # [batch size, num_rx, num_tx, num_ofdm_symbols,...
                #  ...fft_size, num_rx_ant, num_streams]
                h = tf.matmul(h, self._w)

                # Reshape
                # [batch size, num_rx, num_rx_ant, num_tx, num_streams,...
                #  ...num_ofdm_symbols, fft_size]
                h = tf.transpose(h, perm=[0,1,5,2,6,3,4])
            h_hat = h
            err_var = tf.cast(0, dtype=h_hat.dtype.real_dtype)
        else:
            h_hat,err_var = self._channel_estimator(y, no)

        # MIMO Detection
        llr = self._mimo_detector(y, h_hat, err_var, no)

        # Layer demapping
        llr = self._layer_demapper(llr)

        # return the LLRs
        return llr
