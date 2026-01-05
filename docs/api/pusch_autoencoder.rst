pusch_autoencoder
=================

This module implements an end-to-end autoencoder for the 5G NR Physical Uplink Shared Channel (PUSCH).

Configuration
-------------

.. autoclass:: demos.pusch_autoencoder.src.config.Config
   :members:
   :undoc-members:
   :show-inheritance:

Channel Impulse Response
------------------------

.. autoclass:: demos.pusch_autoencoder.src.cir_generator.CIRGenerator
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: demos.pusch_autoencoder.src.cir_manager.CIRManager
   :members:
   :undoc-members:
   :show-inheritance:

Trainable Transmitter
---------------------

.. autoclass:: demos.pusch_autoencoder.src.pusch_trainable_transmitter.PUSCHTrainableTransmitter
   :members:
   :undoc-members:
   :show-inheritance:

Neural Detector
---------------

.. autoclass:: demos.pusch_autoencoder.src.pusch_neural_detector.Conv2DResBlock
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: demos.pusch_autoencoder.src.pusch_neural_detector.PUSCHNeuralDetector
   :members:
   :undoc-members:
   :show-inheritance:

Trainable Receiver
------------------

.. autoclass:: demos.pusch_autoencoder.src.pusch_trainable_receiver.PUSCHTrainableReceiver
   :members:
   :undoc-members:
   :show-inheritance:

System
------

.. autoclass:: demos.pusch_autoencoder.src.system.PUSCHLinkE2E
   :members:
   :undoc-members:
   :show-inheritance:
