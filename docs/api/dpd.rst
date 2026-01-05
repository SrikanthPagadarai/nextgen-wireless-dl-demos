dpd
===

This module implements Digital Pre-Distortion (DPD) techniques for power amplifier linearization.

Configuration
-------------

.. autoclass:: demos.dpd.src.config.Config
   :members:
   :undoc-members:
   :show-inheritance:

Transmitter
-----------

.. autoclass:: demos.dpd.src.tx.Tx
   :members:
   :undoc-members:
   :show-inheritance:

Receiver
--------

.. autoclass:: demos.dpd.src.rx.Rx
   :members:
   :undoc-members:
   :show-inheritance:

Power Amplifier
---------------

.. autoclass:: demos.dpd.src.power_amplifier.PowerAmplifier
   :members:
   :undoc-members:
   :show-inheritance:

Interpolator
------------

.. autoclass:: demos.dpd.src.interpolator.Interpolator
   :members:
   :undoc-members:
   :show-inheritance:

Least-Squares DPD
-----------------

.. autoclass:: demos.dpd.src.ls_dpd.LeastSquaresDPD
   :members:
   :undoc-members:
   :show-inheritance:

Neural Network DPD
------------------

.. autoclass:: demos.dpd.src.nn_dpd.ResidualBlock
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: demos.dpd.src.nn_dpd.NeuralNetworkDPD
   :members:
   :undoc-members:
   :show-inheritance:

System
------

.. autoclass:: demos.dpd.src.system.DPDSystem
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: demos.dpd.src.ls_dpd_system.LS_DPDSystem
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: demos.dpd.src.nn_dpd_system.NN_DPDSystem
   :members:
   :undoc-members:
   :show-inheritance:
