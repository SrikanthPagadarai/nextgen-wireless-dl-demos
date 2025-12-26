"""Batched signal processing utilities."""

import tensorflow as tf


def normalize_to_rms(data, target_rms):
    """
    Normalize batched signal to target RMS power.

    Computes global statistics across all batches (as if concatenated).

    Args:
        data: [B, num_samples] complex tensor
        target_rms: target RMS in dBm

    Returns:
        normalized data [B, num_samples], scale_factor
    """
    # Compute norm using abs to avoid complex->float cast warning
    abs_data = tf.abs(data)  # float32
    sum_sq = tf.reduce_sum(abs_data * abs_data)
    norm = tf.sqrt(sum_sq)

    n = tf.cast(tf.size(data), tf.float32)

    target_power = tf.constant(10 ** ((target_rms - 30) / 10), dtype=tf.float32)
    scale_factor = tf.sqrt(50.0 * n * target_power) / norm

    normalized = data * tf.cast(scale_factor, data.dtype)

    return normalized, scale_factor
