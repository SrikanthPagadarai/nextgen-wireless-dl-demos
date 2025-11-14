import os
import numpy as np
import tensorflow as tf
import sionna
from sionna.phy.channel import CIRDataset
from sionna.rt import (
    load_scene, Camera, Transmitter, Receiver,
    PlanarArray, PathSolver, RadioMapSolver
)

from config import Config
from cir_generator import CIRGenerator

# GPU / TF configuration (run once on import)
if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    gpu_num = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

tf.get_logger().setLevel('ERROR')


# Global configuration
_cfg = Config()

# system parameters directly from config.py
subcarrier_spacing = _cfg.subcarrier_spacing
num_time_steps = _cfg.num_time_steps
num_ue = _cfg.num_ue
num_bs = _cfg.num_bs
num_ue_ant = _cfg.num_ue_ant
num_bs_ant = _cfg.num_bs_ant
batch_size_cir = _cfg.batch_size_cir

# solver parameters
max_depth = _cfg.max_depth
min_gain_db = _cfg.min_gain_db
max_gain_db = _cfg.max_gain_db
min_dist = _cfg.min_dist_m
max_dist = _cfg.max_dist_m

# radio map parameters
rm_cell_size = _cfg.rm_cell_size
rm_samples_per_tx = _cfg.rm_samples_per_tx
rm_vmin_db = _cfg.rm_vmin_db
rm_clip_at = _cfg.rm_clip_at
rm_resolution = _cfg.rm_resolution
rm_num_samples = _cfg.rm_num_samples

target_num_cirs = _cfg.target_num_cirs
batch_size = _cfg.batch_size   # for CIRDataset construction


def build_channel_model():
    """Build and return a CIRDataset-based channel model.

    This function encapsulates the previous top-level logic in cir.py:
    - load scene & configure arrays
    - build radio map & sample UE positions
    - trace paths and build CIRs (a, tau)
    - wrap them using CIRGenerator and CIRDataset

    Returns
    -------
    channel_model : sionna.phy.channel.CIRDataset
        Channel model ready to be passed into the System/Model.
    """
    
    # Load scene    
    scene = load_scene(sionna.rt.scene.munich)

    # base station array
    scene.tx_array = PlanarArray(
        num_rows=1,
        num_cols=num_bs_ant // 2,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="tr38901",
        polarization="cross"
    )

    # base station (transmitter)
    tx = Transmitter(
        name="tx",
        position=[8.5, 21, 27],
        look_at=[45, 90, 1.5],
        display_radius=3.0
    )
    scene.add(tx)

    camera = Camera(position=[0, 80, 500], orientation=np.array([0, np.pi/2, -np.pi/2]))
    
    # Compute radio map    
    rm_solver = RadioMapSolver()
    rm = rm_solver(
        scene,
        max_depth=max_depth,
        cell_size=rm_cell_size,
        samples_per_tx=rm_samples_per_tx
    )

    scene.render_to_file(
        camera=camera,
        radio_map=rm,
        rm_vmin=rm_vmin_db,
        clip_at=rm_clip_at,
        resolution=list(rm_resolution),
        filename="munich_radio_map.png",
        num_samples=rm_num_samples
    )
    
    # Sample initial UE positions    
    ue_pos, _ = rm.sample_positions(
        num_pos=batch_size_cir,
        metric="path_gain",
        min_val_db=min_gain_db,
        max_val_db=max_gain_db,
        min_dist=min_dist,
        max_dist=max_dist
    )

    # UE arrays
    scene.rx_array = PlanarArray(
        num_rows=1,
        num_cols=num_ue_ant // 2,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="cross"
    )

    # create receivers
    for i in range(batch_size_cir):
        p = ue_pos[0, i, :]
        if hasattr(p, "numpy"):
            p = p.numpy()
        p = np.asarray(p, dtype=np.float64)

        try:
            scene.remove(f"rx-{i}")
        except Exception:
            pass

        rx = Receiver(
            name=f"rx-{i}",
            position=(float(p[0]), float(p[1]), float(p[2])),
            velocity=(3.0, 3.0, 0.0),
            display_radius=1.0,
            color=(1, 0, 0)
        )
        scene.add(rx)

    scene.render_to_file(
        camera=camera,
        radio_map=rm,
        rm_vmin=rm_vmin_db,
        clip_at=rm_clip_at,
        resolution=list(rm_resolution),
        filename="munich_radio_map_with_UEs.png",
        num_samples=rm_num_samples
    )

    
    # CIR generation    
    p_solver = PathSolver()
    a_list, tau_list = [], []
    max_num_paths = 0
    num_runs = int(np.ceil(target_num_cirs / batch_size_cir))

    for idx in range(num_runs):
        print(f"Progress: {idx+1}/{num_runs}", end="\r", flush=True)

        ue_pos, _ = rm.sample_positions(
            num_pos=batch_size_cir,
            metric="path_gain",
            min_val_db=min_gain_db,
            max_val_db=max_gain_db,
            min_dist=min_dist,
            max_dist=max_dist,
            seed=idx
        )

        for rx in range(batch_size_cir):
            p = ue_pos[0, rx, :]
            if hasattr(p, "numpy"):
                p = p.numpy()
            p = np.asarray(p, dtype=np.float64)
            scene.receivers[f"rx-{rx}"].position = (float(p[0]), float(p[1]), float(p[2]))

        paths = p_solver(
            scene,
            max_depth=max_depth,
            max_num_paths_per_src=10000
        )
        a, tau = paths.cir(
            sampling_frequency=subcarrier_spacing,
            num_time_steps=num_time_steps,
            out_type="numpy"
        )
        a = a.astype(np.complex64)
        tau = tau.astype(np.float32)
        a_list.append(a)
        tau_list.append(tau)

        num_paths = a.shape[-2]
        max_num_paths = max(max_num_paths, num_paths)
    
    # Padding + stacking    
    a, tau = [], []
    for a_, tau_ in zip(a_list, tau_list):
        num_paths = a_.shape[-2]
        a.append(
            np.pad(
                a_,
                [[0, 0], [0, 0], [0, 0], [0, 0],
                 [0, max_num_paths - num_paths], [0, 0]],
                constant_values=0
            ).astype(np.complex64)
        )

        tau.append(
            np.pad(
                tau_,
                [[0, 0], [0, 0],
                 [0, max_num_paths - num_paths]],
                constant_values=0
            ).astype(np.float32)
        )

    a = np.concatenate(a, axis=0)
    tau = np.concatenate(tau, axis=0)
    
    # Reorder dimensions    
    a = np.transpose(a, (2, 3, 0, 1, 4, 5))
    tau = np.transpose(tau, (1, 0, 2))

    a = np.expand_dims(a, axis=0)
    tau = np.expand_dims(tau, axis=0)

    a = np.transpose(a, [3, 1, 2, 0, 4, 5, 6])
    tau = np.transpose(tau, [2, 1, 0, 3])
    
    # Remove empty CIRs    
    p_link = np.sum(np.abs(a) ** 2, axis=(1, 2, 3, 4, 5, 6))
    a = a[p_link > 0, ...]
    tau = tau[p_link > 0, ...]
    
    # CIRDataset construction    
    cir_generator = CIRGenerator(a, tau, num_ue)
    channel_model = CIRDataset(
        cir_generator,
        batch_size,
        num_bs,
        num_bs_ant,
        num_ue,
        num_ue_ant,
        max_num_paths,
        num_time_steps
    )

    return channel_model
