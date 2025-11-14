from dataclasses import dataclass, field
from typing import Tuple
from enum import IntEnum
import numpy as np


class AntennaPattern(IntEnum):
    ISO = 0
    TR38901 = 1


@dataclass
class Config:
    """
    Central configuration for the PUSCH RT demo.
    """

    # -------------------------------
    # Hard-coded system parameters
    # -------------------------------
    _subcarrier_spacing: float = field(init=False, default=30e3, repr=False)   # Hz
    _num_time_steps: int = field(init=False, default=14, repr=False)           # OFDM symbols per slot

    _num_ue: int = field(init=False, default=4, repr=False)                    # users
    _num_bs: int = field(init=False, default=1, repr=False)                    # base-stations
    _num_ue_ant: int = field(init=False, default=4, repr=False)                # UE antennas
    _num_bs_ant: int = field(init=False, default=16, repr=False)               # BS antennas

    _batch_size_cir: int = field(init=False, default=40, repr=False)          # batch for CIR generation
    _target_num_cirs: int = field(init=False, default=80, repr=False)         # total CIRs to generate

    # Path solver / radio map
    _max_depth: int = field(init=False, default=5, repr=False)                 # max reflections
    _min_gain_db: float = field(init=False, default=-130.0, repr=False)        # ignore below this path gain
    _max_gain_db: float = field(init=False, default=0.0, repr=False)           # ignore above this path gain
    _min_dist_m: float = field(init=False, default=5.0, repr=False)            # sampling annulus: inner radius
    _max_dist_m: float = field(init=False, default=400.0, repr=False)          # sampling annulus: outer radius

    # Radio map rendering (purely for visualization; values unchanged)
    _rm_cell_size: Tuple[float, float] = field(init=False, default=(1.0, 1.0), repr=False)
    _rm_samples_per_tx: int = field(init=False, default=10000, repr=False)
    _rm_vmin_db: float = field(init=False, default=-110.0, repr=False)
    _rm_clip_at: float = field(init=False, default=12.0, repr=False)
    _rm_resolution: Tuple[int, int] = field(init=False, default=(650, 500), repr=False)
    _rm_num_samples: int = field(init=False, default=4096, repr=False)

    # BER/BLER simulation
    _batch_size: int = field(init=False, default=20, repr=False)               # must match CIRDataset batch size

    # Internal seed (kept for parity with ref style)
    _seed: int = field(init=False, default=42, repr=False)

    # ---------------
    # Read-only props
    # ---------------

    # Lowercase get-methods (config_nrx-style)
    @property
    def subcarrier_spacing(self) -> float:
        return self._subcarrier_spacing

    @property
    def num_time_steps(self) -> int:
        return self._num_time_steps

    @property
    def num_ue(self) -> int:
        return self._num_ue

    @property
    def num_bs(self) -> int:
        return self._num_bs

    @property
    def num_ue_ant(self) -> int:
        return self._num_ue_ant

    @property
    def num_bs_ant(self) -> int:
        return self._num_bs_ant

    @property
    def batch_size_cir(self) -> int:
        return self._batch_size_cir

    @property
    def target_num_cirs(self) -> int:
        return self._target_num_cirs

    @property
    def max_depth(self) -> int:
        return self._max_depth

    @property
    def min_gain_db(self) -> float:
        return self._min_gain_db

    @property
    def max_gain_db(self) -> float:
        return self._max_gain_db

    @property
    def min_dist_m(self) -> float:
        return self._min_dist_m

    @property
    def max_dist_m(self) -> float:
        return self._max_dist_m

    @property
    def rm_cell_size(self) -> Tuple[float, float]:
        return self._rm_cell_size

    @property
    def rm_samples_per_tx(self) -> int:
        return self._rm_samples_per_tx

    @property
    def rm_vmin_db(self) -> float:
        return self._rm_vmin_db

    @property
    def rm_clip_at(self) -> float:
        return self._rm_clip_at

    @property
    def rm_resolution(self) -> Tuple[int, int]:
        return self._rm_resolution

    @property
    def rm_num_samples(self) -> int:
        return self._rm_num_samples

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def seed(self) -> int:
        return self._seed

    # Convenience aliases (mirroring existing uppercase accessors)
    @property
    def SUBCARRIER_SPACING(self) -> float:
        return self.subcarrier_spacing

    @property
    def NUM_TIME_STEPS(self) -> int:
        return self.num_time_steps

    @property
    def NUM_UE(self) -> int:
        return self.num_ue

    @property
    def NUM_BS(self) -> int:
        return self.num_bs

    @property
    def NUM_UE_ANT(self) -> int:
        return self.num_ue_ant

    @property
    def NUM_BS_ANT(self) -> int:
        return self.num_bs_ant

    @property
    def BATCH_SIZE_CIR(self) -> int:
        return self.batch_size_cir

    @property
    def TARGET_NUM_CIRS(self) -> int:
        return self.target_num_cirs

    @property
    def MAX_DEPTH(self) -> int:
        return self.max_depth

    @property
    def MIN_GAIN_DB(self) -> float:
        return self.min_gain_db

    @property
    def MAX_GAIN_DB(self) -> float:
        return self.max_gain_db

    @property
    def MIN_DIST(self) -> float:
        return self.min_dist_m

    @property
    def MAX_DIST(self) -> float:
        return self.max_dist_m

    @property
    def RM_CELL_SIZE(self) -> Tuple[float, float]:
        return self.rm_cell_size

    @property
    def RM_SAMPLES_PER_TX(self) -> int:
        return self.rm_samples_per_tx

    @property
    def RM_VMIN_DB(self) -> float:
        return self.rm_vmin_db

    @property
    def RM_CLIP_AT(self) -> float:
        return self.rm_clip_at

    @property
    def RM_RESOLUTION(self) -> Tuple[int, int]:
        return self.rm_resolution

    @property
    def RM_NUM_SAMPLES(self) -> int:
        return self.rm_num_samples

    @property
    def BATCH_SIZE(self) -> int:
        return self.batch_size
