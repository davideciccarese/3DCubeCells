"""
config3d.py
===========

All tunable numbers for the 3D cube model in one place.
"""

from dataclasses import dataclass


@dataclass
class Config3D:
    # cube and grid
    cube: float = 14.0          # small cube the colony grows to fill
    N: int = 32                 # field grid nodes per side
    dt: float = 0.05
    steps_per_frame: int = 4
    n_frames: int = 30
    relax_iters: int = 6
    seed: int = 0

    # cells
    R: float = 0.5              # rod radius
    L_birth: float = 1.3        # length at birth
    L_div: float = 2.6          # length triggering division
    n_seed: int = 130           # founders carpeting the floor
    seed_radius: float = 6.0    # radius of the founder patch

    # 3D contact mechanics (overdamped, force and torque)
    k_contact: float = 0.50     # cell-cell repulsion stiffness
    k_floor: float = 0.70       # floor push-up stiffness
    rot_drag: float = 0.020     # rotational drag coefficient (x length cubed)

    # range-expansion front (only the exposed surface grows)
    front_radius: float = 2.2
    front_lo: float = 0.18
    front_hi: float = 0.60

    # growth kinetics
    g_max: float = 4.0
    K_S: float = 0.15
    K_M: float = 0.12
    K_P: float = 0.14

    # active-interaction costs and public-good gain
    cost_public_good: float = 0.30
    pg_gain: float = 1.5
    fac_base: float = 0.55

    # stoichiometry
    Y_consume: float = 1.80     # strong uptake so consumption is clearly visible
    Y_produce: float = 0.055

    # field transport
    D_S: float = 0.4            # slow diffusion -> strong consumption gradient
    D_M: float = 4.0
    D_P: float = 4.0
    S0: float = 1.0
    decay_M: float = 0.02
    decay_P: float = 0.02

    @property
    def dx(self):
        return self.cube / self.N

    @property
    def center(self):
        return self.cube * 0.5
