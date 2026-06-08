"""
field3d.py
==========

A scalar field on a 3D grid, evolved by explicit diffusion with first order
decay. Two boundary styles, matching a colony attached to a floor inside an open
medium:

  reservoir  the substrate is held at a fixed value on the five open faces (top
             and four sides), like nutrient supplied from the surrounding medium,
             while the floor (z = 0) is no flux (the attachment surface).
  zeroflux   secreted molecules use no flux on every face, so they stay local
             and build up next to the cells that make them.

Grid index order is [ix, iy, iz]; iz = 0 is the floor.
"""

import numpy as np


class Field3D:
    def __init__(self, N, dx, D, dt, c0=0.0, boundary="reservoir",
                 reservoir=1.0, decay=0.0):
        self.N = N
        self.dx = dx
        self.D = D
        self.dt = dt
        self.boundary = boundary
        self.reservoir = reservoir
        self.decay = decay
        self.c = np.full((N, N, N), c0, dtype=np.float64)
        # explicit-diffusion substep count for stability: D dt / dx^2 < 1/6 in 3D
        r = D * dt / (dx * dx)
        self.sub = max(1, int(np.ceil(r / 0.12)))
        self.rs = r / self.sub
        self._apply_bc()

    def _apply_bc(self):
        c = self.c
        if self.boundary == "reservoir":
            R = self.reservoir
            c[0, :, :] = R
            c[-1, :, :] = R
            c[:, 0, :] = R
            c[:, -1, :] = R
            c[:, :, -1] = R           # top
            c[:, :, 0] = c[:, :, 1]   # floor, no flux
        elif self.boundary == "top":
            # supplied only from the top face, like nutrient diffusing in from
            # the medium above; sides and floor are no flux, so the colony draws
            # a real top-to-bottom gradient
            R = self.reservoir
            c[:, :, -1] = R
            c[:, :, 0] = c[:, :, 1]
            c[0, :, :] = c[1, :, :]
            c[-1, :, :] = c[-2, :, :]
            c[:, 0, :] = c[:, 1, :]
            c[:, -1, :] = c[:, -2, :]
        else:  # zeroflux on all faces
            c[0, :, :] = c[1, :, :]
            c[-1, :, :] = c[-2, :, :]
            c[:, 0, :] = c[:, 1, :]
            c[:, -1, :] = c[:, -2, :]
            c[:, :, 0] = c[:, :, 1]
            c[:, :, -1] = c[:, :, -2]

    def step(self):
        c = self.c
        for _ in range(self.sub):
            lap = (
                np.roll(c, 1, 0) + np.roll(c, -1, 0)
                + np.roll(c, 1, 1) + np.roll(c, -1, 1)
                + np.roll(c, 1, 2) + np.roll(c, -1, 2)
                - 6.0 * c
            )
            c += self.rs * lap
            if self.decay:
                c -= self.decay * self.dt / self.sub * c
            self._apply_bc()

    # ---- cell coupling: nearest node sampling and deposition ----
    def _idx(self, pts):
        i = np.clip((pts / self.dx).astype(int), 0, self.N - 1)
        return i[:, 0], i[:, 1], i[:, 2]

    def sample(self, pts):
        ix, iy, iz = self._idx(pts)
        return self.c[ix, iy, iz]

    def deposit(self, pts, amounts):
        ix, iy, iz = self._idx(pts)
        np.add.at(self.c, (ix, iy, iz), amounts)
        np.clip(self.c, 0.0, None, out=self.c)
