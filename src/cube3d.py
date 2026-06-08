"""
cube3d.py
=========

A 3D range expansion inside a cube. Cells are spherocylinders that start as a
small patch of founders on the floor and grow upward and outward. Only the
exposed surface of the colony grows; buried cells freeze. Cells whose centre
leaves the cube are dropped: they have simply grown out of the box.

Three interactions are provided, the three the brief asks for:
  commensalism      A leaks a by-product M for free; B scavenges it (0, +)
  public_good       A pays to secrete a public good P; B free-rides   (-, +)
  mutualism         facultative reciprocal by-product exchange         (+, +)

State is held in plain arrays so the whole thing stays readable:
  pos (n,3) centre, ax (n,3) unit axis, L (n,) cylinder length, R radius,
  sp (n,) strain 0/1, lin (n,) founder lineage id, alive (n,) bool.
"""

import numpy as np
from scipy.spatial import cKDTree

from field3d import Field3D


def monod(c, k):
    return c / (k + c)


# ----------------------------------------------------------------------
# segment to segment closest distance in 3D, vectorised over pairs
# (clamped, after Ericson, Real-Time Collision Detection)
# ----------------------------------------------------------------------
def seg_seg(P1, Q1, P2, Q2):
    d1 = Q1 - P1
    d2 = Q2 - P2
    r = P1 - P2
    a = np.einsum("ij,ij->i", d1, d1)
    e = np.einsum("ij,ij->i", d2, d2)
    f = np.einsum("ij,ij->i", d2, r)
    c = np.einsum("ij,ij->i", d1, r)
    b = np.einsum("ij,ij->i", d1, d2)
    denom = a * e - b * b
    s = np.where(denom > 1e-9, np.clip((b * f - c * e) / np.where(denom > 1e-9, denom, 1.0), 0, 1), 0.0)
    t = (b * s + f) / np.where(e > 1e-9, e, 1.0)
    t = np.clip(t, 0, 1)
    s = np.clip((b * t - c) / np.where(a > 1e-9, a, 1.0), 0, 1)
    c1 = P1 + d1 * s[:, None]
    c2 = P2 + d2 * t[:, None]
    diff = c1 - c2
    dist = np.sqrt(np.einsum("ij,ij->i", diff, diff) + 1e-12)
    return c1, c2, dist


class Colony3D:
    def __init__(self, cfg, rng):
        self.cfg = cfg
        self.rng = rng
        self.pos = np.zeros((0, 3))
        self.ax = np.zeros((0, 3))
        self.L = np.zeros(0)
        self.sp = np.zeros(0, dtype=int)
        self.lin = np.zeros(0, dtype=int)
        self.cid = np.zeros(0, dtype=int)      # unique id of each living cell
        self.alive = np.zeros(0, dtype=bool)
        self.geff = np.zeros(0)                # effective growth at last step
        self.R = cfg.R
        # persistent genealogy, indexed by cid (never culled), for lineage trees
        self.par = []        # parent cid (-1 for founders)
        self.bpos = []       # birth position
        self.bframe = []     # birth frame
        self.founder = []    # root founder id
        self.csp = []        # strain of each cid

    @property
    def n(self):
        return self.pos.shape[0]

    def seed_floor(self, frac_b):
        cfg = self.cfg
        c = cfg.cube * 0.5
        m = cfg.n_seed
        ang = self.rng.uniform(0, 2 * np.pi, m)
        rad = cfg.seed_radius * np.sqrt(self.rng.uniform(0, 1, m))
        x = c + rad * np.cos(ang)
        y = c + rad * np.sin(ang)
        z = np.full(m, self.R + 0.05)
        self.pos = np.column_stack((x, y, z))
        # founders lie near the floor, axes mostly horizontal with a little tilt
        theta = self.rng.uniform(0, 2 * np.pi, m)
        tilt = self.rng.uniform(0.0, 0.35, m)
        ux = np.cos(theta) * np.cos(tilt)
        uy = np.sin(theta) * np.cos(tilt)
        uz = np.sin(tilt)
        self.ax = np.column_stack((ux, uy, uz))
        self._normalize()
        self.L = np.full(m, cfg.L_birth)
        self.sp = (self.rng.uniform(size=m) < frac_b).astype(int)
        self.lin = np.arange(m)
        self.cid = np.arange(m)
        self.alive = np.ones(m, dtype=bool)
        self.geff = np.zeros(m)
        # found the genealogy: m founders, each its own root
        self.par = [-1] * m
        self.bpos = [self.pos[i].copy() for i in range(m)]
        self.bframe = [0] * m
        self.founder = list(range(m))
        self.csp = [int(x) for x in self.sp]

    def _normalize(self):
        n = np.linalg.norm(self.ax, axis=1, keepdims=True)
        n[n == 0] = 1.0
        self.ax = self.ax / n

    def endpoints(self):
        h = 0.5 * self.L[:, None] * self.ax
        return self.pos - h, self.pos + h

    def exposure(self):
        """Per cell surface-exposure score in [0,1]: 1 = sticking out, 0 = buried.

        Built from the resultant of unit vectors to nearby cells. A buried cell
        has neighbours all around so the resultant is small; a surface cell has
        neighbours mostly on one side so the resultant is large. A small upward
        bonus lets the top of the colony grow into the open medium.
        """
        cfg = self.cfg
        if self.n == 0:
            return np.zeros(0)
        tree = cKDTree(self.pos)
        nbrs = tree.query_ball_point(self.pos, cfg.front_radius)
        score = np.zeros(self.n)
        for i, nb in enumerate(nbrs):
            nb = [j for j in nb if j != i]
            if not nb:
                score[i] = 1.0
                continue
            d = self.pos[nb] - self.pos[i]
            dn = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
            res = np.linalg.norm(dn.mean(axis=0))
            score[i] = res
        return score

    def front_factor(self):
        s = self.exposure()
        cfg = self.cfg
        phi = np.clip((s - cfg.front_lo) / (cfg.front_hi - cfg.front_lo), 0, 1)
        return phi

    def grow(self, mu, dt):
        self.L = self.L + mu * dt

    def divide(self, frame=0):
        cfg = self.cfg
        idx = np.where(self.alive & (self.L >= cfg.L_div))[0]
        if idx.size == 0:
            return
        half = 0.25 * self.L[idx]
        jitter = self.rng.normal(0, 0.05, (idx.size, 3))
        newax = self.ax[idx] + jitter            # daughters keep the parent axis
        # parent becomes one daughter, child the other
        offs = half[:, None] * self.ax[idx]
        child_pos = self.pos[idx] + offs
        self.pos[idx] = self.pos[idx] - offs
        self.L[idx] = cfg.L_birth

        # new unique ids and genealogy for the children
        base = len(self.par)
        new_cid = np.arange(base, base + idx.size)
        parent_cid = self.cid[idx]
        for k in range(idx.size):
            pc = int(parent_cid[k])
            self.par.append(pc)
            self.bpos.append(child_pos[k].copy())
            self.bframe.append(frame)
            self.founder.append(self.founder[pc])
            self.csp.append(int(self.sp[idx[k]]))

        self.pos = np.vstack((self.pos, child_pos))
        self.ax = np.vstack((self.ax, newax))
        self.L = np.concatenate((self.L, np.full(idx.size, cfg.L_birth)))
        self.sp = np.concatenate((self.sp, self.sp[idx].copy()))
        self.lin = np.concatenate((self.lin, self.lin[idx].copy()))
        self.cid = np.concatenate((self.cid, new_cid))
        self.alive = np.concatenate((self.alive, np.ones(idx.size, dtype=bool)))
        self.geff = np.concatenate((self.geff, np.zeros(idx.size)))
        self._normalize()

    def spine_points(self, m=5):
        """m points sampled evenly along each cell's spine, shape (n, m, 3)."""
        t = np.linspace(-0.5, 0.5, m)
        return self.pos[:, None, :] + (t[None, :, None] * self.L[:, None, None]) * self.ax[:, None, :]

    def relax(self, iters=8):
        """Overdamped rigid-rod mechanics in 3D.

        Each contact applies a repulsive force along the contact normal at the
        contact point. The force translates a cell (drag grows with length) and,
        because it acts off the centre, also torques it (rotational drag grows
        with length cubed). The floor pushes up on any part of a cell dipping
        below it. There is no scripted upward bias: cells lie down, and the
        vertical structure that appears is buckling, the colony being shoved up
        out of a crowded basal layer. Per-iteration moves are clamped so the
        explicit solver stays stable.
        """
        cfg = self.cfg
        R = self.R
        R2 = 2 * R
        for _ in range(iters):
            n = self.n
            if n < 1:
                break
            F = np.zeros((n, 3))
            Tq = np.zeros((n, 3))

            # cell to cell contacts
            if n >= 2:
                P, Q = self.endpoints()
                tree = cKDTree(self.pos)
                cutoff = self.L.max() + R2 + 0.5
                pairs = tree.query_pairs(cutoff, output_type="ndarray")
                if pairs.size:
                    i, j = pairs[:, 0], pairs[:, 1]
                    c1, c2, dist = seg_seg(P[i], Q[i], P[j], Q[j])
                    ov = R2 - dist
                    hit = ov > 0
                    if np.any(hit):
                        i, j = i[hit], j[hit]
                        c1, c2 = c1[hit], c2[hit]
                        dist, ov = dist[hit], ov[hit]
                        nrm = (c1 - c2) / dist[:, None]
                        f = cfg.k_contact * ov[:, None] * nrm
                        np.add.at(F, i, f)
                        np.add.at(F, j, -f)
                        np.add.at(Tq, i, np.cross(c1 - self.pos[i], f))
                        np.add.at(Tq, j, np.cross(c2 - self.pos[j], -f))

            # floor contact: any spine sample below z = R is pushed up
            s = self.spine_points(5)
            pen = R - s[:, :, 2]
            below = pen > 0
            if np.any(below):
                fz = np.zeros_like(s)
                fz[:, :, 2] = cfg.k_floor * np.where(below, pen, 0.0)
                F += fz.sum(axis=1)
                Tq += np.cross(s - self.pos[:, None, :], fz).sum(axis=1)

            # overdamped update with size-dependent drag, clamped for stability
            zt = np.maximum(self.L, 1.0)[:, None]
            zr = np.maximum(self.L ** 3 * cfg.rot_drag, 1e-3)[:, None]
            dpos = F / zt
            mag = np.linalg.norm(dpos, axis=1, keepdims=True)
            dpos *= np.minimum(1.0, (0.5 * R) / (mag + 1e-9))
            self.pos = self.pos + dpos

            omega = Tq / zr
            wn = np.linalg.norm(omega, axis=1, keepdims=True)
            omega *= np.minimum(1.0, 0.25 / (wn + 1e-9))
            self.ax = self.ax + np.cross(omega, self.ax)
            self._normalize()
            self._floor()

    def _floor(self):
        zlow = self.pos[:, 2] - 0.5 * self.L * np.abs(self.ax[:, 2])
        below = zlow < self.R
        self.pos[below, 2] += (self.R - zlow[below])

    def cull_outside(self):
        """Drop cells whose centre has left the cube (they grew out of the box)."""
        c = self.cfg.cube
        p = self.pos
        keep = (
            (p[:, 0] >= 0) & (p[:, 0] <= c)
            & (p[:, 1] >= 0) & (p[:, 1] <= c)
            & (p[:, 2] >= 0) & (p[:, 2] <= c)
            & self.alive
        )
        if keep.all():
            return
        self.pos = self.pos[keep]
        self.ax = self.ax[keep]
        self.L = self.L[keep]
        self.sp = self.sp[keep]
        self.lin = self.lin[keep]
        self.cid = self.cid[keep]
        self.geff = self.geff[keep]
        self.alive = self.alive[keep]


# ----------------------------------------------------------------------
# interactions: each returns mu (elongation rate) and edits the fields
# ----------------------------------------------------------------------
class Inter3D:
    name = "base"
    row = "base"
    signs = ("0", "0")
    seed_frac = 0.5

    def fields(self, cfg):
        return {}

    def step(self, col, F, cfg, dt):
        raise NotImplementedError


def _S(cfg):
    return Field3D(cfg.N, cfg.dx, cfg.D_S, cfg.dt, c0=cfg.S0,
                   boundary="top", reservoir=cfg.S0)


def _deposit_body(field, col, per_cell, msample=3):
    """Deposit a per-cell amount spread over the cell body (several spine
    points), which is more realistic than a point sink and couples the cells to
    the field strongly enough to draw a real gradient."""
    pts = col.spine_points(msample)                  # (n, m, 3)
    flat = pts.reshape(-1, 3)
    amt = np.repeat(per_cell / msample, msample)
    field.deposit(flat, amt)


class Commensalism3D(Inter3D):
    name = "Commensalism (0, +)"
    row = "commensalism"
    signs = ("0", "+")
    display_fields = [("Substrate S (feeds A)", "S"),
                      ("Metabolite M (feeds B)", "M")]

    def fields(self, cfg):
        return {"S": _S(cfg),
                "M": Field3D(cfg.N, cfg.dx, cfg.D_M, cfg.dt, c0=0.0,
                             boundary="zeroflux", decay=cfg.decay_M)}

    def step(self, col, F, cfg, dt):
        p = col.pos
        S = F["S"].sample(p)
        M = F["M"].sample(p)
        a = col.sp == 0
        b = ~a
        mu = np.zeros(col.n)
        mu[a] = cfg.g_max * monod(S[a], cfg.K_S)
        mu[b] = cfg.g_max * monod(M[b], 0.4 * cfg.K_M)
        _deposit_body(F["S"], col, np.where(a, -cfg.Y_consume * mu, 0.0) * dt)
        F["M"].deposit(p, np.where(a, 1.8 * cfg.Y_produce * mu,
                                   -cfg.Y_consume * mu) * dt)
        return mu


class PublicGood3D(Inter3D):
    name = "Public good (-, +)"
    row = "public_good"
    signs = ("-", "+")
    display_fields = [("Substrate S (feeds A & B)", "S"),
                      ("Public good P (feeds B)", "P")]

    def fields(self, cfg):
        return {"S": _S(cfg),
                "P": Field3D(cfg.N, cfg.dx, cfg.D_P, cfg.dt, c0=0.0,
                             boundary="zeroflux", decay=cfg.decay_P)}

    def step(self, col, F, cfg, dt):
        p = col.pos
        S = F["S"].sample(p)
        P = F["P"].sample(p)
        a = col.sp == 0
        b = ~a
        base = cfg.g_max * monod(S, cfg.K_S)
        mu = np.zeros(col.n)
        mu[a] = base[a] * (1.0 - cfg.cost_public_good)
        mu[b] = base[b] * (1.0 + cfg.pg_gain * monod(P[b], cfg.K_P))
        _deposit_body(F["S"], col, -cfg.Y_consume * mu * dt)
        F["P"].deposit(p, np.where(a, cfg.Y_produce * base, 0.0) * dt)
        return mu


class Mutualism3D(Inter3D):
    name = "Facultative mutualism (+, +)"
    row = "mutualism"
    signs = ("+", "+")
    display_fields = [("Cross-fed Mb (feeds A)", "Mb"),
                      ("Cross-fed Ma (feeds B)", "Ma")]

    def fields(self, cfg):
        return {"S": _S(cfg),
                "Ma": Field3D(cfg.N, cfg.dx, cfg.D_M, cfg.dt, c0=0.0,
                              boundary="zeroflux", decay=cfg.decay_M),
                "Mb": Field3D(cfg.N, cfg.dx, cfg.D_M, cfg.dt, c0=0.0,
                              boundary="zeroflux", decay=cfg.decay_M)}

    def step(self, col, F, cfg, dt):
        p = col.pos
        S = F["S"].sample(p)
        Ma = F["Ma"].sample(p)
        Mb = F["Mb"].sample(p)
        a = col.sp == 0
        b = ~a
        fb = cfg.fac_base
        s = monod(S, cfg.K_S)
        mu = np.zeros(col.n)
        mu[a] = cfg.g_max * s[a] * (fb + (1 - fb) * monod(Mb[a], cfg.K_M))
        mu[b] = cfg.g_max * s[b] * (fb + (1 - fb) * monod(Ma[b], cfg.K_M))
        _deposit_body(F["S"], col, -cfg.Y_consume * mu * dt)
        F["Ma"].deposit(p, np.where(a, cfg.Y_produce * mu,
                                    -0.5 * cfg.Y_consume * mu) * dt)
        F["Mb"].deposit(p, np.where(b, cfg.Y_produce * mu,
                                    -0.5 * cfg.Y_consume * mu) * dt)
        return mu


ALL3D = [Commensalism3D(), PublicGood3D(), Mutualism3D()]


class Snapshot3D:
    __slots__ = ("pos", "ax", "L", "sp", "lin", "cid", "g", "R")

    def __init__(self, col):
        self.pos = col.pos.copy()
        self.ax = col.ax.copy()
        self.L = col.L.copy()
        self.sp = col.sp.copy()
        self.lin = col.lin.copy()
        self.cid = col.cid.copy()
        self.g = col.geff.copy()       # effective growth rate at this frame
        self.R = col.R


def run(inter, cfg, capture_field=None, capture_fields=None):
    """Grow one interaction.

    Returns (frames, field_history, genealogy). capture_field captures one field
    by name; capture_fields captures several. Each frame carries per-cell growth
    g (= mu * front factor). genealogy holds, indexed by unique cell id: parent
    cid, birth position, birth frame, root founder and species, including cells
    that later left the cube, so a full lineage tree can be rebuilt.
    """
    rng = np.random.default_rng(cfg.seed)
    col = Colony3D(cfg, rng)
    col.seed_floor(inter.seed_frac)
    F = inter.fields(cfg)

    keys = list(capture_fields) if capture_fields else []
    if capture_field and capture_field not in keys:
        keys.append(capture_field)

    def record_growth():
        mu = inter.step(col, F, cfg, 0.0)      # dt=0: read mu, no field change
        col.geff = mu * col.front_factor()

    def grab():
        for k in keys:
            fhist[k].append(F[k].c.copy())

    record_growth()
    frames = [Snapshot3D(col)]
    fhist = {k: [] for k in keys}
    grab()

    for fr in range(1, cfg.n_frames):
        for _ in range(cfg.steps_per_frame):
            for f in F.values():
                f.step()
            mu = inter.step(col, F, cfg, cfg.dt)
            phi = col.front_factor()
            col.geff = mu * phi
            col.grow(mu * phi, cfg.dt)
            col.divide(frame=fr)
            col.relax(cfg.relax_iters)
            col.cull_outside()
        record_growth()
        frames.append(Snapshot3D(col))
        grab()

    genealogy = {
        "parent": np.array(col.par),
        "bpos": np.array(col.bpos),
        "bframe": np.array(col.bframe),
        "founder": np.array(col.founder),
        "sp": np.array(col.csp),
    }
    return frames, fhist, genealogy
