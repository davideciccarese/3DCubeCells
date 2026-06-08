"""
make_cube3d.py
==============

Grow the three interactions (commensalism, public good, facultative mutualism)
as 3D range expansions in a cube and render a 3x3 GIF: rows are the
interactions, columns are cells-by-strain, the nutrient field in 3D, and
cells-by-lineage. Cells start on the floor and grow up; any cell that leaves the
cube is dropped.

Usage:
    python scripts/make_cube3d.py [--out PATH] [--frames N]
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config3d import Config3D
import cube3d as C
import render3d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--frames", type=int, default=None)
    args = ap.parse_args()

    cfg = Config3D()
    if args.frames:
        cfg.n_frames = args.frames

    here = os.path.dirname(__file__)
    figdir = os.path.abspath(os.path.join(here, "..", "figures"))
    os.makedirs(figdir, exist_ok=True)
    out = args.out or os.path.join(figdir, "cube3d_multipanel.gif")

    results = []
    for inter in C.ALL3D:
        t0 = time.time()
        keys = [k for (_lab, k) in inter.display_fields]
        frames, fh, _gen = C.run(inter, cfg, capture_fields=keys)
        flist = [(lab, fh[k]) for (lab, k) in inter.display_fields]
        results.append((inter, frames, flist))
        print(f"  {inter.row:14s} {frames[-1].pos.shape[0]:5d} cells "
              f"{time.time() - t0:5.1f}s")

    t0 = time.time()
    render3d.multipanel3d_gif(
        results, cfg, out,
        title="Three interactions as 3D range expansions in a cube "
              "(after Dolinsek et al. 2016)")
    print(f"  render {time.time() - t0:5.1f}s")
    print("wrote", out)


if __name__ == "__main__":
    main()
