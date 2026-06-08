"""
make_cube3d_analysis.py
=======================

Run the three 3D interactions and produce two analysis figures:

  cube3d_lineage_tree.png   the spatial lineage tree of the longest-running
                            clone for each interaction, edges coloured by time
  cube3d_growth.png         growth correlation length, growth versus height
                            (is there a lucky place), and growth over time

Usage:
    python scripts/make_cube3d_analysis.py [--frames N]
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config3d import Config3D
import cube3d as C
import analysis3d as A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=None)
    args = ap.parse_args()

    cfg = Config3D()
    if args.frames:
        cfg.n_frames = args.frames

    here = os.path.dirname(__file__)
    figdir = os.path.abspath(os.path.join(here, "..", "figures"))
    os.makedirs(figdir, exist_ok=True)

    gens, frames_runs = [], []
    for inter in C.ALL3D:
        t0 = time.time()
        frames, _fh, gen = C.run(inter, cfg)
        gens.append((inter, gen))
        frames_runs.append((inter, frames))
        print(f"  {inter.row:14s} {frames[-1].pos.shape[0]:5d} cells, "
              f"{gen['parent'].shape[0]:5d} births  {time.time() - t0:5.1f}s")

    p1 = A.lineage_tree_figure(gens, cfg,
                               os.path.join(figdir, "cube3d_lineage_tree.png"))
    print("wrote", p1)
    pg = A.lineage_tree_gif(gens, cfg,
                            os.path.join(figdir, "cube3d_lineage_tree.gif"))
    print("wrote", pg)
    p2 = A.growth_analysis_figure(frames_runs, cfg,
                                  os.path.join(figdir, "cube3d_growth.png"))
    print("wrote", p2)

    pcg = A.clones_all_gif(gens, cfg,
                           os.path.join(figdir, "cube3d_clones.gif"))
    print("wrote", pcg)
    pcp = A.clones_all_figure(gens, cfg,
                              os.path.join(figdir, "cube3d_clones.png"))
    print("wrote", pcp)


if __name__ == "__main__":
    main()
