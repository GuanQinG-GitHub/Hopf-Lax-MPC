"""
export_plot_data.py -- extract from the (large, unversioned) sweep pickle exactly what the deliverable figure
needs, into a small versioned file, plus a human-readable summary table.

    data/N_sweep_plot_data.npz   per (method, N): tsolve, tcyc, recompile (per cycle), reached, Jpen, Jtrue, Jreal
                                 -> python plot_cycle_band.py --data data/N_sweep_plot_data.npz ...
    data/N_sweep_summary.csv     per (method, N): mean / std ms per cycle (the plotted line), reached,
                                 drove_through (Jpen > 500), Jtrue, Jtrack, Jobs, cycles, run timestamp

USAGE (from N_sweep/)
    python export_plot_data.py [--pkl results_N_sweep_shared_eps1e-06_wall_saddle_bumps_rand.pkl] [--outdir data]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle

import numpy as np

import plot_cycle_band as PB

FIELDS = ("tsolve", "tcyc", "recompile")                              # per-cycle arrays the plot reads
SCALARS = ("reached", "Jpen", "Jtrue", "Jreal")


def export(runs, config, out_npz):
    arrs = {}
    for (m, N), e in sorted(runs.items()):
        for f in FIELDS:
            arrs[f"{m}/{N}/{f}"] = np.asarray(e[f], dtype=bool if f == "recompile" else np.float64)
        for f in SCALARS:
            arrs[f"{m}/{N}/{f}"] = np.asarray(e[f], dtype=np.float64)
    arrs["__config__"] = np.asarray(json.dumps(config, default=str))
    np.savez_compressed(out_npz, **arrs)


def load(path):
    """npz written by export() -> {(method, N): run dict} with the keys plot_cycle_band reads."""
    Z = np.load(path)
    runs = {}
    for k in Z.files:
        if k.startswith("__"):
            continue
        m, N, f = k.rsplit("/", 2)
        v = Z[k]
        runs.setdefault((m, int(N)), {})[f] = v if f in FIELDS else (bool(v) if f == "reached" else float(v))
    return runs


def summary(runs, out_csv, pen=PB.PEN_THRESHOLD):
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "N", "ms_per_cycle_mean", "ms_per_cycle_std", "reached", "drove_through",
                    "Jtrue", "Jtrack", "Jobs", "cycles", "stamp"])
        for m in PB.ORDER:
            for N in sorted(n for (mm, n) in runs if mm == m):
                e = runs[(m, N)]
                tc = PB.cycle_times_ms(e)
                w.writerow([m, N, f"{tc.mean():.4f}", f"{tc.std():.4f}", int(bool(e["reached"])),
                            int(e["Jpen"] > pen), f"{e['Jtrue']:.4f}", f"{e['Jreal']:.4f}", f"{e['Jpen']:.4f}",
                            len(e["tcyc"]), e.get("stamp", "")])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkl", default="results_N_sweep_shared_eps1e-06_wall_saddle_bumps_rand.pkl")
    ap.add_argument("--outdir", default="data")
    args = ap.parse_args()
    with open(args.pkl, "rb") as f:
        D = pickle.load(f)
    os.makedirs(args.outdir, exist_ok=True)
    npz = os.path.join(args.outdir, "N_sweep_plot_data.npz")
    export(D["runs"], D.get("config", {}), npz)
    summary(D["runs"], os.path.join(args.outdir, "N_sweep_summary.csv"))
    back = load(npz)                                                   # round trip: the plotted numbers agree
    for key, e in D["runs"].items():
        assert np.array_equal(PB.cycle_times_ms(e), PB.cycle_times_ms(back[key])), key
        assert all(float(e[f]) == float(back[key][f]) for f in SCALARS), key
    print(f"wrote {npz} ({os.path.getsize(npz) / 1e6:.2f} MB, {len(back)} runs; round trip exact) and "
          f"{os.path.join(args.outdir, 'N_sweep_summary.csv')}")


if __name__ == "__main__":
    main()
