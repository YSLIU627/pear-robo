"""Stage-1 RL curve of this reproduction: reads repro/metrics/metrics_*.jsonl, writes s2_curve.csv and s2_success_curve.png.

The success rate is `training/filter_groups/avg@n`: every trajectory of every group judged in that step, before the
DAPO group filter. Judged groups G = evicted_samples / (passrate/zero + passrate/one), since the filter evicts exactly
the all-0 and all-1 groups; trajectories n = 8G, successes = avg@n * n (an integer, which checks the count). The 95%
interval is Wilson over trajectories; the 8 trajectories of a group share one task, so the true interval is wider.
Usage: python repro/plot_s2_curve.py [metrics_dir] [out_dir]   (needs matplotlib)
"""
import csv
import glob
import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N_PER_GROUP = 8
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "metrics")
OUT = sys.argv[2] if len(sys.argv) > 2 else HERE


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


rows = {}
for f in sorted(glob.glob(os.path.join(SRC, "metrics_*.jsonl"))):
    for line in open(f):
        if not line.strip():
            continue
        d = json.loads(line)
        m = d.get("data", d)
        step = int(d["step"])
        filtered = m["training/filter_groups/passrate/zero"] + m["training/filter_groups/passrate/one"]
        groups = round(m["training/filter_groups/evicted_samples"] / filtered)
        n = groups * N_PER_GROUP
        k = round(m["training/filter_groups/avg@n"] * n)
        lo, hi = wilson(k, n)
        rows[step] = dict(
            step=step, judged_groups=groups, trajectories=n, successes=k,
            success_rate=k / n, ci95_lo=lo, ci95_hi=hi,
            groups_with_a_success=1 - m["training/filter_groups/passrate/zero"],
            step_s=round(m["timing_s/step"]), gen_s=round(m["timing_s/gen"]),
            update_s=round(m["timing_s/update_actor"]), save_s=round(m.get("timing_s/save_checkpoint", 0)),
        )
rows = [rows[s] for s in sorted(rows)]
os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "s2_curve.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    for r in rows:
        w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})

SURFACE, INK, INK2, GRID, SERIES = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"
steps = [r["step"] for r in rows]
pct = [100 * r["success_rate"] for r in rows]
lo = [100 * (r["success_rate"] - r["ci95_lo"]) for r in rows]
hi = [100 * (r["ci95_hi"] - r["success_rate"]) for r in rows]
fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=200, facecolor=SURFACE)
ax.set_facecolor(SURFACE)
ax.errorbar(steps, pct, yerr=[lo, hi], color=SERIES, ecolor=SERIES, elinewidth=1, capsize=0, alpha=0.9,
            linewidth=2, marker="o", markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
for x, y in ((steps[0], pct[0]), (steps[-1], pct[-1])):
    ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8, color=INK)
ax.set_xticks(steps)
ax.set_xlabel("RL step (each trains on 32 kept groups x 8; the rate covers every sampled group, before the DAPO filter)",
              color=INK2, fontsize=8)
ax.set_ylabel("training success rate (%)", color=INK2, fontsize=9)
ax.set_ylim(0, max(100 * r["ci95_hi"] for r in rows) * 1.15)
ax.grid(axis="y", color=GRID, linewidth=0.8)
ax.tick_params(colors=INK2, labelsize=8, length=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title("PEAR stage 1 from the base model, scaled-down batch: success rate per step (95% CI)",
             loc="left", fontsize=9.5, color=INK)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "s2_success_curve.png"), facecolor=SURFACE)
for r in rows:
    print(r["step"], r["judged_groups"], r["trajectories"], r["successes"], f"{100*r['success_rate']:.2f}",
          f"[{100*r['ci95_lo']:.1f}, {100*r['ci95_hi']:.1f}]", f"{r['groups_with_a_success']:.3f}", r["step_s"])
