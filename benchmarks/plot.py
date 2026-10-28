"""
Figures of the benchmark (`bench.py`), from benchmarks/results/runs.jsonl into benchmarks/figures/:

- <system>.pdf: per method and setting, every replicate's relative error of θ, forecast error (normalized RMSE
  against the truth on the test range) and time. Fits that blew up or failed sit at the top, as crosses.
- trajectories_<system>.pdf: one panel per component (PT in two rows), replicate 0, the truth, the data and the fitted model of every method (in its setting
  with the smallest median forecast error), simulated over the whole range, with its training time in the legend
  and its normalized RMSE against the truth per component, on the training and on the test range, at the end of its line; the
  one with the smallest forecast error (nRMSE) in bold.
- overview.pdf: median forecast error against median time, every method and setting, one panel per system.

    python benchmarks/plot.py [--results benchmarks/results/runs.jsonl] [--out benchmarks/figures]
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter

from bench import RESULTS, SYSTEMS, dataset, simulate

METHODS = {"ms": "multiple shooting", "collocation": "collocation", "gm": "gradient matching", "im": "integral matching"}
SETTING = {"ms": "windows", "collocation": "seeds"}
COLORS = {"ms": "#2a78d6", "collocation": "#eb6834", "gm": "#1baf7a", "im": "#eda100"}  # categorical slots 1-4, fixed
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
METRICS = {"theta_err": "relative error of θ", "truth_test": "forecast error (nRMSE)", "time": "time [s]"}

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "axes.titlesize": 12,
    "axes.titlecolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK,
    "ytick.labelcolor": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
    "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def load(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    for row in rows:
        failed = row["status"] in ("error", "killed")
        for metric in ("theta_err", "truth_test"):
            value = row.get(metric, float("inf"))
            row[metric] = float("inf") if failed or value is None or not np.isfinite(value) else value
    return rows


def configurations(rows, system):
    """(method, setting) in the benchmark's order: methods as in METHODS, settings increasing."""
    present = {(r["method"], r["setting"]) for r in rows if r["system"] == system}
    return sorted(present, key=lambda c: (list(METHODS).index(c[0]), c[1] or 0))


def values(rows, system, config, metric):
    return np.array([r[metric] for r in rows if r["system"] == system and (r["method"], r["setting"]) == config])


def strip(ax, rows, system, metric):
    """Every replicate a dot, the median a bar, per configuration; non-finite values as crosses at the top."""
    configs = configurations(rows, system)
    finite = np.concatenate([v[np.isfinite(v)] for c in configs if len(v := values(rows, system, c, metric))])
    top = finite.max() * 3
    rng = np.random.default_rng(0)
    for i, config in enumerate(configs):
        v = values(rows, system, config, metric)
        color = COLORS[config[0]]
        ok = np.isfinite(v)
        x = i + rng.uniform(-0.18, 0.18, len(v))
        ax.scatter(x[ok], v[ok], s=22, color=color, alpha=0.75, linewidths=0, zorder=3)
        ax.scatter(x[~ok], np.full((~ok).sum(), top), s=30, marker="x", color=color, linewidths=1.5, zorder=3)
        if ok.any():
            median = np.median(v)  # inf counts: a method that fails in most replicates has an infinite median
            if np.isfinite(median):
                ax.plot([i - 0.32, i + 0.32], [median, median], color=INK, linewidth=2, zorder=4,
                        solid_capstyle="round")
    ax.set_yscale("log")
    ax.set_ylim(top=top * 2)
    ax.set_xlim(-0.6, len(configs) - 0.4)
    ax.set_xticks(range(len(configs)), [str(s) if s is not None else "" for _, s in configs])
    ax.tick_params(axis="x", length=0)
    # the method under its settings
    for method in dict.fromkeys(m for m, _ in configs):
        at = [i for i, (m, _) in enumerate(configs) if m == method]
        name = METHODS[method] if len(at) > 1 else {"gm": "GM", "im": "IM"}.get(method, method)
        if method in SETTING:
            name += f"\n({SETTING[method]})"
        ax.text(np.mean(at), -0.1, name, transform=ax.get_xaxis_transform(), ha="center", va="top",
                color=INK, fontsize=10)
    ax.set_title(METRICS[metric], loc="left")


def figure_system(rows, system, out):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, metric in zip(axes, METRICS):
        strip(ax, rows, system, metric)
    n_reps = len({r["rep"] for r in rows if r["system"] == system})
    fig.suptitle(f"{system}: {n_reps} replicates, dots = replicates, bars = medians, × = blew up / failed",
                 x=0.01, ha="left", color=MUTED, fontsize=10)
    fig.tight_layout(w_pad=2.5)
    fig.savefig(out / f"{system}.pdf")
    plt.close(fig)


def median(rows, system, config, metric):
    return float(np.median(values(rows, system, config, metric)))


def figure_overview(rows, systems, out):
    fig, axes = plt.subplots(1, len(systems), figsize=(4.2 * len(systems), 4.2), squeeze=False)
    for ax, system in zip(axes[0], systems):
        for method in METHODS:
            configs = [c for c in configurations(rows, system) if c[0] == method]
            if not configs:
                continue
            t = [median(rows, system, c, "time") for c in configs]
            e = [median(rows, system, c, "truth_test") for c in configs]
            ax.plot(t, e, "-o", color=COLORS[method], linewidth=2, markersize=7, label=METHODS[method],
                    markeredgecolor="white", markeredgewidth=1.5)
            for ti, ei, (_, s) in zip(t, e, configs):
                if s is not None and np.isfinite(ei):
                    ax.annotate(str(s), (ti, ei), textcoords="offset points", xytext=(5, 4), fontsize=8,
                                color=MUTED)
        ax.set(xscale="log", yscale="log", xlabel="median time [s]")
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_minor_formatter(NullFormatter())
        ax.grid(True, axis="both")
        ax.set_title(system, loc="left")
    axes[0, 0].set_ylabel("median forecast error (nRMSE)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout()
    fig.savefig(out / "overview.pdf")
    plt.close(fig)


LINESTYLES = {"ms": (0, (6, 3)), "collocation": (0, (0.1, 2.5))}  # dashed and dotted: overlapping fits stay visible


def spread(y, gap, lo, hi):
    """y moved apart to at least `gap` from each other, within [lo, hi], in their order."""
    order = np.argsort(y)
    z = np.clip(y[order], lo, hi)
    for i in range(1, len(z)):
        z[i] = max(z[i], z[i - 1] + gap)
    z -= max(0.0, z[-1] - hi)
    for i in range(len(z) - 2, -1, -1):
        z[i] = min(z[i], z[i + 1] - gap)
    out = np.empty_like(z)
    out[order] = z
    return out


def figure_trajectories(rows, system, out, rep=0):
    sys_ = SYSTEMS[system]
    test_frac = next(r["test_frac"] for r in rows if r["system"] == system)
    ts, truth, ys, n_train, _ = dataset(system, rep, test_frac)
    t = np.linspace(ts[0], ts[-1], 400)
    train = t <= ts[n_train - 1]
    true = simulate(sys_, sys_.theta, sys_.x0, t)
    D = len(sys_.x0)
    names = {"lv": ["prey x₁", "predator x₂"], "pt": ["S", "Sd", "R", "RS", "Rpp"]}[system.split("-")[0]]
    # up to three panels in a row, more in two rows: two in the first, the rest in the second
    layout = [D] if D <= 3 else [2, D - 2]
    columns = int(np.lcm.reduce(layout))
    fig = plt.figure(figsize=(min(15, 4.4 * max(layout)), 3.6 * len(layout)))
    grid = fig.add_gridspec(len(layout), columns)
    axes = [fig.add_subplot(grid[i, j * (columns // n) : (j + 1) * (columns // n)])
            for i, n in enumerate(layout) for j in range(n)]
    for ax, d in zip(axes, range(D)):
        ax.plot(t, true[:, d], color=MUTED, linewidth=4, alpha=0.5, label="truth", solid_capstyle="round")
        ax.axvline(ts[n_train - 1], color=MUTED, linestyle="--", linewidth=1)
        lo, hi = np.nanmin(true[:, d]), np.nanmax(true[:, d])
        ax.set_ylim(lo - 0.3 * (hi - lo), hi + 0.3 * (hi - lo))  # a fit that runs off stays visible as leaving

    shown = []  # (method, config, row, fitted)
    for method in METHODS:
        configs = [c for c in configurations(rows, system) if c[0] == method]
        if not configs:
            continue
        config = min(configs, key=lambda c: median(rows, system, c, "truth_test"))
        row = next((r for r in rows if r["system"] == system and r["rep"] == rep
                    and (r["method"], r["setting"]) == config and "theta" in r), None)
        if row is not None:
            shown.append((method, config, row, simulate(sys_, np.array(row["theta"]), np.array(row["x0"]), t)))
    best = min(shown, key=lambda m: m[2]["truth_test"])[0]  # the smallest forecast error in this replicate: bold

    for method, config, row, fitted in shown:
        label = METHODS[method] + ("" if config[1] is None else
                                   f" ({config[1]} {SETTING[method].removesuffix('s') if config[1] == 1 else SETTING[method]})")
        label += f", {row['time']:.2f} s" if row["time"] < 1 else f", {row['time']:.1f} s"
        for ax, d in zip(axes, range(D)):
            ax.plot(t, fitted[:, d], color=COLORS[method], linewidth=3.5 if method == best else 2, label=label,
                    linestyle=LINESTYLES.get(method, "-"), dash_capstyle="round", zorder=2.5 if method == best else 2)

    # every fit's nRMSE against the truth (training / test range) at the end of its line, in its color
    scale = true.std(axis=0)
    number = lambda e: "<0.001" if e < 0.0005 else f"{e:.3f}" if np.isfinite(e) else "∞"
    for ax, d in zip(axes, range(D)):
        lo, hi = ax.get_ylim()
        ends = np.array([fitted[-1, d] if np.isfinite(fitted[-1, d]) else hi for *_, fitted in shown])
        for (method, _, _, fitted), y in zip(shown, spread(ends, 0.11 * (hi - lo), lo, hi)):
            error = lambda part: np.sqrt(np.mean((fitted[part, d] - true[part, d]) ** 2)) / scale[d]
            ax.annotate(f"{number(error(train))} / {number(error(~train))}", (t[-1], y), xytext=(5, 0),
                        textcoords="offset points", va="center", fontsize=11, color=COLORS[method],
                        fontweight="bold" if method == best else "normal", annotation_clip=False)

    for ax, d in zip(axes, range(D)):
        ax.scatter(ts[:n_train], ys[:n_train, d], s=20, color=INK, zorder=5, label="training data")
        ax.scatter(ts[n_train:], ys[n_train:, d], s=24, facecolors="white", edgecolors=INK, zorder=5,
                   label="test data")
        ax.set_title(names[d], loc="left")
        ax.set_xlabel("t")
    # the methods with their times above the panels, the truth and the data below
    handles, labels = axes[0].get_legend_handles_labels()
    fits = [i for i, label in enumerate(labels) if label.endswith(" s")]
    rest = [i for i in range(len(labels)) if i not in fits]
    fig.tight_layout(w_pad=1.5)
    legend = fig.legend([handles[i] for i in fits], [labels[i] for i in fits], loc="lower center", ncol=2,
                        bbox_to_anchor=(0.5, 1.0), fontsize=13)
    for text in legend.get_texts():
        if text.get_text().startswith(METHODS[best]):
            text.set_fontweight("bold")
    fig.legend([handles[i] for i in rest], [labels[i] for i in rest], loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 0.0), fontsize=12, title_fontsize=12,
               title="numbers at the line ends: nRMSE against the truth (RMSE / std of the true component), training / test range")
    fig.savefig(out / f"trajectories_{system}.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "figures")
    args = parser.parse_args()
    rows = load(args.results)
    args.out.mkdir(parents=True, exist_ok=True)
    systems = [s for s in SYSTEMS if any(r["system"] == s for r in rows)]
    for system in systems:
        figure_system(rows, system, args.out)
        with plt.rc_context({"font.size": 13, "axes.titlesize": 15}):
            figure_trajectories(rows, system, args.out)
    figure_overview(rows, systems, args.out)
    print(f"figures of {len(rows)} fits in {args.out}")


if __name__ == "__main__":
    main()
