#!/usr/bin/env python3
"""
plot_results.py — Generate poster figures for the cross-view invariance benchmark.

Figures produced (saved to plots/):
  1. fig1_model_comparison.png   — exact match accuracy across all models (bar chart)
  2. fig2_accuracy_consistency.png — accuracy vs. viewpoint consistency (scatter)
  3. fig3_difficulty_bins.png    — exact match by camera misalignment bin (line chart)
  4. fig4_thinking_budget.png    — thinking budget vs. exact match (GPT-5.4 + Gemini)

Usage:
    python plot_results.py
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

# ── paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path("results/all_metrics")
OUT_DIR     = Path("plots")
OUT_DIR.mkdir(exist_ok=True)

# ── helpers ───────────────────────────────────────────────────────────────────
def load(fname: str) -> dict:
    with open(RESULTS_DIR / fname) as f:
        return json.load(f)

def exact_match(m: dict) -> float:
    return m["joint_metrics"]["exact_match_accuracy"]

def consistent_correct(m: dict) -> float:
    return m["viewpoint_consistency"]["consistent_correct_rate"]

def bin_exact_match(m: dict, bin_name: str) -> float:
    return m["difficulty_bins"][bin_name]["joint_metrics"]["exact_match_accuracy"]

# ── global style ──────────────────────────────────────────────────────────────
FIGSIZE = (15, 6)  # consistent 5:2 aspect ratio across all figures

# Random chance for exact match with enum prompting: (1/3)^3 axes = 1/27
RANDOM_CHANCE = (1 / 3) ** 3

plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "font.size":      19,
    "axes.titlesize": 21,
    "axes.labelsize": 19,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "legend.fontsize": 17,
    "figure.dpi":     150,
})

COLORS = {
    "open-source": "#8e44ad",
    "gpt":         "#2471a3",
    "gemini":      "#d35400",
}

# ── model registry ────────────────────────────────────────────────────────────
# (display_label, filename, group)
ALL_MODELS = [
    ("LLaVA-1.6 (7B)",  "llava_metrics.json",                                              "open-source"),
    ("Qwen-VL (9.6B)",  "qwen_metrics.json",                                               "open-source"),
    ("GPT-4o",                 "metrics_chatgpt_4o_500viewpoints.json",                    "gpt"),
    ("GPT-5.4 (none)",         "metrics_chatgpt_5.4_nothinking_500viewpoints.json",        "gpt"),
    ("GPT-5.4 (low)",          "metrics_chatgpt_5.4_lowthinking_500viewpoints.json",       "gpt"),
    ("GPT-5.4 (med)",          "metrics_chatgpt_5.4_medthinking_500viewpoints.json",       "gpt"),
    ("Gemini 2.5 Pro (128)",   "metrics_gemini_2.5pro_128thinking_500viewpoints.json",     "gemini"),
    ("Gemini 2.5 Pro (2048)",  "metrics_gemini_2.5pro_2048thinking_500viewpoints.json",    "gemini"),
    ("Gemini 2.5 Pro (8192)",  "metrics_gemini_2.5pro_8192thinking_500viewpoints.json",    "gemini"),
    ("Gemini 2.5 Pro (16384)", "metrics_gemini_2.5pro_16384thinking_500viewpoints.json",   "gemini"),
]

METRICS = {label: load(fname) for label, fname, _ in ALL_MODELS}

# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Overall model comparison (exact match bar chart)
# ══════════════════════════════════════════════════════════════════════════════
def fig1_model_comparison():
    labels  = [label for label, _, _ in ALL_MODELS]
    values  = [exact_match(METRICS[label]) for label in labels]
    colors  = [COLORS[group] for _, _, group in ALL_MODELS]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.8, zorder=3)

    # value labels on top of bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.005,
            f"{val:.1%}",
            ha="center", va="bottom", fontsize=16,
        )

    ax.axhline(RANDOM_CHANCE, color="grey", linestyle="--", linewidth=1.2)
    ax.set_ylabel("Exact Match Accuracy")
    ax.set_title("Model Comparison — Exact Match Accuracy (all 3 axes correct)")
    ax.set_ylim(0, max(values) * 1.18)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)

    # group legend
    legend_handles = [
        mlines.Line2D([], [], color=COLORS["open-source"], marker="s", linestyle="None", markersize=10, label="Open-source models"),
        mlines.Line2D([], [], color="grey", linestyle="--", linewidth=1.2, label=f"Random chance ({RANDOM_CHANCE:.1%})"),
        mlines.Line2D([], [], color=COLORS["gpt"],         marker="s", linestyle="None", markersize=10, label="GPT (OpenAI)"),
        mlines.Line2D([], [], color=COLORS["gemini"],      marker="s", linestyle="None", markersize=10, label="Gemini (Google)"),
    ]
    ax.legend(handles=legend_handles, loc="upper left")

    fig.tight_layout()
    path = OUT_DIR / "fig1_model_comparison.png"
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Accuracy vs. viewpoint consistency scatter
# ══════════════════════════════════════════════════════════════════════════════
def fig2_accuracy_consistency():
    fig, ax = plt.subplots(figsize=FIGSIZE)

    markers = {"open-source": "o", "gpt": "s", "gemini": "^"}

    # Manual offsets only for the bottom-left cluster that overlaps
    manual_offsets = {
        "LLaVA-1.6 (7B)":  (-10,  10),
        "Qwen-VL (9.6B)":   (  6, -14),
    }

    for label, _, group in ALL_MODELS:
        m = METRICS[label]
        x = exact_match(m)
        y = consistent_correct(m)
        ax.scatter(x, y,
                   color=COLORS[group], marker=markers[group],
                   s=100, zorder=3, edgecolors="white", linewidths=0.6)
        ox, oy = manual_offsets.get(label, (6, 4))
        ax.annotate(label, (x, y),
                    textcoords="offset points", xytext=(ox, oy),
                    fontsize=12, color="#333333")

    ax.set_xlabel("Exact Match Accuracy")
    ax.set_ylabel("Consistent & Correct Rate")
    ax.set_title("Accuracy vs. Viewpoint Consistency")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(linestyle="--", alpha=0.35, zorder=0)

    ax.axvline(RANDOM_CHANCE, color="grey", linestyle="--", linewidth=1.2)

    # diagonal reference: consistent_correct ≤ exact_match always
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim], [0, lim], color="grey", linestyle=":", linewidth=1, label="y = x (upper bound)")

    legend_handles = [
        mlines.Line2D([], [], color=COLORS["open-source"], marker="o", linestyle="None", markersize=9, label="Open-source models"),
        mlines.Line2D([], [], color=COLORS["gpt"],    marker="s", linestyle="None", markersize=9, label="GPT (OpenAI)"),
        mlines.Line2D([], [], color=COLORS["gemini"], marker="^", linestyle="None", markersize=9, label="Gemini (Google)"),
        mlines.Line2D([], [], color="grey", linestyle=":", linewidth=1, label="y = x"),
        mlines.Line2D([], [], color="grey", linestyle="--", linewidth=1.2, label=f"Random chance ({RANDOM_CHANCE:.1%})"),
    ]
    ax.legend(handles=legend_handles, loc="upper left")

    fig.tight_layout()
    path = OUT_DIR / "fig2_accuracy_consistency.png"
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Performance by difficulty bin
# ══════════════════════════════════════════════════════════════════════════════
def fig3_difficulty_bins():
    DIFF_BINS   = ["aligned", "slight", "moderate", "strong", "extreme"]
    DIFF_LABELS = ["0–30°", "30–60°", "60–120°", "120–150°", "150–180°"]

    # pick representative models for a readable chart
    selected = [
        ("LLaVA-1.6 (7B)",       "llava_metrics.json",                                    "open-source", "o",  "--"),
        ("Qwen-VL (9.6B)",        "qwen_metrics.json",                                     "open-source", "v",  ":"),
        ("GPT-4o",                "metrics_chatgpt_4o_500viewpoints.json",                 "gpt",         "s",  "-"),
        ("GPT-5.4 (none)",        "metrics_chatgpt_5.4_nothinking_500viewpoints.json",     "gpt",         "^",  "-."),
        ("Gemini 2.5 Pro (8192)", "metrics_gemini_2.5pro_8192thinking_500viewpoints.json", "gemini",      "D",  "-"),
    ]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = np.arange(len(DIFF_BINS))

    for label, fname, group, marker, ls in selected:
        m = load(fname)
        values = [bin_exact_match(m, b) for b in DIFF_BINS]
        ax.plot(x, values, marker=marker, linestyle=ls,
                color=COLORS[group], linewidth=2, markersize=7, label=label)

    ax.axhline(RANDOM_CHANCE, color="grey", linestyle="--", linewidth=1.2,
               label=f"Random chance ({RANDOM_CHANCE:.1%})")
    ax.set_xticks(x)
    ax.set_xticklabels(DIFF_LABELS)
    ax.set_xlabel("Camera-to-Arrow Misalignment")
    ax.set_ylabel("Exact Match Accuracy")
    ax.set_title("Performance by Difficulty Bin")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(linestyle="--", alpha=0.35, zorder=0)
    ax.legend()

    fig.tight_layout()
    path = OUT_DIR / "fig3_difficulty_bins.png"
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Thinking budget vs. exact match
# ══════════════════════════════════════════════════════════════════════════════
def fig4_thinking_budget():
    # GPT-5.4: categorical thinking levels
    gpt_labels = ["none", "low", "medium"]
    gpt_files  = [
        "metrics_chatgpt_5.4_nothinking_500viewpoints.json",
        "metrics_chatgpt_5.4_lowthinking_500viewpoints.json",
        "metrics_chatgpt_5.4_medthinking_500viewpoints.json",
    ]
    gpt_values = [exact_match(load(f)) for f in gpt_files]

    # Gemini 2.5 Pro: explicit token budgets
    gem_budgets = [128, 2048, 8192, 16384]
    gem_files   = [
        "metrics_gemini_2.5pro_128thinking_500viewpoints.json",
        "metrics_gemini_2.5pro_2048thinking_500viewpoints.json",
        "metrics_gemini_2.5pro_8192thinking_500viewpoints.json",
        "metrics_gemini_2.5pro_16384thinking_500viewpoints.json",
    ]
    gem_values = [exact_match(load(f)) for f in gem_files]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE)
    fig.suptitle("Effect of Thinking Budget on Exact Match Accuracy", fontsize=21)

    # — GPT-5.4 subplot —
    x_gpt = np.arange(len(gpt_labels))
    ax1.plot(x_gpt, gpt_values, marker="s", color=COLORS["gpt"],
             linewidth=2.5, markersize=9, zorder=3)
    for xi, val in zip(x_gpt, gpt_values):
        ax1.annotate(f"{val:.1%}", (xi, val),
                     textcoords="offset points", xytext=(0, 10),
                     ha="center", fontsize=16)
    ax1.set_xticks(x_gpt)
    ax1.set_xticklabels(gpt_labels)
    ax1.set_xlabel("Reasoning Effort")
    ax1.set_ylabel("Exact Match Accuracy")
    ax1.set_title("GPT-5.4")
    ax1.axhline(RANDOM_CHANCE, color="grey", linestyle="--", linewidth=1.2,
                label=f"Random chance ({RANDOM_CHANCE:.1%})")
    ax1.set_ylim(0, max(gpt_values) * 1.25)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax1.grid(linestyle="--", alpha=0.35, zorder=0)
    ax1.legend(fontsize=16)

    # — Gemini subplot —
    ax2.plot(gem_budgets, gem_values, marker="^", color=COLORS["gemini"],
             linewidth=2.5, markersize=9, zorder=3)
    for bud, val in zip(gem_budgets, gem_values):
        ax2.annotate(f"{val:.1%}", (bud, val),
                     textcoords="offset points", xytext=(0, 10),
                     ha="center", fontsize=16)
    ax2.set_xscale("log")
    ax2.set_xticks(gem_budgets)
    ax2.set_xticklabels([str(b) for b in gem_budgets])
    ax2.set_xlabel("Thinking Token Budget")
    ax2.set_ylabel("Exact Match Accuracy")
    ax2.set_title("Gemini 2.5 Pro")
    ax2.axhline(RANDOM_CHANCE, color="grey", linestyle="--", linewidth=1.2)
    ax2.set_ylim(0, max(gem_values) * 1.25)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax2.grid(linestyle="--", alpha=0.35, zorder=0)

    fig.tight_layout()
    path = OUT_DIR / "fig4_thinking_budget.png"
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Per-axis accuracy (lateral / depth / vertical)
# ══════════════════════════════════════════════════════════════════════════════
def fig5_per_axis_accuracy():
    AXES        = ["lateral", "depth", "vertical"]
    AXIS_COLORS = ["#2ecc71", "#e74c3c", "#3498db"]

    labels = [label for label, _, _ in ALL_MODELS]
    n_models = len(labels)
    n_axes   = len(AXES)
    width    = 0.25
    x        = np.arange(n_models)

    fig, ax = plt.subplots(figsize=FIGSIZE)

    for i, (axis, color) in enumerate(zip(AXES, AXIS_COLORS)):
        values = [METRICS[label]["axis_metrics"][axis]["accuracy"] for label in labels]
        bars = ax.bar(x + (i - 1) * width, values, width,
                      label=axis.capitalize(), color=color,
                      edgecolor="white", linewidth=0.8, zorder=3)

    ax.axhline(1 / 3, color="grey", linestyle="--", linewidth=1.2,
               label=f"Random chance ({1/3:.1%})")

    ax.set_ylabel("Accuracy")
    ax.set_title("Per-Axis Accuracy — Lateral, Depth, Vertical")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.legend()

    fig.tight_layout()
    path = OUT_DIR / "fig5_per_axis_accuracy.png"
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fig1_model_comparison()
    fig2_accuracy_consistency()
    fig3_difficulty_bins()
    fig4_thinking_budget()
    fig5_per_axis_accuracy()
    print(f"\nAll figures saved to {OUT_DIR}/")
