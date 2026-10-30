"""Turn results.json into the plots and the GIF used in the README."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

R = json.loads(Path("results.json").read_text())
Path("assets").mkdir(exist_ok=True)
BLUE, RED, GREEN, GRAY, ORANGE = "#2563eb", "#dc2626", "#16a34a", "#94a3b8", "#ea580c"
TEMP_COLORS = ["#1e3a8a", "#2563eb", "#60a5fa", "#93c5fd", "#cbd5e1"]


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(f"assets/{name}", dpi=150)
    plt.close(fig)


def coin() -> None:
    c = R["coin"]
    p = np.array(c["p"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.3))
    a1.plot(p, c["likelihood"], color=BLUE, lw=2)
    a1.axvline(c["argmax_formula"], color=RED, ls="--", lw=1, label=f"maximum at p = k/n = {c['argmax_formula']}")
    a1.set(
        xlabel="p = P(heads)", ylabel="L(p) = p⁷ (1−p)³", title=f"Likelihood of {c['heads']} heads in {c['n']} flips"
    )
    a1.legend(fontsize=9)
    a1.grid(alpha=0.3)
    for key, lab, col in [("loglik", f"n = {c['n']}", BLUE), ("loglik_big", f"n = {c['big']['n']}", ORANGE)]:
        ll = np.array(c[key])
        a2.plot(p, ll - ll.max(), color=col, lw=2, label=f"{lab} (70 % heads)")
    a2.axvline(0.6, color=GRAY, ls=":", lw=1)
    a2.set(
        xlabel="p",
        ylabel="log L(p) − max",
        ylim=(-12, 0.5),
        title="Log-likelihood: same peak, more data → sharper peak",
    )
    a2.legend(fontsize=9)
    a2.grid(alpha=0.3)
    save(fig, "coin.png")


def softmax_plot() -> None:
    s = R["softmax"]
    fig, a = plt.subplots(figsize=(8, 4))
    k = len(s["logits"])
    x = np.arange(k)
    temps = list(s["temperatures"])
    w = 0.8 / len(temps)
    for i, (t, col) in enumerate(zip(temps, TEMP_COLORS)):
        a.bar(x + (i - len(temps) / 2 + 0.5) * w, s["temperatures"][t], w, color=col, label=f"T = {t}")
    a.set_xticks(x, [f"logit {v:g}" for v in s["logits"]])
    a.set(ylabel="probability", title="softmax(z / T): low T → sharp (argmax), high T → flat (uniform)")
    a.legend(fontsize=8, ncol=5, loc="upper right")
    a.grid(alpha=0.3, axis="y")
    save(fig, "softmax.png")


def ce_mle() -> None:
    m = R["ce_is_mle"]
    h = m["history"]
    steps = [r["step"] for r in h]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.4))
    a1.plot(steps, [r["nll"] for r in h], color=BLUE, lw=2, label="NLL of the data = CE(empirical, q)")
    a1.plot(steps, [r["kl"] for r in h], color=ORANGE, lw=2, label="KL(empirical ‖ q)")
    a1.axhline(
        m["entropy_empirical"], color=GREEN, ls="--", lw=1.2, label=f"H(empirical) = {m['entropy_empirical']:.3f}"
    )
    a1.set(
        xlabel="gradient-descent step",
        ylabel="nats",
        xlim=(0, None),
        title="CE = H + KL: only KL can shrink, and it goes to 0",
    )
    a1.legend(fontsize=8)
    a1.grid(alpha=0.3)
    k = len(m["true_p"])
    x = np.arange(k)
    a2.bar(x - 0.27, m["true_p"], 0.26, color=GRAY, label="true p (unknown to the model)")
    a2.bar(x, m["empirical"], 0.26, color=GREEN, label=f"frequencies in {m['n']} samples = MLE")
    a2.bar(x + 0.27, m["fitted"], 0.26, color=BLUE, label="q after minimising cross-entropy")
    a2.set_xticks(x, [f"class {i}" for i in x])
    a2.set(ylabel="probability", title="Minimising cross-entropy lands exactly on the MLE")
    a2.legend(fontsize=8)
    a2.grid(alpha=0.3, axis="y")
    save(fig, "ce_mle.png")


def kl_plot() -> None:
    k = R["kl"]
    x = np.array(k["x"])
    fig, a = plt.subplots(figsize=(9, 4.2))
    a.fill_between(x, k["p"], color=GRAY, alpha=0.5, label="p: two-peak data distribution")
    f, r = k["forward"], k["reverse"]
    a.plot(x, f["q"], color=BLUE, lw=2, label=f"min KL(p‖q): μ={f['mu']}, σ={f['sigma']} → covers both peaks")
    a.plot(x, r["q"], color=RED, lw=2, label=f"min KL(q‖p): μ={r['mu']}, σ={r['sigma']} → picks one peak")
    a.set(xlabel="x", ylabel="density", title="KL is not symmetric: fitting one Gaussian to two peaks")
    a.legend(fontsize=8, loc="upper left")
    a.grid(alpha=0.3)
    save(fig, "kl.png")


def gif() -> None:
    m = R["ce_is_mle"]
    h = [r for r in m["history"] if r["step"] <= 120]  # most of the movement happens early
    k = len(m["true_p"])
    x = np.arange(k)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1.6, 1]})
    a1.bar(x, m["empirical"], 0.7, color="none", edgecolor=GREEN, lw=2, label="data frequencies (MLE)")
    bars = a1.bar(x, h[0]["q"], 0.5, color=BLUE, alpha=0.8, label="model q = softmax(logits)")
    a1.set(ylim=(0, 0.45), xticks=x, xlabel="class", ylabel="probability")
    a1.legend(fontsize=8, loc="upper right")
    title = a1.set_title("")
    hbar = a2.bar([0], [m["entropy_empirical"]], 0.5, color=GREEN, label="H(data): fixed")
    kbar = a2.bar([0], [h[0]["kl"]], 0.5, bottom=[m["entropy_empirical"]], color=ORANGE, label="KL(data ‖ q)")
    a2.set(
        xlim=(-0.8, 0.8),
        ylim=(1.6, 1.82),
        xticks=[],
        ylabel="nats (axis starts at 1.6)",
        title="cross-entropy = H + KL",
    )
    a2.legend(fontsize=8, loc="center right")
    txt = a2.text(0, 1.805, "", ha="center", fontsize=10)
    del hbar

    def frame(i):
        r = h[min(i, len(h) - 1)]
        for b, v in zip(bars, r["q"]):
            b.set_height(v)
        kbar[0].set_height(r["kl"])
        title.set_text(f"gradient descent on cross-entropy, step {r['step']}")
        txt.set_text(f"CE {r['ce']:.3f} = {m['entropy_empirical']:.3f} + {r['kl']:.3f}")
        return [*bars, kbar[0], title, txt]

    fig.tight_layout()
    FuncAnimation(fig, frame, frames=len(h) + 10).save("assets/ce_training.gif", writer=PillowWriter(fps=6), dpi=80)


if __name__ == "__main__":
    coin()
    softmax_plot()
    ce_mle()
    kl_plot()
    gif()
