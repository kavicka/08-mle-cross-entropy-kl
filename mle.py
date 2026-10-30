"""Step 08 · MLE, softmax, cross-entropy and KL: why minimising cross-entropy = maximum likelihood.

Five small experiments, NumPy only, fixed seed (every number in the README comes from results.json):
1. MLE for a coin: the log-likelihood of 7 heads in 10 flips peaks at p = 0.7 = k / n.
2. Softmax: logits → probabilities, temperature, and why you subtract the max (overflow).
3. Cross-entropy = MLE: fit a 6-class distribution to samples by gradient descent on cross-entropy.
   It lands on the sample frequencies (the MLE), and CE = H(p) + KL(p‖q) at every step.
4. The gradient of cross-entropy w.r.t. the logits is softmax − one-hot (checked numerically).
5. KL is not symmetric: fit one Gaussian to a two-peak mixture with forward vs reverse KL.

Run:  uv run python mle.py   → results.json
      uv run python plot.py  → assets/*.png, assets/ce_training.gif
"""

import json
from pathlib import Path

import numpy as np

rng = np.random.default_rng(0)


# --- 1. MLE for a coin -----------------------------------------------------------------
def coin_mle(heads: int = 7, n: int = 10) -> dict:
    """Likelihood L(p) = p^k (1−p)^(n−k). Its log turns the product into a sum, same maximum."""
    p = np.linspace(0.001, 0.999, 999)
    loglik = heads * np.log(p) + (n - heads) * np.log(1 - p)
    big = {"heads": heads * 10, "n": n * 10}  # same 70 %, 10× more data: a sharper peak
    loglik_big = big["heads"] * np.log(p) + (big["n"] - big["heads"]) * np.log(1 - p)
    return {
        "heads": heads,
        "n": n,
        "p": p[::10].tolist(),
        "likelihood": np.exp(loglik)[::10].tolist(),
        "loglik": loglik[::10].tolist(),
        "loglik_big": loglik_big[::10].tolist(),
        "big": big,
        "argmax_numeric": float(p[np.argmax(loglik)]),
        "argmax_formula": heads / n,
        # how far the log-likelihood falls 0.1 away from the peak: data size sharpens it
        "drop_at_0.6": float(loglik.max() - loglik[np.argmin(np.abs(p - 0.6))]),
        "drop_at_0.6_big": float(loglik_big.max() - loglik_big[np.argmin(np.abs(p - 0.6))]),
    }


# --- 2. Softmax ----------------------------------------------------------------------------
def softmax(z: np.ndarray, t: float = 1.0) -> np.ndarray:
    """Subtracting the max changes nothing mathematically (it cancels) but keeps exp() from overflowing."""
    z = np.asarray(z, float) / t
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def softmax_naive(z: np.ndarray) -> np.ndarray:
    e = np.exp(np.asarray(z, float))
    return e / e.sum()


def softmax_demo() -> dict:
    logits = np.array([2.0, 1.0, 0.1, -1.0])
    with np.errstate(over="ignore", invalid="ignore"):
        naive_big = softmax_naive(logits + 1000)
    return {
        "logits": logits.tolist(),
        "temperatures": {str(t): softmax(logits, t).tolist() for t in (0.25, 0.5, 1.0, 2.0, 5.0)},
        "shift_invariant": bool(np.allclose(softmax(logits), softmax(logits + 1000))),
        "naive_plus_1000": [None if np.isnan(v) else float(v) for v in naive_big],
        "stable_plus_1000": softmax(logits + 1000).tolist(),
    }


# --- 3. Cross-entropy = MLE ------------------------------------------------------------------
def entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def cross_entropy(p: np.ndarray, q: np.ndarray) -> float:
    m = p > 0
    return float(-(p[m] * np.log(q[m])).sum())


def kl(p: np.ndarray, q: np.ndarray) -> float:
    m = p > 0
    return float((p[m] * np.log(p[m] / q[m])).sum())


def ce_is_mle(true_p=(0.35, 0.25, 0.15, 0.12, 0.08, 0.05), n: int = 500, steps: int = 300, lr: float = 0.2) -> dict:
    true_p = np.array(true_p)
    x = rng.choice(len(true_p), n, p=true_p)
    counts = np.bincount(x, minlength=len(true_p))
    emp = counts / n  # the MLE of a categorical distribution: just the frequencies
    z = np.zeros(len(true_p))  # logits, start from uniform
    hist = []
    for s in range(steps + 1):
        q = softmax(z)
        nll = float(-np.log(q[x]).mean())  # average negative log-likelihood of the data
        hist.append({"step": s, "q": q.tolist(), "nll": nll, "ce": cross_entropy(emp, q), "kl": kl(emp, q)})
        z -= lr * (q - emp)  # gradient of the mean NLL w.r.t. the logits (part 4)
    q = softmax(z)
    # NLL of the data equals cross-entropy(empirical, q) for ANY q, not just at the end: check on random q's
    gaps = []
    for _ in range(100):
        r = softmax(rng.normal(0, 2, len(true_p)))
        gaps.append(abs(-np.log(r[x]).mean() - cross_entropy(emp, r)))
    return {
        "true_p": true_p.tolist(),
        "n": n,
        "counts": counts.tolist(),
        "empirical": emp.tolist(),
        "fitted": q.tolist(),
        "max_abs_fitted_minus_empirical": float(np.abs(q - emp).max()),
        "entropy_empirical": entropy(emp),
        "entropy_true": entropy(true_p),
        "final_nll": float(-np.log(q[x]).mean()),
        "nll_uniform": float(np.log(len(true_p))),
        "max_abs_nll_minus_ce_random_q": float(max(gaps)),
        "history": hist[::3] + [hist[-1]],
        "lr": lr,
        "steps": steps,
    }


# --- 4. Gradient of cross-entropy --------------------------------------------------------------
def gradient_check(classes: int = 5) -> dict:
    """loss(z) = −log softmax(z)[y]. Analytic gradient: softmax(z) − onehot(y)."""
    z, y = rng.normal(0, 1, classes), 2

    def loss(v):
        return -np.log(softmax(v)[y])

    analytic = softmax(z) - np.eye(classes)[y]
    h = 1e-6
    numeric = np.array([(loss(z + h * e) - loss(z - h * e)) / (2 * h) for e in np.eye(classes)])
    return {
        "logits": z.tolist(),
        "target": y,
        "analytic": analytic.tolist(),
        "numeric": numeric.tolist(),
        "max_abs_diff": float(np.abs(analytic - numeric).max()),
        "sum_of_gradient": float(analytic.sum()),
    }


# --- 5. Forward vs reverse KL ----------------------------------------------------------------------
def normal_pdf(x, mu, s):
    return np.exp(-0.5 * ((x - mu) / s) ** 2) / (s * np.sqrt(2 * np.pi))


def kl_asymmetry() -> dict:
    """p = two-peak mixture, q = one Gaussian. Grid search over (mu, sigma) for each direction of KL."""
    x = np.linspace(-8, 8, 4001)
    dx = x[1] - x[0]
    p = 0.5 * normal_pdf(x, -2.5, 0.8) + 0.5 * normal_pdf(x, 2.5, 0.8)
    eps = 1e-300

    def kl_grid(a, b):
        return float(np.sum(a * (np.log(a + eps) - np.log(b + eps))) * dx)

    best = {"forward": (np.inf, None), "reverse": (np.inf, None)}
    for mu in np.arange(-4, 4.001, 0.05):
        for s in np.arange(0.3, 4.001, 0.05):
            q = normal_pdf(x, mu, s)
            f, r = kl_grid(p, q), kl_grid(q, p)
            if f < best["forward"][0]:
                best["forward"] = (f, (mu, s))
            if r < best["reverse"][0]:
                best["reverse"] = (r, (mu, s))
    # sanity: forward-KL optimum = moment matching (mean 0, std of the mixture)
    mix_std = float(np.sqrt(0.8**2 + 2.5**2))
    step = 20
    out = {"x": x[::step].tolist(), "p": p[::step].tolist(), "mixture_std": mix_std}
    for k, (val, (mu, s)) in best.items():
        q = normal_pdf(x, mu, s)
        out[k] = {
            "kl": val,
            "mu": round(float(mu), 2) + 0.0,
            "sigma": round(float(s), 2),
            "q": q[::step].tolist(),
            "other_direction": kl_grid(q, p) if k == "forward" else kl_grid(p, q),
        }
    return out


def main() -> None:
    res = {
        "seed": 0,
        "coin": coin_mle(),
        "softmax": softmax_demo(),
        "ce_is_mle": ce_is_mle(),
        "gradient": gradient_check(),
        "kl": kl_asymmetry(),
    }
    c, s, m, g, k = res["coin"], res["softmax"], res["ce_is_mle"], res["gradient"], res["kl"]
    print(
        f"coin: argmax {c['argmax_numeric']:.3f} (k/n = {c['argmax_formula']})  log-lik drop at p=0.6: "
        f"{c['drop_at_0.6']:.3f} (n=10) vs {c['drop_at_0.6_big']:.3f} (n=100)"
    )
    print(f"softmax shift-invariant {s['shift_invariant']}  naive(+1000) {s['naive_plus_1000']}")
    print(
        f"CE=MLE: fitted vs empirical max diff {m['max_abs_fitted_minus_empirical']:.2e}  final NLL "
        f"{m['final_nll']:.4f} = H(emp) {m['entropy_empirical']:.4f}; uniform {m['nll_uniform']:.4f}; "
        f"|NLL − CE| on random q ≤ {m['max_abs_nll_minus_ce_random_q']:.1e}"
    )
    print(f"gradient: max |analytic − numeric| {g['max_abs_diff']:.2e}, sum {g['sum_of_gradient']:.1e}")
    for d in ("forward", "reverse"):
        print(
            f"KL {d}: mu {k[d]['mu']} sigma {k[d]['sigma']} KL {k[d]['kl']:.3f} other dir {k[d]['other_direction']:.3f}"
        )
    print(f"mixture std {k['mixture_std']:.3f}")
    Path("results.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
