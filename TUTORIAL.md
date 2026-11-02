# Tutorial · 08 · MLE, softmax, cross-entropy & KL

🇸🇰 [Slovenská verzia](TUTORIAL.sk.md) · 📄 [README (results)](README.md)

This tutorial walks through [`mle.py`](mle.py), five small NumPy experiments that together answer one question: **why is cross-entropy the right loss for a language model?** For each experiment: what it computes, why it is set up that way, and how the maths works.

**You need:** projects 02 (derivatives), 04 (makemore used NLL) and 06 (probability). Only NumPy; every claim is checked numerically with seed 0.

---

## Step 1 · Maximum likelihood on a coin (`coin_mle`)

```python
p = np.linspace(0.001, 0.999, 999)
loglik = heads * np.log(p) + (n - heads) * np.log(1 - p)
argmax = p[np.argmax(loglik)]          # 0.700
```

**The idea:** you saw 7 heads in 10 flips. For each possible coin `p = P(heads)`, the probability of exactly this data is the **likelihood** `L(p) = p⁷(1−p)³`. **Maximum likelihood estimation (MLE)** picks the `p` that makes the observed data most probable.

**Why take the log:**
- The likelihood of a big dataset is a product of thousands of numbers below 1, which underflows to 0 in floating point. The log turns the product into a sum: `7 log p + 3 log(1−p)`.
- `log` is increasing, so it does not move the maximum.

**How the answer appears:** set the derivative to zero: `7/p − 3/(1−p) = 0` → `p = 7/10`. The grid search confirms 0.700. **MLE for counts is just the frequency.**

**Why the second curve (70 of 100):** same peak, but much sharper. Moving to `p = 0.6` costs 0.216 in log-likelihood with 10 flips and 2.160 with 100: 10× more data, 10× more evidence against wrong values.

---

## Step 2 · Softmax: from scores to probabilities (`softmax`)

```python
def softmax(z, t=1.0):
    z = np.asarray(z, float) / t
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)
```

**What it does:** a network outputs arbitrary real numbers (**logits**). `exp` makes them positive and keeps their order; dividing by the sum makes them add up to 1. Now they are a probability distribution.

**Why subtract the max:** `softmax(z + c) = softmax(z)` because `e^c` cancels in the ratio. So subtracting the max changes nothing mathematically, but it keeps `exp` from overflowing. The script proves it: with logits + 1000, the naive version returns `[NaN, NaN, NaN, NaN]` (`exp(1002) = inf`, `inf/inf = NaN`), the stable version returns the correct answer.

**Temperature `t`:** divide logits before softmax. For logits [2, 1, 0.1, −1]: T = 0.25 → top class 98.2 %, T = 1 → 63.8 %, T = 5 → 32.8 %. Low T sharpens, high T flattens. This is the sampling knob from project 04 and 12.

---

## Step 3 · Cross-entropy is maximum likelihood (`ce_is_mle`)

### 3a · Set up data and a model

```python
true_p = np.array([0.35, 0.25, 0.15, 0.12, 0.08, 0.05])
x = rng.choice(6, 500, p=true_p)          # 500 samples
emp = np.bincount(x, minlength=6) / 500    # empirical frequencies p̂
z = np.zeros(6)                            # model logits; q = softmax(z) starts uniform
```

The model is as simple as possible: 6 free logits, one per class. That makes the maths visible without a network in the way.

### 3b · Why NLL and cross-entropy are the same number

```
NLL(q) = −(1/n) Σᵢ log q(xᵢ)          average over the 500 samples
       = −Σₖ (countₖ/n) log q(k)       group equal samples by class
       = −Σₖ p̂(k) log q(k)             = H(p̂, q), cross-entropy(data, model)
```

So "maximise the likelihood of the data" = "minimise the average negative log-likelihood" = "minimise cross-entropy between the data frequencies and the model". The script checks the identity on 100 random models: the largest difference is 8.9e-16 (float rounding).

### 3c · Train with gradient descent

```python
for s in range(steps + 1):
    q = softmax(z)
    nll = -np.log(q[x]).mean()
    z -= lr * (q - emp)                   # gradient from Step 4
```

After 300 steps `q` matches `p̂` to 4.7e-4. **Gradient descent on cross-entropy finds the MLE**, i.e. the frequencies from Step 1.

**Note what it learned:** q(class 0) = 0.296, the *sample* frequency, not the true 0.35. MLE fits the data you have. That is why more data and regularisation matter (project 04's overfitting).

### 3d · Why the loss never reaches 0

```
H(p̂, q) = H(p̂) + KL(p̂ ‖ q)
```

- `H(p̂)` = entropy of the data: fixed, the model can't change it.
- `KL(p̂ ‖ q) ≥ 0`, and 0 only when q = p̂.

So the loss falls from `log 6 = 1.792` (uniform) to **1.652 = H(p̂)**, and KL goes 0.140 → 0.001. **The irreducible floor of a language model's loss is the entropy of language itself.**

**Connection to project 04:** `exp(NLL)` is the **perplexity**, "how many equally likely choices the model is torn between". Bigram 2.462 → perplexity 11.7; MLP 1.952 → 7.0.

---

## Step 4 · The gradient is "prediction minus target" (`gradient_check`)

```python
analytic = softmax(z) - np.eye(classes)[y]
numeric  = [(loss(z + h*e) - loss(z - h*e)) / (2*h) for e in np.eye(classes)]
```

For one example with true class `y`, `loss = −log softmax(z)[y]`. Working out the derivative gives

```
∂loss/∂zₖ = softmax(z)ₖ − [k = y]
```

**Why it is checked numerically:** same technique as project 02, central differences. Max difference 1.75e-10.

**What follows:**
- The gradient is simply predicted probability minus one-hot target. That is the update rule in Step 3c, and what PyTorch's `F.cross_entropy` computes internally.
- The entries sum to 0: pushing the correct class up always pushes the others down.
- It never explodes: every entry is between −1 and 1. That is one reason softmax + cross-entropy is the standard pairing.

---

## Step 5 · KL is not symmetric (`kl_asymmetry`)

```python
p = 0.5 * normal_pdf(x, -2.5, 0.8) + 0.5 * normal_pdf(x, 2.5, 0.8)   # two peaks
for mu in ...: for s in ...:
    q = normal_pdf(x, mu, s)
    forward = KL(p ‖ q); reverse = KL(q ‖ p)
```

**Setup:** the truth `p` has two peaks; the model `q` is a single Gaussian, so it *cannot* fit perfectly. A grid search finds the best `(μ, σ)` for each direction of KL.

**How the two directions behave:**
- **Forward KL(p‖q) = Σ p log(p/q):** heavily punished where `p` has mass but `q ≈ 0` (log of a huge ratio). So `q` spreads to cover both peaks: μ = 0, σ = 2.65. **Mode-covering.** This is the direction MLE / cross-entropy minimises (Step 3).
- **Reverse KL(q‖p) = Σ q log(q/p):** punished where `q` has mass but `p ≈ 0`, e.g. the empty valley between the peaks. So `q` hides inside one peak: μ = 2.5, σ = 0.8. **Mode-seeking.**

**Why it matters later:** reverse-style KL appears in the KL penalty of RLHF/DPO (project 13) and in variational inference.

**Sanity check in the code:** the forward-KL optimum should match the mixture's mean (0) and standard deviation (`√(0.8² + 2.5²) = 2.625`). The grid finds 2.65, the nearest grid point (step 0.05).

---

## Step 6 · Save and visualise

`main()` prints each check and writes `results.json`; [`plot.py`](plot.py) draws the coin likelihood, softmax temperatures, CE = H + KL, the KL fits and `ce_training.gif` (the model's bars sliding into the data frequencies while KL melts to zero).

```bash
uv run python mle.py
uv run python plot.py
```

**Why no Racing Lab chapter:** the cars learn from rewards and imitation; a distribution you can watch move shows "CE = H + KL" more directly than any car could.

---

## Try it yourself

1. In `ce_is_mle`, use `n = 50` and `n = 5000`. How close does the fitted `q` get to the *true* `p` in each case?
2. Add label smoothing: train towards `0.9·p̂ + 0.1/6` instead of `p̂`. What is the new floor of the loss?
3. Give the two peaks in Step 5 unequal weights (0.8 / 0.2). Which peak does reverse KL pick?

## Recap

| Step | Idea |
|---|---|
| 1 | MLE = parameters that make the data most probable; for counts, the frequency |
| 2 | Softmax turns logits into probabilities; subtract the max; temperature sharpens/flattens |
| 3 | Average NLL = cross-entropy(data, model); minimising it is MLE |
| 3d | CE = H + KL: the loss floor is the data's entropy |
| 4 | ∂CE/∂logits = softmax − one-hot |
| 5 | Forward KL covers modes (MLE), reverse KL picks one (RLHF penalty) |
