# Tutoriál · 08 · MLE, softmax, cross-entropy a KL

🇬🇧 [English version](TUTORIAL.md) · 📄 [README (výsledky)](README.sk.md)

Tento tutoriál prejde [`mle.py`](mle.py), päť malých NumPy experimentov, ktoré spolu odpovedajú na jednu otázku: **prečo je cross-entropy správny loss pre jazykový model?** Pri každom experimente: čo počíta, prečo je postavený takto a ako funguje matematika.

**Potrebuješ:** projekty 02 (derivácie), 04 (makemore používal NLL) a 06 (pravdepodobnosť). Len NumPy; každé tvrdenie sa overí numericky so seedom 0.

---

## Krok 1 · Maximum likelihood na minci (`coin_mle`)

```python
p = np.linspace(0.001, 0.999, 999)
loglik = heads * np.log(p) + (n - heads) * np.log(1 - p)
argmax = p[np.argmax(loglik)]          # 0,700
```

**Myšlienka:** videl si 7 hláv v 10 hodoch. Pre každú možnú mincu `p = P(hlava)` je pravdepodobnosť presne týchto dát **vierohodnosť** (likelihood) `L(p) = p⁷(1−p)³`. **Odhad maximálnej vierohodnosti (MLE)** vyberie `p`, pri ktorom sú pozorované dáta najpravdepodobnejšie.

**Prečo logaritmus:**
- Vierohodnosť veľkého datasetu je súčin tisícov čísel menších než 1, ktorý vo floatoch podtečie na 0. Logaritmus premení súčin na súčet: `7 log p + 3 log(1−p)`.
- `log` je rastúci, takže maximum neposunie.

**Ako vyjde odpoveď:** derivácia rovná nule: `7/p − 3/(1−p) = 0` → `p = 7/10`. Grid search potvrdí 0,700. **MLE pre počty je jednoducho relatívna početnosť.**

**Prečo druhá krivka (70 zo 100):** rovnaký vrchol, ale oveľa ostrejší. Posun na `p = 0,6` stojí 0,216 log-vierohodnosti pri 10 hodoch a 2,160 pri 100: 10× viac dát, 10× viac dôkazov proti zlým hodnotám.

---

## Krok 2 · Softmax: zo skóre pravdepodobnosti (`softmax`)

```python
def softmax(z, t=1.0):
    z = np.asarray(z, float) / t
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)
```

**Čo robí:** sieť vracia ľubovoľné reálne čísla (**logity**). `exp` ich spraví kladnými a zachová poradie; delenie súčtom zabezpečí, že dajú spolu 1. Teraz sú rozdelením pravdepodobnosti.

**Prečo odčítať maximum:** `softmax(z + c) = softmax(z)`, lebo `e^c` sa v podiele vykráti. Odčítanie maxima teda matematicky nič nezmení, ale zabráni pretečeniu `exp`. Skript to dokáže: s logitmi + 1000 naivná verzia vráti `[NaN, NaN, NaN, NaN]` (`exp(1002) = inf`, `inf/inf = NaN`), stabilná vráti správnu odpoveď.

**Teplota `t`:** vydeľ logity pred softmaxom. Pre logity [2, 1, 0,1, −1]: T = 0,25 → najvyššia trieda 98,2 %, T = 1 → 63,8 %, T = 5 → 32,8 %. Nízke T zaostrí, vysoké sploští. Toto je gombík na vzorkovanie z projektov 04 a 12.

---

## Krok 3 · Cross-entropy je maximum likelihood (`ce_is_mle`)

### 3a · Priprav dáta a model

```python
true_p = np.array([0.35, 0.25, 0.15, 0.12, 0.08, 0.05])
x = rng.choice(6, 500, p=true_p)          # 500 vzoriek
emp = np.bincount(x, minlength=6) / 500    # empirické početnosti p̂
z = np.zeros(6)                            # logity modelu; q = softmax(z) začína rovnomerne
```

Model je čo najjednoduchší: 6 voľných logitov, jeden na triedu. Matematika je tak viditeľná bez siete, ktorá by zavadzala.

### 3b · Prečo sú NLL a cross-entropy to isté číslo

```
NLL(q) = −(1/n) Σᵢ log q(xᵢ)          priemer cez 500 vzoriek
       = −Σₖ (countₖ/n) log q(k)       zoskup rovnaké vzorky podľa triedy
       = −Σₖ p̂(k) log q(k)             = H(p̂, q), cross-entropy(dáta, model)
```

Takže „maximalizuj vierohodnosť dát“ = „minimalizuj priemernú negatívnu log-vierohodnosť“ = „minimalizuj cross-entropy medzi početnosťami v dátach a modelom“. Skript overí túto rovnosť na 100 náhodných modeloch: najväčší rozdiel je 8,9e-16 (zaokrúhlenie floatov).

### 3c · Trénuj gradient descentom

```python
for s in range(steps + 1):
    q = softmax(z)
    nll = -np.log(q[x]).mean()
    z -= lr * (q - emp)                   # gradient z Kroku 4
```

Po 300 krokoch sa `q` zhoduje s `p̂` na 4,7e-4. **Gradient descent na cross-entropy nájde MLE**, teda početnosti z Kroku 1.

**Všimni si, čo sa naučil:** q(trieda 0) = 0,296, *výberová* početnosť, nie skutočných 0,35. MLE fituje dáta, ktoré máš. Preto záleží na množstve dát a regularizácii (preučenie v projekte 04).

### 3d · Prečo loss nikdy nedosiahne 0

```
H(p̂, q) = H(p̂) + KL(p̂ ‖ q)
```

- `H(p̂)` = entropia dát: pevná, model ju nezmení.
- `KL(p̂ ‖ q) ≥ 0` a je 0, len keď q = p̂.

Loss teda klesne z `log 6 = 1,792` (rovnomerné) na **1,652 = H(p̂)** a KL ide 0,140 → 0,001. **Nepreskočiteľné dno lossu jazykového modelu je entropia samotného jazyka.**

**Súvis s projektom 04:** `exp(NLL)` je **perplexita**, „medzi koľkými rovnako pravdepodobnými možnosťami model váha“. Bigram 2,462 → perplexita 11,7; MLP 1,952 → 7,0.

---

## Krok 4 · Gradient je „predikcia mínus cieľ“ (`gradient_check`)

```python
analytic = softmax(z) - np.eye(classes)[y]
numeric  = [(loss(z + h*e) - loss(z - h*e)) / (2*h) for e in np.eye(classes)]
```

Pre jeden príklad so skutočnou triedou `y` je `loss = −log softmax(z)[y]`. Po zderivovaní vyjde

```
∂loss/∂zₖ = softmax(z)ₖ − [k = y]
```

**Prečo sa overuje numericky:** rovnaká technika ako v projekte 02, centrálne diferencie. Max rozdiel 1,75e-10.

**Čo z toho plynie:**
- Gradient je jednoducho predpovedaná pravdepodobnosť mínus one-hot cieľ. To je pravidlo aktualizácie z Kroku 3c a to, čo interne počíta `F.cross_entropy` v PyTorchi.
- Zložky dajú spolu 0: keď tlačíš správnu triedu hore, ostatné vždy idú dole.
- Nikdy nevybuchne: každá zložka je medzi −1 a 1. To je jeden z dôvodov, prečo je softmax + cross-entropy štandardná dvojica.

---

## Krok 5 · KL nie je symetrická (`kl_asymmetry`)

```python
p = 0.5 * normal_pdf(x, -2.5, 0.8) + 0.5 * normal_pdf(x, 2.5, 0.8)   # dva vrcholy
for mu in ...: for s in ...:
    q = normal_pdf(x, mu, s)
    forward = KL(p ‖ q); reverse = KL(q ‖ p)
```

**Nastavenie:** skutočné `p` má dva vrcholy; model `q` je jedna Gaussovka, takže sa *nemôže* trafiť dokonale. Grid search nájde najlepšie `(μ, σ)` pre každý smer KL.

**Ako sa oba smery správajú:**
- **Dopredná KL(p‖q) = Σ p log(p/q):** veľmi trestá miesta, kde `p` má hmotu, ale `q ≈ 0` (log obrovského podielu). Preto sa `q` roztiahne a pokryje oba vrcholy: μ = 0, σ = 2,65. **Pokrýva módy (mode-covering).** Tento smer minimalizuje MLE / cross-entropy (Krok 3).
- **Spätná KL(q‖p) = Σ q log(q/p):** trestá miesta, kde `q` má hmotu, ale `p ≈ 0`, napr. prázdne údolie medzi vrcholmi. Preto sa `q` schová do jedného vrcholu: μ = 2,5, σ = 0,8. **Hľadá jeden mód (mode-seeking).**

**Prečo na tom neskôr záleží:** KL v tomto „spätnom“ duchu sa objaví v KL penalizácii pri RLHF/DPO (projekt 13) a vo variačnej inferencii.

**Kontrola v kóde:** optimum doprednej KL by malo sedieť so strednou hodnotou zmesi (0) a jej smerodajnou odchýlkou (`√(0,8² + 2,5²) = 2,625`). Grid nájde 2,65, najbližší bod mriežky (krok 0,05).

---

## Krok 6 · Ulož a vizualizuj

`main()` vypíše každú kontrolu a zapíše `results.json`; [`plot.py`](plot.py) nakreslí vierohodnosť mince, teploty softmaxu, CE = H + KL, KL fity a `ce_training.gif` (stĺpce modelu sa zosúvajú do početností dát, kým KL sa roztopí na nulu).

```bash
uv run python mle.py
uv run python plot.py
```

**Prečo nie kapitola v Racing Lab:** autá sa učia z odmien a imitácie; rozdelenie, ktoré vidíš hýbať sa, ukáže „CE = H + KL“ priamejšie, než by to vedeli autíčka.

---

## Vyskúšaj sám

1. V `ce_is_mle` použi `n = 50` a `n = 5000`. Ako blízko sa nafitované `q` dostane ku *skutočnému* `p` v každom prípade?
2. Pridaj label smoothing: trénuj na `0,9·p̂ + 0,1/6` namiesto `p̂`. Aké je nové dno lossu?
3. Daj dvom vrcholom v Kroku 5 nerovnaké váhy (0,8 / 0,2). Ktorý vrchol si vyberie spätná KL?

## Zhrnutie

| Krok | Myšlienka |
|---|---|
| 1 | MLE = parametre, pri ktorých sú dáta najpravdepodobnejšie; pre počty je to početnosť |
| 2 | Softmax spraví z logitov pravdepodobnosti; odčítaj max; teplota zaostrí/sploští |
| 3 | Priemerné NLL = cross-entropy(dáta, model); jej minimalizácia je MLE |
| 3d | CE = H + KL: dno lossu je entropia dát |
| 4 | ∂CE/∂logity = softmax − one-hot |
| 5 | Dopredná KL pokrýva módy (MLE), spätná si vyberie jeden (penalizácia pri RLHF) |
