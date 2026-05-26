# LDP Poison Toolkit

A research toolkit for evaluating **poisoning attacks on Local Differential Privacy (LDP) protocols** for ranking estimation. The toolkit implements three LDP protocols and four attack strategies, providing a clean, modular API suitable for security research and reproducible experiments.

> **Disclaimer:** This toolkit is intended for security research and educational purposes only. All attack implementations are designed to help understand and defend against adversarial threats to LDP systems.

---

## Overview

Local Differential Privacy (LDP) allows users to share data with a server while protecting individual privacy — each user perturbs their own value before submission. However, LDP systems are vulnerable to **poisoning attacks** where adversarial fake users inject crafted reports to manipulate the server's aggregate estimates.

This toolkit focuses on **ranking estimation attacks**: the adversary controls `n2` fake users and aims to maximize the rank improvement of a set of **target items** `T` in the server's estimated frequency ranking.

### Supported LDP Protocols

| Protocol | Description | Report Format |
|----------|-------------|---------------|
| **GRR** | Generalized Random Response | Single integer in `[0, d-1]` |
| **OUE** | Optimized Unary Encoding | Binary vector of length `d` |
| **OLH** | Optimized Local Hashing | Tuple `(hashed_value, seed)` |

### Supported Attack Strategies

| Attack | Protocols | Description |
|--------|-----------|-------------|
| **Random** | GRR, OUE, OLH | Each fake user reports a randomly chosen target item |
| **ROA** | OUE, OLH | Random Output Attack — noise baseline injecting non-target support |
| **Greedy** | GRR, OUE, OLH | Iteratively promotes the most cost-effective non-target item |
| **MPOIA** | GRR | Greedy with statistical confidence margins for noisy environments |

---

## Installation

```bash
git clone https://github.com/your-org/ldp-poison-toolkit.git
cd ldp-poison-toolkit
pip install -e .
```

### Requirements

- Python ≥ 3.8
- numpy, pandas, scipy, xxhash, scikit-learn, matplotlib

---

## Quick Start

```python
from ldp_poison_toolkit import AttackRunner

# Set up an experiment
runner = AttackRunner(
    n=10000,    # Honest users
    d=50,       # Domain size (number of distinct items)
    epsilon=2.0,# LDP privacy budget
    n2=1000,    # Fake users (~10% of total)
    r=3,        # Number of target items
    x=10,       # Select targets from top-10 most frequent items
    seed=42,    # Reproducibility
)

# Run a greedy attack on GRR
result = runner.run(protocol='grr', attack='greedy')
print(result.summary())
```

**Sample output:**
```
Protocol   : GRR
Attack     : greedy
ε          : 2.0
n / n2     : 10000 / 1000 (9.1% fake)
d / r      : 50 / 3
Targets    : [4, 11, 23]
Freq gain  : +312
Rank gain  : +7
  item   4 : rank 5 → 2
  item  11 : rank 8 → 6
  item  23 : rank 12 → 10
```

---

## Package Structure

```
ldp_poison_toolkit/
├── __init__.py              # Public API
│
├── core/                    # LDP Protocol Implementations
│   ├── grr.py               # Generalized Random Response
│   ├── oue.py               # Optimized Unary Encoding
│   └── olh.py               # Optimized Local Hashing
│
├── attacks/                 # Poisoning Attack Strategies
│   ├── grr_attacks.py       # random, greedy, MPOIA for GRR
│   ├── oue_attacks.py       # random, ROA, greedy for OUE
│   └── olh_attacks.py       # random, ROA, greedy for OLH
│
├── utils/                   # Shared Utilities
│   ├── data.py              # Dataset generation and preprocessing
│   ├── metrics.py           # Attack evaluation metrics
│   └── estimators.py        # Protocol wrappers + expected frequency helpers
│
├── experiments/             # Experiment Infrastructure
│   ├── runner.py            # AttackRunner + AttackResult
│   └── benchmark.py         # Multi-config benchmarking
│
├── tests/
│   └── test_toolkit.py      # 27 unit + integration tests
│
└── examples/
    ├── quickstart.py        # Single-run demo
    └── benchmark_example.py # Multi-trial benchmark demo
```

---

## API Reference

### `AttackRunner`

The primary high-level interface. Handles dataset generation, LDP estimation, attack injection, and metric computation.

```python
AttackRunner(n, d, epsilon, n2, r=3, x=10, m=10.0, seed=None)
```

| Parameter | Description |
|-----------|-------------|
| `n` | Number of honest users |
| `d` | Domain size |
| `epsilon` | LDP privacy budget (ε) |
| `n2` | Number of adversarial fake users |
| `r` | Number of target items |
| `x` | Target selection pool (top-x items) |
| `m` | Skewness of synthetic dataset (higher = more uniform) |
| `seed` | Random seed for reproducibility |

```python
runner.run(protocol, attack, **kwargs) -> AttackResult
```

`protocol`: `'grr'` | `'oue'` | `'olh'`

`attack`:
- GRR: `'random'` | `'greedy'` | `'mpoia'`
- OUE: `'random'` | `'roa'` | `'greedy'`
- OLH: `'random'` | `'roa'` | `'greedy'`

### `AttackResult`

```python
result.freq_gain    # int: total frequency gain for target items
result.rank_gain    # int: total rank improvement for target items
result.before_freq  # dict: frequencies before attack
result.after_freq   # dict: frequencies after attack
result.before_ranks # dict: ranks before attack
result.after_ranks  # dict: ranks after attack
result.summary()    # str: human-readable summary
```

### `benchmark_attacks`

```python
from ldp_poison_toolkit import benchmark_attacks

df = benchmark_attacks(
    n=10000,
    d=50,
    epsilons=[1.0, 2.0, 4.0],
    n2_ratios=[0.1, 0.2, 0.3],
    protocols=['grr', 'oue'],
    attacks={'grr': ['random', 'greedy'], 'oue': ['greedy']},
    r=3, x=10, trials=10,
)
print(df)
```

Returns a pandas DataFrame with mean and std of `freq_gain` and `rank_gain` across trials.

---

## Low-Level Protocol API

Use the core protocols directly for custom experiments:

```python
from ldp_poison_toolkit.core.grr import GRR_Client, GRR_Aggregator_MI

# Client-side: perturb a single value
report = GRR_Client(input_data=7, d=20, epsilon=2.0)

# Server-side: aggregate reports
reports = [GRR_Client(v, d=20, epsilon=2.0) for v in my_dataset]
freq_estimates = GRR_Aggregator_MI(reports, d=20, epsilon=2.0)
```

```python
from ldp_poison_toolkit.core.oue import UE_Client, UE_Aggregator_MI

vec = UE_Client(input_data=3, d=20, epsilon=2.0)  # binary vector
freq = UE_Aggregator_MI([vec, ...], d=20, epsilon=2.0)
```

```python
from ldp_poison_toolkit.core.olh import LH_Client, LH_Aggregator_MI

report = LH_Client(input_data=5, d=100, epsilon=2.0)  # (value, seed) tuple
freq = LH_Aggregator_MI([report, ...], d=100, epsilon=2.0)
```

---

## Attack Details

### Random Attack
**Baseline.** Each fake user uniformly picks one target item and submits a
report as if they honestly held that value. Simple but easily detectable since
it produces an unusually high concentration of reports for target items.

### ROA (Random Output Attack)
**Noise baseline for OUE/OLH.** Fake users submit reports that support only
non-target items, chosen uniformly at random. ROA inflates non-target counts
equally, lowering target items' relative standing — useful as a *negative*
baseline showing what indiscriminate noise injection achieves.

### Greedy Attack
**Strategic rank manipulation.** The adversary identifies *effective attack
items* (non-target items ranked above at least one target item) and iteratively
focuses fake users on the item requiring the fewest additional reports to be
surpassed by a target item. This minimizes wasted budget.

For OUE, each fake binary vector supports up to `E_1` positions simultaneously,
making the per-user impact higher than GRR.

### MPOIA (GRR only)
**Statistical greedy.** Extends the greedy attack by accounting for the
variance of the Matrix Inversion (MI) estimator. The distance threshold δ is
set so that the attack item overtakes the target item with probability ≥
`confidence` (default 90%), providing better performance when n is small
relative to d and perturbation noise dominates.

---

## Running Tests

```bash
pip install pytest
pytest tests/test_toolkit.py -v
```

All 27 tests should pass (unit tests for core protocols, utilities, attacks, and integration tests for `AttackRunner`).

---

## Extending the Toolkit

### Adding a New Protocol

1. Create `core/myprotocol.py` with `MyProtocol_Client` and `MyProtocol_Aggregator_MI`.
2. Add it to `core/__init__.py`.
3. Add a wrapper in `utils/estimators.py`.
4. Wire it into `experiments/runner.py`.

### Adding a New Attack

1. Create `attacks/myprotocol_attacks.py` with a function:
   ```python
   def my_attack(n2, target_items, ...) -> List[report]:
       ...
   ```
2. Add it to `attacks/__init__.py`.
3. Add dispatch in `AttackRunner._myprotocol_fake()`.

---

## Citation

If you use this toolkit in your research, please cite the foundational protocols:

```bibtex
@inproceedings{wang2017locally,
  title={Locally differentially private protocols for frequency estimation},
  author={Wang, Tianhao and Blocki, Jeremiah and Li, Ninghui and Jha, Somesh},
  booktitle={USENIX Security},
  year={2017}
}
```

This toolkit's protocol implementations are adapted from the
[Multi-Freq-LDPy](https://github.com/hharcolezi/multi-freq-ldpy) package.
Special thanks to its authors.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
