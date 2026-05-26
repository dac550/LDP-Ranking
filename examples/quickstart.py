"""
Quick Start Example
====================
Demonstrates a single attack run using the AttackRunner API.

Run:
    python examples/quickstart.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ldp_poison_toolkit import AttackRunner

def main():
    print("=" * 60)
    print("  LDP Poison Toolkit — Quick Start")
    print("=" * 60)

    # Configuration
    n       = 5000    # Honest users
    d       = 30      # Domain size
    epsilon = 2.0     # Privacy budget
    n2      = 500     # Fake users (~10%)
    r       = 3       # Target items
    x       = 8       # Pool: select targets from top-8

    runner = AttackRunner(
        n=n, d=d, epsilon=epsilon, n2=n2,
        r=r, x=x, seed=42,
    )

    print(f"\nDataset: n={n} users, d={d} items, ε={epsilon}")
    print(f"Attack budget: n2={n2} fake users ({100*n2/(n+n2):.1f}%)")
    print(f"Target items: {list(runner.target_items)}\n")

    for protocol in ["grr", "oue", "olh"]:
        attacks = {
            "grr": ["random", "greedy", "mpoia"],
            "oue": ["random", "roa", "greedy"],
            "olh": ["random", "roa"],
        }[protocol]

        print(f"─── {protocol.upper()} ───────────────────────────────")
        for attack in attacks:
            try:
                result = runner.run(protocol=protocol, attack=attack)
                print(
                    f"  {attack:8s} | freq_gain={result.freq_gain:+5d} | "
                    f"rank_gain={result.rank_gain:+4d}"
                )
            except Exception as e:
                print(f"  {attack:8s} | ERROR: {e}")
        print()

    print("Done. For multi-trial benchmarks, see examples/benchmark_example.py")

if __name__ == "__main__":
    main()
