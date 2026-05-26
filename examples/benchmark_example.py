"""
Benchmark Example
==================
Runs a multi-epsilon, multi-n2 benchmark across GRR and OUE protocols
and prints a summary table.

Run:
    python examples/benchmark_example.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ldp_poison_toolkit import benchmark_attacks

def main():
    print("=" * 70)
    print("  LDP Poison Toolkit — Benchmark")
    print("=" * 70)

    df = benchmark_attacks(
        n=5000,
        d=30,
        epsilons=[1.0, 2.0, 4.0],
        n2_ratios=[0.1, 0.2],
        protocols=["grr", "oue"],
        attacks={
            "grr": ["random", "greedy"],
            "oue": ["random", "greedy"],
        },
        r=3,
        x=8,
        trials=3,
        seed=0,
        verbose=True,
    )

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
