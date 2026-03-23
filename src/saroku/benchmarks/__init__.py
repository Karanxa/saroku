"""
saroku.benchmarks — Versioned static benchmark registry.

Usage:
    from saroku.benchmarks import load_benchmark, list_benchmarks, BENCH_REGISTRY

    instances = load_benchmark("bench-v1")
"""

from saroku.benchmarks.bench_v1 import BENCH_V1_INSTANCES, BENCH_VERSION, BENCH_DESCRIPTION

BENCH_REGISTRY: dict[str, dict] = {
    "bench-v1": {
        "instances": BENCH_V1_INSTANCES,
        "version": BENCH_VERSION,
        "description": BENCH_DESCRIPTION,
        "instance_count": len(BENCH_V1_INSTANCES),
    },
}


def load_benchmark(name: str):
    """Load ProbeInstances for a named benchmark version."""
    if name not in BENCH_REGISTRY:
        available = ", ".join(BENCH_REGISTRY.keys())
        raise ValueError(f"Unknown benchmark '{name}'. Available: {available}")
    return BENCH_REGISTRY[name]["instances"]


def list_benchmarks() -> list[dict]:
    """Return metadata for all registered benchmarks."""
    return [
        {
            "name": name,
            "version": meta["version"],
            "description": meta["description"],
            "instance_count": meta["instance_count"],
        }
        for name, meta in BENCH_REGISTRY.items()
    ]
