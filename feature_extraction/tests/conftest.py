import sys
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]
BENCHMARK_PACKAGE = HERE.parents[1] / "ToothFairy2-Benchmark" / "benchmark_networks"
sys.path.insert(0, str(BENCHMARK_PACKAGE))
sys.path.insert(0, str(HERE))
