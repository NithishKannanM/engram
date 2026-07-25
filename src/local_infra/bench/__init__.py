"""Benchmark rig — measure recall@k, IRR, latency, and cost per backend.

M11 ports ex_mem's synthetic-conversation benchmark onto the ``MemoryService`` seam
so BOTH backends (exmem, cognee) run through the same harness and produce comparable
numbers. The rig drives only the public interface — ``remember`` then ``recall`` — so
it never reaches past the seam into a concrete backend.

Two honesty rules from the plan are baked in here:
  * the pricing table (:mod:`bench.pricing`) uses REAL Anthropic model IDs and list
    prices; local Ollama inference is genuinely free, so a local run reports $0.00.
  * the IRR judge is deterministic and local by default (:mod:`bench.judge`), so the
    whole rig runs offline with no cloud judge cost, and judgments are cached.
"""

from local_infra.bench.metrics import BenchmarkResult
from local_infra.bench.runner import BenchmarkRunner

__all__ = ["BenchmarkResult", "BenchmarkRunner"]
