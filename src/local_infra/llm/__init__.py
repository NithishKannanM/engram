"""Chat and RAG-synthesis LLM routing.

Deliberately separate from ``local_infra.extraction``: the extraction ladder speaks
grammar-constrained JSON to known-good local models and is guarded so it can never
reach a cloud provider. This package handles free-form generation, which *may* use a
cloud model when explicitly configured and not offline.

The one thing they share is :mod:`local_infra.llm.gpu` — the GPU serialization lock.

**Never import ``litellm`` at module scope anywhere in this package.** It costs 2.2s
and 213MB RSS, and it fetches a model-cost map over the network at import time unless
``LITELLM_LOCAL_MODEL_COST_MAP`` is set first. :mod:`local_infra.llm.client` handles
both; everything else imports lazily through it.
"""
