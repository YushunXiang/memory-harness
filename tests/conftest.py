from __future__ import annotations

import os


# The harness regression suite is CPU-only.  Keep it off experiment GPUs unless
# a caller explicitly selects another JAX backend.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")
