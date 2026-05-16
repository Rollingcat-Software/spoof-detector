"""Public re-export of ``src.infrastructure.analyzers`` under the
``spoof_detector.infrastructure.analyzers`` namespace.

Downstream consumers (e.g. FIVUCSAS biometric-processor) should import
the EAR helper / BlinkAnalyzer from this module rather than reaching
into the unstable ``src.*`` paths. The 2026-05-11 blink-cache + EAR
recalibration work landed in ``src.infrastructure.analyzers.blink_analyzer``
and was previously not reachable through the public shim.
"""

# Re-export the blink_analyzer submodule by name so that
# ``from spoof_detector.infrastructure.analyzers.blink_analyzer import
# compute_ear`` resolves against the on-disk implementation in ``src/``.
from src.infrastructure.analyzers import blink_analyzer  # noqa: F401
