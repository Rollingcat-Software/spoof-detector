"""Public API alias for the on-disk ``src`` package.

This shim re-exports the curated production layer of the spoof-detector
library under the ``spoof_detector`` namespace. Downstream consumers
(e.g. FIVUCSAS biometric-processor) import from here. In-tree tests still
use ``from src.* import ...`` directly; both work.

The shim approach (vs renaming ``src/`` to ``spoof_detector/``) keeps the
existing 114 tests passing without a refactor pass.
"""

__version__ = "0.2.1"

# Re-export top-level subpackages so ``import spoof_detector.gates`` works.
from src import gates  # noqa: F401
from src import fusion  # noqa: F401
from src import pipeline  # noqa: F401
from src import application  # noqa: F401
from src import domain  # noqa: F401
from src import infrastructure  # noqa: F401
from src import presentation  # noqa: F401
