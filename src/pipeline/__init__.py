"""Anti-spoof pipeline assembler.

Wires the gates + fusion evaluator + (caller-supplied) device-spoof
evaluator into one duck-typed adapter so the FIVUCSAS biometric-processor
can attach a single structured result to its `/verify` payload.
"""

from src.pipeline.assembler import (
    AntispoofPipelineAssembler,
    AntispoofPipelineResult,
)

__all__ = ["AntispoofPipelineAssembler", "AntispoofPipelineResult"]
