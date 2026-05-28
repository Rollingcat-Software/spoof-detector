# Security Policy

## Reporting a Vulnerability

If you believe you've found a security vulnerability in spoof-detector — the anti-spoofing and liveness detection engine of the FIVUCSAS biometric authentication platform — please report it privately so we can fix it before disclosing publicly.

**Email:** info@fivucsas.com (primary) or rollingcat.help@gmail.com (alternate) — subject prefix: `[SECURITY] spoof-detector`

Please include:
- A clear description of the issue and its impact.
- Steps to reproduce, ideally with a minimal proof of concept.
- Affected versions or commit SHAs if known.
- Whether the issue is already public.

We commit to:
- Acknowledging your report within **3 business days**.
- Providing a full assessment within **10 business days**.
- Coordinating disclosure timing with you once a fix is ready.

## Scope

In scope:
- Presentation-attack / liveness-detection bypass (print, replay, screen, mask, deepfake).
- Anti-spoofing classifier evasion or tampering with detection thresholds.
- Model or weight integrity (poisoning, substitution, adversarial inputs that defeat detection).
- Server-side injection (command, path traversal, deserialization) in the detection pipeline.
- Data exposure of captured frames, embeddings, or detection telemetry beyond the authenticated caller.

Out of scope:
- Issues requiring physical access to a user's device.
- Social-engineering of platform staff or end-users.
- Best-practice hardening recommendations without a concrete attack path (please open a regular issue).
- Self-XSS, missing security headers without exploit, clickjacking on non-state-changing pages.

## Safe-Harbor

Good-faith research that respects user privacy, doesn't degrade service, and follows this disclosure process is welcome. We will not pursue legal action against researchers who follow this policy.
