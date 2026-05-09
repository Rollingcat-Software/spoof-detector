# Ahmet Abdullah Gultekin — original `spoof-detector` work

**GitHub**: [@ahabgu](https://github.com/ahabgu)

The user's original `spoof-detector` body of work (session engine + analyzer
wiring + ISO 30107-3 calibration + the assembler pipeline) was extracted from
`fivucsas/practice-and-test/spoof-detector/` into the dedicated repository in
[commit `43d23a9`](https://github.com/Rollingcat-Software/spoof-detector/commit/43d23a9)
(2026-05-09).

That code now lives at `/src/` — it is the productized v0.2.0 baseline. This
directory exists purely as a historical / attribution pointer so the
research-consolidation tree is self-describing.

## Where it lives

| Concern                              | Path                              |
|--------------------------------------|-----------------------------------|
| Domain model                         | `/src/domain/`                    |
| Application services                 | `/src/application/`               |
| Pre-fusion gates                     | `/src/gates/`                     |
| Calibrated fusion                    | `/src/fusion/`                    |
| End-to-end assembler                 | `/src/pipeline/`                  |
| Adapters / infra                     | `/src/infrastructure/`            |
| HTTP / presentation surfaces         | `/src/presentation/`              |
| Tests (114 green at consolidation)   | `/tests/`                         |

## How this fits with the other two contributors

- **Aysenur** (`@Aysenur15`) — research material under `../aysenur/`. Several
  of her modules were ported into `/src/gates/` and `/src/fusion/` in v0.2.0
  (see `../../ATTRIBUTION.md`).
- **Ayşe Gülsüm Eren** — co-author on Aysenur's `liveness_capture` and
  `working_spoof_detection` branches; see `../ayse-gulsum-eren/README.md`.

The user's framing: "my work was very smart but not well enough — Aysenur's
and Ayşe Gülsüm's was very useful and working; we need to mix and compile all
of them in spoof-detector and extract a product from there to the real world."
This consolidation pass is step 1 of that plan.
