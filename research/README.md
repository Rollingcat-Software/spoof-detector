# Research Index

This directory is the consolidated home of every liveness / anti-spoof research
contribution that has fed into the production code under `src/`. Nothing here
is on the import path of the productized library. It is reference material,
kept verbatim from its original branches so that future researchers (and
reviewers / auditors) can trace algorithmic ideas back to their authors and
their original commits.

## Layout

```
research/
├── aysenur/                         Aysenur (@Aysenur15) FIVUCSAS R&D branches
│   ├── working_spoof_detection/     flagship — most fertile branch
│   ├── liveness_capture/            rPPG + screen-replay + MRZ work (co-authored
│   │                                with Ayşe Gülsüm Eren)
│   ├── Spoof-Detection/             early integration branch
│   ├── feat-anti-spoof-pipeline-local/  Aysenur's clean squash for review
│   ├── liveness-cascade-frr-reduction/  cascade-style False-Reject reduction
│   ├── liveness-p0-frr-reduction/   P0 FRR work
│   └── liveness-p3-frr-reduction/   P3 FRR work
├── ayse-gulsum-eren/                attribution + commit pointers
└── ahmet-original-spoof-detector/   pointer to the user's pre-extraction work
                                     (the contents now live in /src — kept here
                                     as an attribution record only)
```

## Provenance

Each `aysenur/<branch>/` directory is a verbatim copy of
`practice-and-test/liveness-antispoof-research/aysenur15/<branch>/unique-source/`
as of 2026-05-09. Caches (`__pycache__`, `.pytest_cache`, `*.pyc`) and files
larger than 1 MB were excluded.

The `BRANCH_INFO.md` from each upstream snapshot is preserved as `README.md`
inside each branch directory.

## Resuming research

`aysenur/working_spoof_detection/` is the recommended starting point — it
contains the union of every other Aysenur branch plus the flash / Gabor moire /
hybrid-fusion work that was later ported into `src/`. Several modules already
deployed in `biometric-processor` originated there; see `../from_biometric_processor/`.

## Where things live

`spoof-detector` is the single canonical home for every liveness / anti-spoof
asset across the FIVUCSAS programme.

| Directory                  | Status     | Audience                                 |
|----------------------------|------------|------------------------------------------|
| `src/`                     | production | importable Python library; v0.2.0 curated. |
| `web/`                     | production | TypeScript port published as `@rollingcat/spoof-detector` + `/amispoof/` browser tester. |
| `tests/`                   | production | Python tests, green.                     |
| `tools/`                   | operator   | offline / desktop scripts.               |
| `research/aysenur/`        | reference  | Aysenur's 7 FIVUCSAS R&D branches.       |
| `research/ayse-gulsum-eren/` | reference  | attribution + commit pointers.         |
| `research/ahmet-original-spoof-detector/` | reference  | pointer; content is in `src/`. |
| `from_biometric_processor/`  | mirror     | algorithms also deployed in `bio` main; mirror-only — edit upstream first. |

See [`../from_biometric_processor/README.md`](../from_biometric_processor/README.md)
for the mirror sync policy.

## Productization checklist (research → src)

Any module graduating from `research/` (or `from_biometric_processor/`) into
`src/` must satisfy:

1. **Tests** — at least one unit test per public function; integration test
   if it crosses a gate / fusion / pipeline boundary. Co-locate under
   `tests/unit/<module-area>/`.
2. **Dependencies** — every imported package present in `requirements.txt`
   with a pinned version, and the new pin must pass Dependabot / `pip-audit`.
3. **Public API** — top-level docstring; `__all__` declared; type hints on
   public surface; no `app.*` imports left over from upstream namespaces.
4. **Provenance** — module docstring lists original author, original branch,
   and the consolidation commit it was promoted from.
5. **Attribution** — entry added to [`../AUTHORS.md`](../AUTHORS.md).

When a module graduates from `from_biometric_processor/`, also coordinate
with the bio team so that `biometric-processor` imports the productized
spoof-detector copy (rather than maintaining its own fork) — this is how the
mirror eventually retires.
