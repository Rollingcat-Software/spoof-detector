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

For productization of any module — i.e. graduating it from `research/` to
`src/` — see [`../ROADMAP.md`](../ROADMAP.md).
