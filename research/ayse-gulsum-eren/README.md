# Ayşe Gülsüm Eren — contribution map

**Email**: ayse.gulsum@marun.edu.tr (academic) · aysegulsumeren@gmail.com (public)
**GitHub**: [@aysegulsum](https://github.com/aysegulsum)
**Commit-message style**: Conventional Commits

Ayşe Gülsüm Eren is a co-author on the FIVUCSAS liveness/anti-spoof R&D track.
Her work is interleaved within the Aysenur (`@Aysenur15`) branches snapshotted
under `../aysenur/`, primarily in:

- `../aysenur/liveness_capture/` — rPPG, screen-replay defence, MRZ pipeline.
- `../aysenur/working_spoof_detection/` — selected modules carried forward
  into the flagship branch.

Because the snapshots are full unique-source dumps (not per-author splits),
specific authorship attaches to individual commits in the upstream
`biometric-processor` repository rather than to specific files here. To find
her commits:

```bash
git -C /opt/projects/fivucsas/biometric-processor log \
    --author='aysegulsumeren@gmail.com' \
    --pretty=format:'%h %ad %s' --date=short
```

If a graduation candidate (`research/ → src/`) draws on her work, please
preserve that attribution in the relocated module's docstring and in
`../../ATTRIBUTION.md`.

## Reporting an attribution gap

If you believe your work is included here without proper attribution, please
open an issue at https://github.com/Rollingcat-Software/spoof-detector/issues —
we will fix it as a priority.
