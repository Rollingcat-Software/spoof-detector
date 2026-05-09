# Open-access FAS datasets — discovery report

Generated 2026-05-09 by an automated dataset hunt. The brief was: find
every face-anti-spoofing dataset we can download in ~30 minutes without
going through an institutional EULA / "request access" form.

The four classical academic benchmarks (OULU-NPU, SiW, CASIA-SURF,
CelebA-Spoof) all sit behind some flavour of EULA. They are kept as
adapters in this folder for completeness but cannot be acquired from
this hunt.

Below is what we *did* successfully pull, along with everything we
evaluated and rejected. All downloads landed in `/tmp/fas_datasets/`
(not under `/opt/projects/...` per repo policy: only KVKK-consented
in-house data ships in this repo).

## Acquired (4 datasets, ~2.4 GB total)

### 1. CASIA-FASD (akahana HuggingFace mirror)

| Field | Value |
| --- | --- |
| Source | https://huggingface.co/datasets/akahana/anti-spoofing-casiafasd |
| License | Not stated on dataset card. Original CASIA-FASD: research-only. Mirror is publicly downloadable. |
| Size | 69 MB (compressed `casiafasd.tar.gz`) → 157 MB extracted |
| Local path | `/tmp/fas_datasets/akahana_casiafasd/extracted/` |
| Splits | train: 1655 frames (404 real / 1251 fake) · test: 2408 frames (591 real / 1817 fake) |
| Total samples | **4063** |
| Modality | RGB JPEGs + Kinect depth maps |
| Adapter | `casia_fasd.py` — `iter_casia_fasd(root, split, modality)` |
| Verified | Yes — adapter smoke-tested, both splits enumerate correctly |

```bash
huggingface-cli download akahana/anti-spoofing-casiafasd \
    --repo-type dataset \
    --local-dir /tmp/fas_datasets/akahana_casiafasd
cd /tmp/fas_datasets/akahana_casiafasd && \
    mkdir -p extracted && tar xzf casiafasd.tar.gz -C extracted/
```

PAI taxonomy is **not** preserved in this mirror — only real / fake.
Papers that need warped-vs-cut-vs-replay must fall back to the
original gated release.

### 2. Kainyyy/face-anti-spoof (largeCrowd-spoof)

| Field | Value |
| --- | --- |
| Source | https://huggingface.co/datasets/Kainyyy/face-anti-spoof |
| License | Not declared. Treat as research-only. |
| Size | ~1.05 GB (3611 PNG stills) |
| Local path | `/tmp/fas_datasets/kainyyy_face_anti_spoof/` |
| Splits | live: 720 · spoof: 2891 |
| Total samples | **3611** |
| Modality | RGB PNG stills, device-tagged filenames |
| Adapter | `kainyyy_largecrowd.py` — `iter_kainyyy_largecrowd(root, split)` |
| Verified | Yes — adapter smoke-tested |

```bash
huggingface-cli download Kainyyy/face-anti-spoof \
    --repo-type dataset \
    --local-dir /tmp/fas_datasets/kainyyy_face_anti_spoof
```

Filenames encode device + subject + scene tokens
(e.g. `AGL752VM_id147_s0_120.png`). Subject IDs are surfaced in
`Sample.metadata["subject_id"]` for subject-disjoint protocols.

### 3. AxonData attack-only video samples (CC-BY-4.0)

Two attack-only HF previews of AxonLabs commercial liveness datasets,
both released under CC-BY-4.0:

| Subset | Source | Size | Videos | Attack type |
| --- | --- | --- | --- | --- |
| Cut 2D-mask print | https://huggingface.co/datasets/AxonData/Anti_Spoofing_Cut_print_attack | 1.1 GB | 15 | print |
| 3D paper-mask | https://huggingface.co/datasets/AxonData/3D_paper_mask_attack_dataset_for_Liveness | 168 MB | 15 | mask |

| Field | Value |
| --- | --- |
| License | CC-BY-4.0 (commercial use of the *full* sets requires Axon licensing) |
| Local paths | `/tmp/fas_datasets/axon_cut_print/` · `/tmp/fas_datasets/axon_3d_mask/` |
| Total samples | **30** videos |
| Modality | iOS / Android phone video, 7–15 s each |
| Adapter | `axon_video.py` — `iter_axon_video(root, kind={"cut_print","3d_mask"})` |
| Verified | Yes — adapter smoke-tested |

```bash
huggingface-cli download AxonData/Anti_Spoofing_Cut_print_attack \
    --repo-type dataset \
    --local-dir /tmp/fas_datasets/axon_cut_print
huggingface-cli download AxonData/3D_paper_mask_attack_dataset_for_Liveness \
    --repo-type dataset \
    --local-dir /tmp/fas_datasets/axon_3d_mask
```

Both are **attack-only** — no bonafide samples. They must be paired
with a live-face corpus (Marmara in-house, CASIA-FASD live frames,
or Kainyyy live PNGs) to form a usable binary PAD train/eval set.

### 4. CelebA-Spoof eval shard (nguyenkhoa HuggingFace mirror)

| Field | Value |
| --- | --- |
| Source | https://huggingface.co/datasets/nguyenkhoa/antispoofing-3 |
| License | Not declared on the HF mirror. Original CelebA-Spoof: research-only (CUHK). |
| Size | 440 MB (one of four eval shards; full set is ~16 GB and skipped) |
| Local path | `/tmp/fas_datasets/nguyenkhoa_eval_shard/data/eval-00000-of-00004.parquet` |
| Splits | eval (this shard): 2611 rows (874 live / 1737 spoof) |
| Total samples (this shard) | **2611** |
| Modality | Parquet rows with `cropped_image` + full `image` JPEG bytes + bbox |
| Adapter | `celeba_spoof_hf.py` — `iter_celeba_spoof_hf(parquet_paths, ...)` |
| Verified | Yes — adapter smoke-tested |

```bash
huggingface-cli download nguyenkhoa/antispoofing-3 \
    data/eval-00000-of-00004.parquet \
    --repo-type dataset \
    --local-dir /tmp/fas_datasets/nguyenkhoa_eval_shard
```

PAI taxonomy is also flattened to live/spoof in the HF mirror.
Three more eval shards (~1.6 GB total) and 29 train shards (~14 GB)
are available from the same repo if a paper run needs them — the
adapter accepts a list of parquet paths.

## Aggregate counts

| Source | Bonafide | Attack | Total |
| --- | ---: | ---: | ---: |
| CASIA-FASD train | 404 | 1251 | 1655 |
| CASIA-FASD test  | 591 | 1817 | 2408 |
| Kainyyy largeCrowd | 720 | 2891 | 3611 |
| Axon cut_print  |   0 |   15 |   15 |
| Axon 3d_mask    |   0 |   15 |   15 |
| nguyenkhoa eval shard #0 | 874 | 1737 | 2611 |
| **Total** | **2589** | **7726** | **10315** |

## Recommended for benchmarks

1. **CASIA-FASD test** — 2408 frames, real/fake, classical baseline,
   matches the headline numbers in most prior FAS papers.
2. **nguyenkhoa CelebA-Spoof eval shard #0** — 2611 face-cropped
   samples, fastest path to a CelebA-Spoof reference number without
   the EULA. Pulling shards 1-3 brings the total to ~10k face crops
   (~1.6 GB).
3. **Kainyyy largeCrowd-spoof** — 3611 stills, useful as a
   third-party in-the-wild check that doesn't overlap the two
   benchmarks above.
4. **Axon cut_print + 3d_mask + Marmara in-house live** — for an
   attack-type-disaggregated APCER report (print vs. mask vs. replay).
   Small, but the only EULA-free source of explicitly labelled
   attack categories we found in this hunt.

## Rejected candidates (and why)

| Candidate | Reason |
| --- | --- |
| `nguyenkhoa/antispoofing` (494k samples) | 46 GB train shard set — exceeds 10 GB disk budget |
| `nguyenkhoa/celeba-spoof-for-face-antispoofing-test` (67k samples) | 4.9 GB — would consume half the budget for one dataset; eval shard from antispoofing-3 covers the same source at smaller cost |
| `nguyenkhoa/antispoofing-3` full set | 16 GB — only one eval shard pulled |
| `Bahareh0281/Liveness_Detection_Videos_Frames` | 1.45 GB single zip, no license declared, no labels file in the README — risky |
| `AxonData/2d-paper-mask-face-anti-spoofing` | CC-BY-NC-4.0 — non-commercial restriction excludes the eventual paper deliverable |
| `AxonData/face-anti-spoofing-dataset` (151 files) | CC-BY-NC-4.0 — same |
| `AxonData/print-cardboard-mask-face-spoofing` | CC-BY-NC-4.0 |
| `AxonData/liveness-detection-dataset` | CC-BY-NC-4.0 |
| `Shravan-2007/face-anti-spoofing-dataset` (151 files) | CC-BY-NC-4.0 |
| `VonRommel/face-anti-spoofing-dataset` (151 files) | CC-BY-NC-4.0 |
| `kakusyun/face-anti-spoofing-advanced-paper-attacks` (5 KB) | README only — no actual binaries on the HF repo |
| `nguyenkhoa/antispoofing-1`, `-2` | Same family as `-3`; redundant for the budget |
| `ArturoHurtado7/AntiSpoofing` | **Audio** anti-spoofing (ASVspoof family), not face — out of scope |
| `jonathansuru/anti_spoofing` | Only `files.csv` (manifest), no binaries on HF |
| Kaggle datasets | `kaggle` CLI not configured in this environment |
| ROSE-YOUTU, NUAA, MSU-MFSD, Idiap Replay-Attack | All require institutional EULA forms — checked, not pulled |
| `IS2AI/Kazakh-Face-Anti-Spoofing-Dataset` (GitHub) | Repo 404 |
| `Podidiving/lcc_fasd` (GitHub) | Repo 404 |
| `ee09115/spoofing_detection` (GitHub) | Trained models + scripts only, no data |
| `Davidzhangyuanhan/CelebA-Spoof` (GitHub) | Original CelebA-Spoof — link to gated CUHK release, no data in repo |
| `clks-wzz/FAS-SGTD` (GitHub) | Code only, no bundled data |

## Reproducibility

To pull every acquired dataset on a clean machine:

```bash
pip install --upgrade huggingface_hub pyarrow pandas
mkdir -p /tmp/fas_datasets

# 1) CASIA-FASD (~70 MB)
huggingface-cli download akahana/anti-spoofing-casiafasd \
    --repo-type dataset \
    --local-dir /tmp/fas_datasets/akahana_casiafasd
( cd /tmp/fas_datasets/akahana_casiafasd && \
  mkdir -p extracted && tar xzf casiafasd.tar.gz -C extracted/ )

# 2) Kainyyy largeCrowd-spoof (~1.0 GB)
huggingface-cli download Kainyyy/face-anti-spoof \
    --repo-type dataset \
    --local-dir /tmp/fas_datasets/kainyyy_face_anti_spoof

# 3) Axon cut-print + 3d-mask attack videos (CC-BY-4.0, ~1.3 GB)
huggingface-cli download AxonData/Anti_Spoofing_Cut_print_attack \
    --repo-type dataset \
    --local-dir /tmp/fas_datasets/axon_cut_print
huggingface-cli download AxonData/3D_paper_mask_attack_dataset_for_Liveness \
    --repo-type dataset \
    --local-dir /tmp/fas_datasets/axon_3d_mask

# 4) CelebA-Spoof eval shard (~440 MB)
huggingface-cli download nguyenkhoa/antispoofing-3 \
    data/eval-00000-of-00004.parquet \
    --repo-type dataset \
    --local-dir /tmp/fas_datasets/nguyenkhoa_eval_shard
```

Total on-disk footprint: ~2.4 GB (well under the 10 GB budget).
