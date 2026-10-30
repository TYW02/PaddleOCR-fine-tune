# Super-resolution research archive

**Status: recorded experiments; excluded from the application pipeline.**
The experiments did not demonstrate a consistent PaddleOCR CER improvement on the
unseen whiteboard. This is useful negative evidence, not proof that all SR methods
fail. [Presentation notes](PRESENTATION_NOTES.md) summarize what to show.

## Notebook map

| Notebook in `notebooks/experiments/` | Purpose |
|---|---|
| `super_resolution_pretrained.ipynb` | ESPCN, FSRCNN, EDSR, Real-ESRGAN and text-specific pretrained comparisons |
| `super_resolution_trained.ipynb` | Synthetic degradation, small CNNs, real aligned close/far pairs, pixel/stroke/recognition losses |
| `textzoom_external_test.ipynb` | TextZoom LMDB test data, shallow versus residual CNNs and official text-SR comparisons |
| `tpgsr_textzoom_experiment.ipynb` | Single-stage TSRN_TL reproduction, text-prior training, blends and confidence gates |
| `tpgsr_whiteboard_finetune.ipynb` | Adapt TextZoom training to whiteboard word crops and evaluate assembled lines |
| `tpgsr_debug.ipynb` | Prior substitutions, receptive-field probes, stroke versus background changes |
| `sr_debug_activations.ipynb` | Earlier CNN filters, activations and where corrections are applied |

The notebooks retain historical code and text/table outputs. Large image/widget
payloads were removed from these Git copies; selected figures are in `results/`.
The original notebooks in `whiteboard-ocr` were not edited. Section numbers and
saved outputs describe the experiment version that was run, not a promise that
the current settings reproduce every earlier run. Some settings start training
or resume an existing checkpoint: inspect them before running cells.

## Evidence layout

`results/` preserves the subpaths originally inside `outputs/`:

- `_sr_review/run1_colour_model/` through `run4_pixel_loss/`: per-pair OCR and PSNR.
- `_sr_review/m1_stroke_loss/` and `paired_dataset_17oct/`: M1/M2 results and figures.
- `tpgsr_textzoom/`: histories, test measurements, blend/gate/checkpoint comparisons.
- `tpgsr_whiteboard/`: line-level comparisons, training history and word-swap tests.
- `tpgsr_debug/`: prior tests and diagnostic figures.
- `textzoom_external_test/` and `super_resolution/`: earlier benchmark results.

In the [historical journal](docs/SUPER_RESOLUTION_RUNS.md), a reference to
`outputs/X` means `results/X` when reading saved evidence here. Research code
still writes new runs into ignored `outputs/`. `historical/` contains available
run-specific trainer/normalization snapshots; `scripts/` contains the latest
local research source. Do not substitute the latest trainer when claiming to
reproduce an earlier run's exact settings.

## Re-running is a separate setup

Start Jupyter **inside this `sr_research/` directory**, so notebook root searches
find its `scripts/`, `requirements.txt` and `notebooks/`, rather than the outer
PaddleOCR repository. Use a separate research environment. `requirements.txt`
records the source project's versions; it is not a new lockfile validated on all
CUDA systems. Install a compatible PaddlePaddle CPU/GPU build and PyTorch build
for the target machine. Do not replace a working GPU installation with the CPU
package just to match a historical requirements line. BasicSR compatibility
shims remain in the pretrained notebook.

Code and evidence are included, but the following assets are required separately:

- `data/textzoom/`: the authors' paired train/test LMDB data, with official test
  subsets kept out of training. Use the [TextZoom release](https://github.com/JasonBoy1/TextZoom).
- `data/board2notes_v1/`: the earlier photo/label set used by the first notebooks.
- `outputs/paired_dataset_17oct/`, `outputs/paired_dataset_airdrop/` and
  `outputs/paired_dataset_airdrop_synth/`: reviewed manifests and the aligned
  input/target crops named by those manifests. Metrics in `results/` alone are
  not training data. Restore paths consistently with the original layout.
- Local trained checkpoints, including the accepted-pair restorer and the
  epoch-2/soft-prior TPGSR checkpoints used by resume/comparison cells.
- Some data-building/reader-loss scripts expect sibling folders one level above
  this directory: `iam_line_paddle/`, `airdrop_rec_paddle/`, `17oct_rec_paddle/`,
  `PhotosToUse/`, and `models/TPGSR/`. Inspect their constants or recreate that
  layout in a separate working copy before reproducing those steps.

TPGSR source is an external dependency, not vendored here. The local experiments
used [the authors' repository](https://github.com/mjq11302010044/TPGSR) at commit
`dc3e3b1dc1425fc4fcb3f4439b27482a6f92476b`, including its pretrained CRNN prior.
Our `train_tpgsr_textzoom.py` is a **single-stage adaptation** with a frozen prior;
it is not a full reproduction of the paper's training procedure.
For the older pretrained alternatives, see the historical
[model download notes](docs/ORIGINAL_MODEL_SETUP.md). Those notes describe the old
workspace and include outdated pipeline commentary: the current application
entry point is notebook 10 and defaults to baseline preprocessing.

Existing measurements use different sample sets and some historical CER helpers
cap each line at 1. Keep those definitions when reproducing saved tables; do not
pool them with uncapped CER from newer experiments. Repeated photos of one text
are not independent test sentences. The same B3 board was used repeatedly for
research, so a final confirmation requires fresh boards and verified labels.

The journal's later reader/fusion/vocabulary sections are historical follow-up
work. They are not enabled in the shipped pipeline, and the newer team recognizer
training results should be reported separately.
