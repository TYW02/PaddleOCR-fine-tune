# Handover checks — 30 October 2026

Checked in the existing local Python 3.12 OCR environment:

- All 104 copied source/evidence files match their recorded packaged hashes.
- All eight notebooks pass nbformat validation; their code cells and Python
  modules compile, including the standalone notebook's embedded module bodies.
- The paired B3 CER table and the run 3/run 4 stroke PSNR gains in the presentation
  notes were recomputed from the included per-pair CSVs.
- The copied pipeline notebook executed with cached real PaddleOCR models from
  an otherwise empty temporary folder. External `preprocessing` imports were
  blocked. It exported a nonempty transcript containing `Animal`, OCR JSON,
  annotated boxes and crops using baseline mode.
- The original notebook 08, source photograph and copied notebook remained
  unchanged by the smoke run. No datasets, weights or dependency checkouts are
  included. Credential-pattern and staged whitespace checks passed.

These checks validate the handover and one baseline execution. They do not
retrain the archived models, test every optional preprocessing mode, establish
OCR accuracy on a new dataset, or validate a fresh install on every platform.
