# Super-resolution: what we tried and what we learned

## Aim and evaluation

Improve PaddleOCR reading of small, distant handwriting after text cropping.
Compare the original input, the same preprocessing without the CNN, the restored
image, and a real clear target. **CER (lower is better)** is the relevant outcome;
PSNR measures pixel similarity and is a secondary diagnostic.

## Experiments

1. Test pretrained general-image and text-specific upscalers.
2. Train small/residual CNNs with synthetic degradation and aligned close/far pairs.
3. Remove color/brightness shortcuts and focus the loss on ink and stroke connectivity.
4. Add a frozen reader's recognition loss (M2).
5. Train a TPGSR-style word restorer on TextZoom, then attempt whiteboard adaptation.
6. Test size gates, blending and confidence selection; inspect whether the prior
   affects strokes or mainly changes the background.

These are the team's tested implementations. They do not exhaust the literature,
and the TPGSR adaptation differs from the full published method.

## Results to show

**Matched B3 test: 24 paired views of six distinct lines.** Mean line CER,
using the saved historical scoring definition:

| Method | Raw input | No CNN | Restored |
|---|---:|---:|---:|
| Run 3: ink-matched pixel training | 0.157 | 0.169 | 0.154 |
| Run 4: additional validation and training | 0.157 | 0.169 | 0.174 |
| M1: stroke-weighted / Dice / clDice loss | 0.157 | 0.169 | 0.168 |
| M2: frozen-reader recognition loss | 0.157 | 0.169 | 0.181 |

Run 3's small numerical gain was not a convincing general improvement. Run 4
increased stroke PSNR gain from **+0.41 to +1.36 dB**, while CER worsened from
0.154 to 0.174. M1 and M2 failed the predeclared rule on this board: beat both
raw and no-CNN inputs, with more paired views improving than worsening.

**TPGSR whiteboard test: 23 eligible B3 line views.** A line without word boxes
was excluded, so this table must not be treated as the exact same sample set:

| Version | Mean line CER |
|---|---:|
| Raw input | 0.153 |
| Size-gated resize path, no network | 0.153 |
| Size-gated TPGSR output | 0.166 |
| Blend with raw image, alpha 0.3 | 0.159 |
| Confidence gate selected on validation | 0.167 |

Whiteboard fine-tuning selected epoch 0: later epochs did not improve validation
CER. The reported fine-tuned checkpoint therefore remained the pretrained one.

Sources: the [experiment journal](docs/SUPER_RESOLUTION_RUNS.md),
[run 3 measurements](results/_sr_review/run3_ink_matched/accepted_model_ocr_comparison.csv),
[run 4 measurements](results/_sr_review/run4_pixel_loss/accepted_model_ocr_comparison.csv),
[M1 measurements](results/paired_dataset_17oct/m1_stroke_loss_ocr_comparison.csv),
[M2 measurements](results/paired_dataset_17oct/m2_recognition_loss_ocr_comparison.csv),
and [TPGSR summary](results/tpgsr_whiteboard/summary.txt).

## Figures for the slides

- [Run 4: original / restored / target](results/_sr_review/run4_pixel_loss/accepted_model_visual_comparison.png)
- [M1 stroke loss comparison](results/_sr_review/m1_stroke_loss/m1_stroke_loss_visual_comparison.png)
- [M2 recognition loss comparison](results/paired_dataset_17oct/m2_recognition_loss_visual_comparison.png)
- [TPGSR line comparison](results/tpgsr_whiteboard/visual_comparison.png)
- [Correct versus incorrect text priors](results/tpgsr_debug/a_prior_examples.png)

## Interpretation and limits

Sharper-looking images did not consistently preserve the letter distinctions the
reader needed. Pixel objectives could improve background or ink appearance;
the tested auxiliary reader was weak on this handwriting, and diagnostic prior
substitutions barely changed the TPGSR output. Those are observed weaknesses of
these runs. Missing visual detail and domain mismatch are plausible explanations,
not proof that a better handwriting-specific SR model cannot succeed.

Evidence is limited: one repeatedly inspected test board, six distinct texts in
the paired test, imperfect alignment/label review, and single-seed experiments.
Some source crops were selected using an existing OCR reader. Claims about all
handwriting, all distant photos, or all SR methods would exceed these tests.

**Project decision:** keep SR out of the default application pipeline; present
the negative results, diagnostics and lessons as a research contribution.
Evaluate any future SR proposal on new boards with verified labels and OCR CER.
