# Super-resolution for whiteboard OCR: what was tried, why it failed, what replaces it (draft write-up)

*Draft for the report. Numbers come from the result files saved per run in `outputs/_sr_review/run*/`; the code is in `scripts/` and section 6 of `notebooks/experiments/super_resolution_trained.ipynb`.*

## 1. Goal and how it is judged

**Goal.** Text photographed from far away is small and soft, and PaddleOCR misreads it. The super-resolution (SR) step should **reconstruct the correct strokes** of such a line so that PaddleOCR reads it better.

**Pipeline.** The far line crop is first enlarged to the size of a close-up (bicubic). The board is then flattened (each image divided by its own estimated board background, `pair_data.normalise_board`) and converted to grey. A small CNN corrects the strokes at that size (SRCNN-style, three convolutions, residual, starts as an identity map). Finally the result is mapped back to the photo's own brightness before OCR.

**Judged by.** The decision metric is PaddleOCR's **character error rate (CER)** on held-out pairs: a far photo of a line plus a close photo of the same line, with a human transcription. PSNR (pixel similarity to the close photo) is reported but does not decide. Three versions are always compared:
- `raw input`: what the pipeline sees today;
- `input, no CNN`: the same path with the network removed;
- `restored`: with the network.

**Restored vs no CNN** isolates the effect of the network.

## 2. Data

| Set | Pairs | Use |
|---|---|---|
| 17 Oct board: real far/side/medium → close pairs | 174 train, 20 val, 40 test | Whole input photos held out for val and test (test photos 4748, 4750, 4697) |
| 17 Oct board: synthetic (close crop degraded) | 138 | Training only |
| Airdrop boards B1, B4, B5: synthetic pairs from sharp line crops (blur, 3–6× downscale, noise, JPEG) | 276 | Training only (from run 3) |
| Airdrop B2 (television board): real far → sharp crop pairs, ECC-aligned | 10 | Test in runs 1–3, validation from run 4 |
| Airdrop B3 (CNN board): real far → sharp crop pairs, ECC-aligned | 24 (6 different lines) | Unseen-board test |

## 3. The four runs

Each run removed a **shortcut**: an easier way for the network to lower the pixel loss than sharpening strokes.

| Run | Change | What the network actually learned | Stroke PSNR gain, 17 Oct / B3 | CER on B3, restored (raw 0.157, no CNN 0.169*) |
|---|---|---|---|---|
| 1 | Board normalisation, ink-centred patches, photo-held-out splits | **Recoloured** the 17 Oct blue marker (dark and dull from far away, bright blue close up) | +0.93 / −0.17 dB | 0.178 |
| 2 | Greyscale input and target | **Lightened** the 17 Oct strokes (the marker looks darker from far away) | +0.63 / −0.30 dB | 0.161 |
| 3 | Target ink depth matched to the input's; +276 synthetic pairs from other boards | Little: validation (17 Oct only, already near-sharp) stopped it at **epoch 1** | +0.02 / +0.41 dB | **0.154** (6 better / 16 same / 2 worse) |
| 4 | B2 real pairs join validation; patience 10 | Sharper, darker strokes; best epoch 5 of 15 | −0.20 / **+1.36 dB** (24/24 pairs) | 0.174 (3 better / 16 same / 5 worse) |

\* In run 1, "no CNN" was the normalised colour image (CER 0.174). PSNR values are not comparable across runs, because the targets changed (grey from run 2, ink-matched from run 3). CER is comparable: same pairs, same labels, same recogniser.

On the 17 Oct test photos, CER stayed at 0.095–0.098 in every grey run. That set has no room for improvement: the clear target reads no better (0.099).

## 4. Main finding: pixel fidelity and readability moved in opposite directions

On the same 24 CNN-board pairs, run 4 raised stroke PSNR from +0.41 dB (run 3) to **+1.36 dB**, yet CER went from **0.154 to 0.174**. Run 3 was barely trained (epoch 1) and was the only run that beat both `raw input` and `input, no CNN`. Run 4 beat neither. The differences are small and come from 6 different lines, so neither run's CER is a reliable gain or loss on its own. The direction, however, is clear: **more pixel fidelity did not bring more readability.**

**Why.** A far photo loses exactly the details that decide a letter:
- the gap between "u" and "a" in "Visualising";
- the diagonal that separates "N" from "M" in "CNN";
- the thin "r" in "Architecture".

When the network cannot tell which shape was there, an L1/L2 pixel loss rewards the **average** of the plausible shapes: a darker, tidier blob. That blob is closer to the target in pixels, so PSNR rises, but it is no more readable. In run 4's outputs "ua" stays merged, and PaddleOCR misreads "CNN Architecture" in all 9 of its pairs with or without the network (e.g. "CMN Achitecture", "CNN Achitecture").

**Two side observations:**
- **B2 could not inform readability.** All 10 B2 pairs read perfectly even from the far photo (CER 0.000), so using B2 for validation selects for pixel fidelity only.
- **Normalisation alone helps 17 Oct but hurts Airdrop.** On 17 Oct it lowers CER from 0.116 to 0.095. On B3 it raises it from 0.157 to 0.169. The grey, normalised image is therefore not uniformly easier for PaddleOCR to read.

## 5. Limitations

- **Small tests.** One unseen board, 24 pairs, 6 different lines. A one-word change moves the mean CER by about 0.01.
- **Single seeds and noisy validation.** Run 4's B2 validation gain jumped between 0.23 and 0.97 dB across epochs.
- **17 Oct labels are an unchecked AI first pass.** One 17 Oct pair is unreadable even in the clear target.
- **The 17 Oct split holds out camera conditions, not content.** Every val and test line also appears in training (through other photos or synthetic copies), so the Airdrop board is the content-independent test.

## 6. What changed next (M1) and the pass/fail rule

**M1 changes only the loss:**
- stroke pixels weighted 5× in the L1 term;
- a soft ink **Dice** term (missing or extra ink);
- a soft **clDice** term that compares stroke skeletons (Shit et al., CVPR 2021), so a broken stroke or two strokes merged into a blob is penalised even when few pixels differ.

The term weights (Dice 0.25, clDice 0.3) make each stroke term about 0.9× the ink-weighted L1 term at the start of training, averaged over the training mix of real pairs. The first calibration used the median patch and gave clDice 0.5. In the smoke run that made clDice about 2.5× L1, because patches of tiny, smeared text have very low stroke overlap and pull the mean up. It was corrected to 0.3 before the full run, with no M1 result seen.

**Pass/fail rule, fixed before training M1.** On real pairs from an unseen board:
1. the mean restored CER must be lower than both `input, no CNN` and `raw input`;
2. more pairs must improve than worsen.

Until a far-shot photo set exists, any verdict is preliminary.

**New test material from the teammates' PhotosToUse folder** (`scripts/make_phototouse_pairs.py`). Its "far" shots are still too large to need SR (letters 50–250 px tall), so it provides:
- **Low-resolution test (synthetic):** 15 labelled lines from 11 texts, shrunk to the real far-crop range (0.22 and 0.35 source pixels per target pixel) with area averaging only. 30 pairs; reported, not decisive.
- **Camera-shake stress test:** 8 real shaken single-line crops, scored by CER only.
- **Training pairs:** 60 synthetic pairs from 15 OnePlus whiteboard photos whose texts are in neither test. These are kept for the run after M1, so that each run changes one thing.

**Camera shake is handled by a retake prompt, not restoration** (`scripts/capture_check.py`). The check measures a line crop's sharpness and how strongly its edges share one direction. Thresholds were calibrated on the 11 real shake crops against 1,014 other line crops:
- all 11 shake crops were caught, and 10 of 11 when each was left out of the fit;
- the only other crop flagged, Airdrop `IMG_4505_L02` ("apple car", read as "apple cor"), proved to be visibly shaken too.

## 7. Reproducing

Run section 6 of the notebook:
- `RUN_TAG = 'run4_pixel_loss'` with `PAIR_TRAIN = False` scores the saved run-4 model;
- `RUN_TAG = 'm1_stroke_loss'` with `PAIR_TRAIN = True` trains M1. It runs the loss self-test first, and `PAIR_SMOKE = True` gives a 2-epoch check.

Result files are named after the run tag. Runs 1–4 (model, notebook, result CSVs) are saved in `outputs/_sr_review/run1_colour_model`, `run2_grey_model`, `run3_ink_matched` and `run4_pixel_loss`.

## 8. M1 result (stroke-weighted loss): fails the rule

*From `outputs/_sr_review/m1_stroke_loss/m1_stroke_loss_ocr_comparison.csv` (99 pairs). Mean CER, lower = better.*

| Test set | Pairs | raw input | input, no CNN | **restored (M1)** | clear target | restored vs no CNN: better / same / worse |
|---|---|---|---|---|---|---|
| Airdrop, unseen board (B3) | 24 | **0.157** | 0.169 | 0.168 | 0.067 | 6 / 11 / 7 |
| 17 Oct, held-out photos | 37 | 0.116 | **0.095** | 0.099 | 0.099 | 6 / 22 / 9 |
| PhotosToUse, low resolution (synthetic) | 30 | **0.153** | 0.184 | 0.232 | 0.037 | 0 / 20 / 10 |
| PhotosToUse, camera shake (stress test) | 8 | 0.570 | **0.556** | 0.645 | 0.051 | 1 / 2 / 5 |

**Verdict against the rule fixed before training:** fails both parts on the unseen board. Restored 0.168 is not lower than raw input 0.157, and 6 pairs improve vs 7 that get worse. On the synthetic low-resolution set the network makes 10 of 30 pairs worse and none better.

So **making the loss focus on strokes did not make the network help reading**, just like the pixel losses before it. The gap to the clear target (0.157 vs 0.067 on B3) is real but is not closed by any restoration tried so far.

## 8b. M2 result (recognition loss with our own line reader, 18 Oct): fails the rule

*From `outputs/paired_dataset_17oct/m2_recognition_loss_ocr_comparison.csv`; reader `scripts/line_recognizer.py` (trained by `scripts/train_line_recognizer.py`); trainer `scripts/train_accepted_pairs.py --recognizer`; notebook section 6-0 and run tag `m2_recognition_loss`.*

**Idea (as in TextSR).** Run 4's pixel loss plus the CTC loss of a frozen line reader on the restored line, so that a restoration which reads as the wrong letter is penalised. PaddleOCR could not pass gradients to our PyTorch restorer, so a small CRNN was trained for the job on 6,482 IAM lines plus 548 crops of boards B1, B4, B5 and the 17 Oct training photos (never B2 or B3): CER 0.054 on IAM validation, **0.242 on the B3 crops** (PaddleOCR reads them at 0.125). The reading term (2 whole lines per step, weighted to equal the pixel loss at the start) covered the 276 synthetic Airdrop lines only: the 17 Oct real-pair canvases exceed its 300 px height limit. The epoch was chosen on PSNR gain (epoch 25 of 30).

| Test set | Pairs | raw input | input, no CNN | **restored (M2)** | clear target | restored vs no CNN: better / same / worse |
|---|---|---|---|---|---|---|
| Airdrop, unseen board (B3) | 24 | **0.157** | 0.169 | 0.181 | 0.067 | 3 / 13 / 8 |
| 17 Oct, held-out photos | 37 | 0.116 | 0.095 | **0.087** | 0.099 | 11 / 21 / 5 |
| PhotosToUse, low resolution (synthetic) | 30 | **0.153** | 0.184 | 0.202 | 0.037 | 3 / 19 / 8 |
| PhotosToUse, camera shake | 8 | 0.570 | **0.556** | 0.596 | 0.051 | 1 / 4 / 3 |

**Verdict: fails** on the unseen board: worse than both raw input and the no-CNN path, 3 pairs better vs 8 worse (the colour version as well: 0.188 vs 0.165). The 17 Oct gain cannot be trusted: the reader was trained on 17 Oct crops of the same text as these test lines, so the loss could pull the restorer towards phrases the reader already knew. Weak points of this version: the examiner reads our boards poorly (0.242 on B3); the reading term never saw a real far line; the epoch was chosen on PSNR, with the B2 validation board, which reads perfectly even from far (the reader's CER on its 10 labelled lines was 0.000 before and after), so selection had no readability signal.

## 9. Notes on two recurring questions (20 Oct)

**Do the losses ignore the white background and focus on the strokes?**
- Ink is only about **10% of the pixels** of a typical line crop (median 9.8%, middle half 8–12%; measured on 208 crops of `board2notes_dataset_combined/crops/images`, at 64 px tall).
- With a plain pixel loss, about 90% of the loss is therefore background. The network can lower it a lot by cleaning the board (noise, JPEG blocks, glare) without fixing a single letter. That is one reason PSNR rises while CER does not.
- Experiment 1 (ESPCN, L1) and Experiment 2 (`LineSR`, Charbonnier + 0.1 × Sobel edge loss) scored every pixel equally; they only required training patches to contain some ink.
- Runs 1–4 used ink-centred patches. **M1 is the stroke-focused version**: stroke pixels ×5, plus Dice and clDice. It did not help (section 8).
- If tried again: never weight the background 0. The network could then paint smudges or ghost strokes on the board for free.

**Does converting to greyscale make learning better (one channel instead of three)?**
- **Already done everywhere.** Experiments 1 and 2 learn on brightness (Y) only and copy colour from bicubic; runs 2–4 and M1 use grey input and grey target.
- Why it is sensible: letter shape is carried almost entirely by brightness, so the network does not waste capacity on colour, and it cannot learn colour shortcuts. (Run 1, in colour, learned to *recolour* the blue marker instead of sharpening it.)
- What it does not do: it adds no information. The missing letter detail at 16–20 px is missing in every channel, so greyscale cannot close the readability gap on its own.
- Caution: the standard grey conversion makes coloured marker lighter than black marker, which weakens contrast. Taking the darkest channel (the minimum of R, G, B) keeps any marker colour dark.
- For reading: PaddleOCR was trained on colour images. A grey image copied into 3 channels works, but the grey, normalised path helped 17 Oct (CER 0.116 → 0.095) and hurt the Airdrop board (0.157 → 0.169). It is not uniformly easier to read.

## 10. How much of the error is blur at all? The word-swap test (20 Oct)

*`scripts/word_swap_counterfactual.py` → `outputs/tpgsr_whiteboard/word_swap_*.csv`, `word_swap_summary.txt`, `word_swap_examples.png`. No model is trained: PaddleOCR is used as a black box (Lecture 8, counterfactual swap).*

**Method.** Every far line is cut into word boxes (from the far photo only, as in the app). PaddleOCR then reads:
- the far line as it is;
- the far line with **every word box replaced** by the same box from the aligned clear photo: what a *perfect* super-resolution of the words could reach;
- the whole clear line (to see how much error sits outside the word boxes: commas, underlines);
- the far line with **one word at a time** replaced, and with **all but one** replaced.

**Results** (mean line CER, lower = better):

| Set | Lines / words | Far line | Every word from the clear photo | Whole clear line | Gap a perfect SR could win |
|---|---|---|---|---|---|
| Airdrop B3 (unseen board) | 23 / 64 | 0.153 | **0.068** | 0.067 | **0.084 (55% of the error)** |
| 17 Oct, held-out photos | 37 / 184 | 0.120 | 0.110 | 0.129 | 0.010 (8%) |

(B3 has 23 lines here, not 24: one line yields no word box.)

**Where the gap is on B3: in small words.** Gain from replacing one word, by its height in the far photo:

| Word height | Words | Mean CER gain |
|---|---|---|
| up to 10 px | 3 | 0.000 (but leaving it blurry costs the most: +0.106) |
| 10–13 px | 16 | +0.033 |
| **13–16 px** | 13 | **+0.076** |
| 16–20 px | 6 | +0.016 |
| 20–32 px | 26 | −0.017 (noise) |

Word width does not matter (correlation +0.02). Fixing only the single most important word of a line recovers 69% of the gap.

**What it means.**
- On a far board, **blur in words under ~16 px is the main cause of misreading**. Above ~20 px, PaddleOCR reads the far word as well as the clear one.
- On 17 Oct the clear line reads *worse* than the far line in 12 of 37 lines and better in 11: its errors are the handwriting and the reader, not blur. The same comparison shows the **reader's own noise is about ±0.02 CER** between two photos of the same text; smaller differences should not be trusted.
- The remaining 0.068 on B3 with clear words ("Architectre", "Akxnet", "outpat", "CUN") is the reader's own mistakes on this handwriting: no image restoration can fix it.

**Caveats.** B3 is 6 different sentences photographed several times ("CNN Architecture" 8 times), so its 23 lines are thin evidence. Pasting clear boxes into the far line leaves faint seams; PaddleOCR may react to them.

## 11. TPGSR, pretrained on TextZoom and fine-tuned on our words (20–21 Oct)

*`scripts/finetune_tpgsr_whiteboard.py`, `notebooks/experiments/tpgsr_whiteboard_finetune.ipynb` → `outputs/tpgsr_whiteboard/summary.txt`, `models/superres/tpgsr_whiteboard_best.pt`.*

**Why this model.** Our own CNNs see about 17 px, less than one letter (`sr_debug_activations.ipynb`), and know nothing about letters. TPGSR (TSRN_TL generator) has bidirectional GRUs that run across the whole word, and a **text prior**: a frozen CRNN reads the word first and its letter probabilities guide the generator. It was pretrained by the team on TextZoom (~17k real text pairs).

**Set-up.**
- Word pairs: far word 16 × 64 → clear word 32 × 128 (TextZoom sizes), 2,676 pairs from the same line pairs as before.
- Size rules from section 10: training uses only words **10–32 px** tall in the far photo (710 of 2,149: 507 synthetic Airdrop, 203 17 Oct; under 10 px a person cannot read the word either, over 32 px the 16 px route only loses detail). Validation uses the 45 of 66 validation lines that contain such a word.
- Each batch: 8 of our words + 8 TextZoom words; loss as in the TextZoom training (L1 + 0.2 × gradient + 0.01 × soft prior); learning rate 5e-5; the epoch is chosen by **PaddleOCR line CER on validation** (epoch 0 = the pretrained model).
- At test time a **size gate** (only words under 16 / 20 / 24 / 32 px are restored), a blend weight and a confidence gate are all chosen on validation, then B3 is read once.

**Training.** Validation CER per epoch: 0.196 (pretrained), 0.204, 0.210, 0.196, 0.201 → early stop; **epoch 0 was kept**. The "fine-tuned" model *is* the pretrained one: fine-tuning on our words never improved reading, although the training loss fell (0.079 → 0.075).

**Result on B3** (23 lines; size gate chosen on validation: 16 px):

| Version | CER | vs raw [95% CI by photo] | Lines better / same / worse |
|---|---|---|---|
| raw input | **0.153** | | |
| 16px path, no network | 0.153 | −0.001 [−0.016, +0.016] | 2 / 19 / 2 |
| TPGSR (= pretrained) | 0.166 | −0.014 [−0.040, +0.004] | 1 / 18 / 4 |
| TPGSR, every word (no size gate) | 0.247 | −0.094 [−0.136, −0.059] | 2 / 7 / 14 |
| blend (alpha 0.3) | 0.159 | −0.007 [−0.025, +0.016] | 3 / 15 / 5 |
| confidence gate (chosen on validation) | 0.167 | −0.015 [−0.024, −0.005] | 1 / 18 / 4 |

**Verdict:** fails the rule fixed before training (0.167 is not lower than raw 0.153 or the 16px path 0.153; 1 line better, 4 worse). Every choice made on validation moved towards *using the network as little as possible*: the smallest size gate, the weakest blend, epoch 0.

**A side result that must not be claimed.** On 17 Oct, TPGSR applied to every word lowered CER from 0.120 to 0.102 (12 lines better, 9 worse): it *restyles* large words into something PaddleOCR prefers (the clear photo itself only gains 0.010). This was found by looking at the test set (validation chose the 16 px gate), rests on 3 photos, is close to the ±0.02 reader noise, and the same setting is much worse on B3 (0.247). At most: future work.

## 12. Model debug: why TPGSR did not help (21 Oct)

*`scripts/tpgsr_debug.py`, `notebooks/experiments/tpgsr_debug.ipynb` → `outputs/tpgsr_debug/`. Forward passes only. Main set: the 20 B3 lines whose label has one token per word box (54 words, all under 32 px), every word restored (we debug the network, not the gate).*

**First: does the prior's source read our words?** The frozen CRNN behind the text prior (trained on printed scene text, letters a–z and digits) reads the B3 far words at CER 0.509 (4% exact) and even the **clear** B3 words only at 0.258 (**28% exact**). It does not know this handwriting.

**Three hypotheses, each tested with a Lecture 8 tool:**

| Hypothesis | Test | Result | Verdict |
|---|---|---|---|
| The receptive field is too small (our CNNs' problem) | **Occlusion:** blank a 4 px strip of the input, measure the output change at every distance | 90% of the influence lies within 31 px (half the word); **26% beyond one letter** (8 px). Bicubic: 2 px, 0% | **Rejected**: the GRUs do carry information across the word (weak per pixel: ~20 grey levels at the strip, <1 one letter away) |
| The text prior is wrong | **Counterfactual priors** on the same words: CRNN on far word / CRNN on clear word / **true label** / **wrong label** / no letters | Word CER 0.317–0.329 and line CER 0.260–0.274 under all five; even the true label changes the picture by **~1 grey level** on average; the wrong label does not hurt; PaddleOCR reads "Saleacce" under every prior | **Rejected**: the network ignores its prior, so a better prior (e.g. a CRNN trained on our handwriting) would not help |
| The pixel loss rewards cleaning, not rebuilding letters | **Where the output changes** vs the 16px path, split by the clear word's strokes | Strokes (15% of pixels): +1.3 lighter, error 48.7 → 49.4 (worse). Next to strokes (27%): +3.8 lighter, 22.8 → 21.9. Board (58%): 16.3 → 15.7 | **Confirmed**: the same shortcut as M1 (section 8): the halo next to strokes is cleaned, the strokes are not rebuilt |

Inside the network, the true label changes the features of blocks 2–4 by at most ~5% of their size and only 1.4% by block 6: the prior enters and is washed out. The colour tint is not the cause either: about 1 grey level on the B3 board, and removing it moves B3's CER from 0.274 to 0.271.

**A correction to section 11's "16px path = raw".** That holds with the 16 px gate. When every B3 word under 32 px is restored, the 16 px route alone already costs 0.165 → 0.189 on these 20 lines, and TPGSR adds more on top (0.274).

**Conclusion.** Neither a global receptive field nor a text prior overcame a problem that comes from the **training objective** (a pixel loss is won by cleaning the 85% of pixels that are board or halo), and none of the tested restorers recovered the letter detail that 9–16 px words lose. That is evidence that missing detail is a bottleneck, not proof that no single-image method could do better: a restorer can bring in letter shapes learned from its training data, and ours saw little handwriting. One gap in this test: our prior reader stayed frozen (TextZoom training and fine-tuning alike), whereas the original TPGSR also trains it; the counterfactual priors suggest this was not the limiting factor here (even the true label barely changed the output). The "use the hint" route also has a catch: at test time the prior must come from reading the blurry word, and PaddleOCR already reads it better (line CER 0.165) than the CRNN prior does (word CER 0.509). **The tested super-resolution methods did not consistently improve recognition on the unseen board; a clearer capture was more effective** (section 13.1: retaking the small lines, simulated with the clear photos, lowers B3 from 0.153 to 0.088). A reading loss was tried as well (M2, section 8b), with a weak examiner; a version with PP-OCRv6 as the examiner, the clear target kept and the epoch chosen on reading remains possible, tested on a fresh board. Burst (multi-photo) super-resolution, which adds new measurements of the writing rather than reinterpreting one photo, is future work; its text-level version, fusing the readings of several photos, passes its test (section 13.4).

## 13. What replaces super-resolution (21 Oct)

**13.1 A "move closer" prompt at capture time** (`scripts/capture_check.py`, `capture_advice(line_bgr, scale=...)` and `photo_advice(...)`). The blur that matters (section 10) can be avoided instead of repaired.
- **Measure: text height**, the height of the line's ink band in pixels of the original photo, not the crop height (crops carry padding: the far B3 crops are 21–39 px tall, their writing 5–22 px). Underlines and table rules are removed first, the block of rows with the most ink is kept (a slanted underline can be the single inkiest row), and the band is measured on unsmoothed rows so the result does not depend on the crop's resolution. Checked on synthetic crops (underlines, touching rules, specks, 1 px strokes, 3× enlargement).
- **Threshold: 12 px.** On ~1,000 line crops cut from the original photos, PaddleOCR's CER rises steadily as text shrinks and is about double below 12 px (Airdrop 0.27 vs 0.15 for the rest; 17 Oct 0.21 vs 0.12; the 17 Oct crops flagged are all from its far and side-angle photos). On B3 (word-swap lines), the 10 lines under 12 px lose +0.148 CER each to blur, the other 13 only +0.035; if just those 10 were retaken as close as the clear photo, B3 would fall from 0.153 to **0.088**. None of the 37 17 Oct test lines is flagged.
- The photo-level message appears when at least 25% of the lines are too small (a UI choice, not calibrated). `python scripts/capture_check.py --calibrate-size` reproduces all numbers. The B3 figure is optimistic (these are test lines and a retake is assumed to be as good as the clear photo).

**13.2 Vocabulary correction of the text** (`scripts/vocab_correction.py`). Knowledge the photo does not contain (which words the lecture uses) is added after reading. Rule fixed beforehand: a word not in the vocabulary is replaced by the closest vocabulary word within 1 edit (3–4 letters) or 2 edits (5+ letters); shorter words and numbers are kept. No vocabulary is built from test labels.

| Readings | Vocabulary | CER (lower = better) | Lines better / worse | Words fixed / broken |
|---|---|---|---|---|
| B3, 23 far lines | none | 0.153 | | |
| | **Lecture 8 slides** | **0.121** | 14 / 3 | 14 / 3 |
| | slides + general English (IAM labels) | 0.133 | 13 / 4 | 13 / 3 |
| | general English only | 0.184 | 5 / 11 | 8 / 13 |
| 17 Oct, 37 far lines | general English | 0.120 → 0.158 | 2 / 21 | 5 / 15 |
| ~1,000 line crops | general English | Airdrop 0.172 → 0.222, 17 Oct 0.154 → 0.192 | | about 3 broken per fixed |

**Topic vocabulary helps; general English hurts**, because course notes are full of technical words ("CMN" → "Can", "ousian" → "russian"). One real failure even with the slides: they spell "visualize", so the correct British "Visualising" becomes "Visualizing". A natural fix (correct only *towards* slide words, use general English only to leave words alone) was not tried: it would be tuned on the test board and needs a fresh one. In the app the user supplies the lecture's slides: the same code works for any subject (a scalability point).

**13.3 Fine-tuning the reader** (`scripts/finetune_reader.py`, `notebooks/experiments/reader_finetune.ipynb`, run on a rented RTX 4090 with `scripts/runpod/run_cloud.sh`). The 0.068 that B3 keeps even with clear words, and almost all of 17 Oct's error, are the reader's own mistakes on marker handwriting. The pipeline's reader PP-OCRv6_medium_rec is fine-tuned from PaddleOCR's pretrained weights on 434 of our line crops (B1, B4, B5, 17 Oct outside val/test; labels of at most 25 characters, each crop 4× per epoch) plus 307 IAM lines, 20 epochs, batch 64. Batch-norm statistics stay frozen and crops are trained at 48 × 320: two fixes found in Mac smoke tests, without which training collapses into repeated letters ("e e ee": tiny batches overwrite the batch-norm statistics, and in training mode the model has only 40 output steps, too few for long labels). Three settings, each kept at its best epoch and compared on validation (B2 + the 17 Oct validation photos, 46 lines); **B3 (90 crops, 7 sentences, 19 photos) is the test, read once**; IAM test checks that general handwriting is kept. Pass rule fixed before training: lower B3 CER than the pretrained reader, more crops better than worse, IAM test CER up by at most 0.02.

| Setting | Validation CER (pretrained 0.034) | Lines better / worse |
|---|---|---|
| A: learning rate 1e-4, all layers | 0.012 | 12 / 2 |
| **B: learning rate 2e-5, all layers (chosen)** | **0.010** | 13 / 0 |
| C: learning rate 1e-4, backbone frozen | 0.014 | 11 / 1 |

| Test set (setting B) | Pretrained → fine-tuned CER | Lines better / same / worse |
|---|---|---|
| **B3, unseen board (decides)** | **0.125 → 0.140** (gain, 95% CI by photo: −0.033 to +0.002) | 29 / 31 / 30 |
| 17 Oct test photos | 0.000 → 0.000 | 0 / 3 / 0 |
| IAM test (2,915 lines) | 0.177 → 0.121 | 1,200 / 690 / 1,025 |

**Verdict: fails.** On the board it has never seen, the fine-tuned reader is no better than the pretrained one (slightly worse, within noise: as many lines got worse as better, and exact lines fell 12 → 8). Forgetting is not the problem: IAM improved, but IAM train and test come from the same dataset, so that gain says little about whiteboards. The pipeline keeps the pretrained reader.

**Why validation was misleading.** The validation set was split by photo, not by text: 14 of its 46 lines (10 of 20 phrases) appear word for word in the training set, while B3 shares none. The training set is also repetitive: 434 crops, but only 150 distinct phrases, each seen many times per epoch. So the validation gain (0.034 → 0.010) partly measures recognising phrases already seen rather than reading new handwriting, and choosing among A, B and C by it does not predict B3. The test board did its job: it is the only set that asks the question the app faces (new notes). The B3 readings that changed show the same character confusions as before, just moved around ("Visualising CNN" → "Visualising cuw" pretrained, "Visualising cun" fine-tuned).

A fair retry would need a validation set held out **by board** (text never seen in training) and more distinct text, not more photos of the same phrases; with one test board, it would also need a fresh board to test on, since B3 has now been read. (The pretrained B3 CER here, 0.125 over all 90 crops, differs from the 0.153 in 13.1–13.2, which covers only the 23 far-photo word-swap lines.)

*Caveat found on 22 Oct: every crop in these lists (train, validation and test) was kept by the crop-making scripts (`airdrop_rec_paddle/make_airdrop_rec_crops.py`; the 17 Oct crops follow the same accepted / review rule) only if PP-OCRv6 already read it with CER ≤ 0.25. Crops read at 0.25–0.5 went to `review/` and were not used; worse ones were never matched to a label. So the test crops are the ones the pretrained reader found easy (on B3, 90 accepted crops were used and 46 review crops left out), which understates the errors and tilts any comparison towards the pretrained reader that did the selecting. The fine-tune verdict holds for these crops only; a fair retest needs crops matched to their labels without the reader (by position on the board) and checked by a person.*

**13.4 Several photos instead of one: reading fusion** (`scripts/fusion_check.py`, rules in `scripts/reading_fusion.py`). Super-resolution reinterprets one photo; another photo adds new measurements of the same writing. Each photo's own crop of a line is read by PP-OCRv6 and the readings are combined at the text level, after Nascimento et al. 2025 (licence plates from several video frames): keep the most confident reading (hc), the most frequent one (mv), the one closest to all others (medoid), or vote character by character after aligning them (ROVER-style). Crops: all 573 labelled line crops of boards B1–B5 and 17 Oct (the pretrained reader never trained on any of them). Rule fixed before reading: the user takes 3 photos (random 3-photo subsets of each line, averaged); the combining rule is chosen on all boards except B3; B3 decides (gain over one photo > 0.02 and more lines better than worse).

| Lines (mean line CER) | 1 photo | hc | mv | medoid | vote | best photo (oracle) |
|---|---|---|---|---|---|---|
| Other boards, 3 photos (77 lines; chooses the rule) | 0.049 | **0.021** | 0.023 | 0.023 | 0.023 | 0.012 |
| **B3, 3 photos (7 lines; decides)** | 0.128 | **0.080** | 0.080 | 0.096 | 0.089 | 0.067 |
| B3, 2 photos | 0.128 | 0.097 | 0.097 | 0.097 | 0.097 | 0.087 |
| Text under 16 px in every photo, 3 photos (12 lines, all boards) | 0.114 | 0.079 | 0.079 | 0.091 | 0.082 | 0.057 |

**Verdict: passes.** Keeping the most confident of three readings lowers B3 from 0.128 to **0.080**, all 7 lines better, close to the best possible choice (0.067). The simplest rule won; voting letter by letter added nothing. The gain holds when every photo shows small text (0.114 → 0.079), so it is not only that one of the photos happened to be close.

Caveats: (1) which crops show the same line was known from the labels; the app must match lines across photos (box overlap after perspective correction, or text similarity), and matching mistakes will cost some of the gain; (2) the photos of a board were taken from different positions; three quick shots from one spot are more alike and should gain less; (3) B3 has 7 lines; (4) these CERs average every crop of a line, so they are not comparable with the 23 far-line figures of 13.1–13.2 (0.153).

In the app: ask for 2–3 photos of the board (or a short burst), read each, and keep the most confident reading of every line. It also helps where moving closer is not possible (e.g. a seat at the back of the room).

*Same caveat as 13.3: these crops were pre-selected by PP-OCRv6 (CER ≤ 0.25). Both arms of the comparison use the same reader, so the direction of the fusion gain is likely robust, but its size on unselected crops (where single photos are read worse) is not known.*
