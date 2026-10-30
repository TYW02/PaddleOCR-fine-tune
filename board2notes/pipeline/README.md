# Board2Notes teammate notebook

Open **`10_board2notes_team_pipeline.ipynb`** (in `notebooks/` in the repository). Notebook 08 remains
the experiment/development notebook and is not changed by this handover.

The notebook embeds its entire pipeline implementation and imports no
project Python files. **Share the `.ipynb` alone.** No `preprocessing/`
folder is required. Advanced source cells are folded where supported;
expand them to read the code. Internal modules exist only in notebook memory.

The optional **`board2notes_team_pipeline.zip`** in Assignment contains the
same standalone notebook, these instructions and a requirements file. It
contains no `.py` helpers, photographs, datasets or model weights.

## Existing project environment

Choose the working Python 3.12 OCR kernel, edit the notebook's settings cell,
and run the cells from top to bottom. Models load once and are reused across
photos. The first use of a pretrained model downloads its weights if they
are not already cached. A downloaded/exported custom reader is used locally.

## Fresh CPU environment

From the extracted `board2notes_team_pipeline` directory:

```bash
python3.12 -m venv .venv_team
source .venv_team/bin/activate
python -m pip install --upgrade pip
python -m pip install paddlepaddle==3.3.1
python -m pip install -r requirements_team.txt
python -m ipykernel install --user --name board2notes-team --display-name "Board2Notes team"
python -m jupyterlab
```

On Windows, use `.venv_team\Scripts\activate` instead of the `source` line.
For an existing NVIDIA environment, retain its verified PaddlePaddle GPU
installation; do not install the CPU package over it. The requirements file
deliberately leaves the PaddlePaddle CPU/GPU choice to the environment.
The fresh install commands are a setup recipe; the handover was executed
with the existing working local environment, not freshly installed on
every operating system.

## Settings teammates edit

- `PHOTO_PATHS`: a list of photo files or folders. Folder input includes
  supported images directly inside that folder, without recursion.
- `OUTPUT_DIR`: where results should be written.
- `RECOGNIZER_NAME` and `RECOGNIZER_DIR`: select the reader.
- `PREPROCESS`: `baseline` by default. Select one correction at a time.
- `DIRECTIONAL_GROUPING`: `False` by default; an optional development candidate.

For the team's exported PP-OCRv5 reader, set **both**:

```python
RECOGNIZER_NAME = "PP-OCRv5_server_rec"
RECOGNIZER_DIR = Path("/your/path/to/exported_model_folder")
```

Use the actual architecture name corresponding to your export. The folder
needs `inference.pdiparams`, `inference.yml` and `inference.json` (or the
older `inference.pdmodel`). Training checkpoints such as `best_accuracy`
are not exported inference folders. A missing/incomplete folder stops with
an error; the notebook does not silently substitute a pretrained reader.

## What runs

Photo → writing detection → selected optional correction → line crops → OCR.
Photos are decoded at native size, with EXIF orientation applied. Detection
can use a smaller copy and retry text regions at native resolution. The
saved reader crops come from the decoded full-resolution image or selected
correction, rather than a display preview.

Default `baseline` matches the isolated baseline used by the preprocessing
audit: common writing-region crop policy, unclip 1.2, legacy line grouping
and no optional enlargement. The recognizer still performs its required
input resize internally.

| PREPROCESS | Behavior |
|---|---|
| `baseline` | Common detection/cropping only |
| `rotation` | Small-roll correction only when independent text rows agree |
| `enlargement` | Bicubic enlargement of small line crops only, with a scale cap |
| `perspective` | Experimental physical-board geometry and readability checks; uncertain cases skip |
| `lighting` | Experimental dim/contrast/uneven-light rule and existing lighting operator |

The chosen mode can skip correction if its rule does not pass. The decision
and signals are exported. Rotation is not a 90°/180° orientation detector.
Perspective may skip a real side view when reliable board edges or focal
metadata are unavailable. Lighting can harm recognition. Directional
grouping improved some development groups but regressed B3, so it is not
enabled by default. No combination is claimed to be the winning pipeline.
SR, sharpening and blemish removal are excluded from this interface.

## Outputs

Every photo creates a new folder; earlier results are retained:

```text
outputs/team_pipeline/<photo>_<time>_<id>/
    ocr.txt          # raw recognized text in reading order; ready for the LLM step
    ocr.json         # text, line boxes, confidence, settings and decision log
    processed.png   # exact image used to cut the lines
    annotated.jpg   # numbered line boxes for visual checking
    crops/          # actual images fed to recognition
```

JSON boxes use **processed.png coordinates**, not the original photo's
coordinates. Detection box errors and incomplete lines can remain; inspect
the numbered boxes and crops before trusting the transcript. Confidence is
not accuracy, and CER is not calculated without verified labels. The
notebook neither trains a model nor changes a training dataset.

After the embedded notebook cells have run, the same API is available:

```python
engine = TeamPipeline(Settings())  # load once
result = engine.process_photo("photo.jpg", "outputs/team_pipeline")
print(result["text"])
```

The notebook can be moved to any folder and run with installed dependencies,
photo paths and optional custom reader weights. It does not search for the
original repository. Models remain external assets because pretrained models
are downloaded/cached by PaddleOCR and custom exports are selected by path.
