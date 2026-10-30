# Board2Notes: OCR pipeline and super-resolution experiments

## Start here

**For the application, use [the standalone team notebook](pipeline/notebooks/10_board2notes_team_pipeline.ipynb).**
Read [the setup and output guide](pipeline/README.md), install the dependencies,
choose the photo paths and recognizer, then run the notebook from top to bottom.
It embeds the implementation and does not import project Python helpers.

The default is detection, line cropping and recognition. Optional rotation,
bicubic enlargement, perspective and lighting modes are experimental and selected
one at a time. The notebook saves raw text, JSON, numbered boxes and the actual
recognition crops for integration with the team's LLM and slide converter.
SR is not enabled in this pipeline. No optional correction is claimed to improve
every photo. The original development notebook 08 remains in the local research
workspace and has not been removed or replaced.

## Research and presentation

- [SR findings for the presentation](sr_research/PRESENTATION_NOTES.md): methods,
  results and limitations in one place.
- [SR archive guide](sr_research/README.md): seven experiment notebooks,
  supporting source, recorded measurements and selected comparison figures.
- [Historical experiment journal](sr_research/docs/SUPER_RESOLUTION_RUNS.md): the
  original detailed chronology, including later alternatives to SR.
- [Source manifest](SOURCE_MANIFEST.json): source and packaged SHA-256 hashes;
  records the reduction of notebook image outputs for Git.

Research notebooks are archived experiments, not an additional application step.
Their text outputs and result tables are retained; selected comparison images are
separate PNG files. Re-running research requires the original datasets,
checkpoints and third-party research code described in the archive guide.
Those assets are distributed separately from Git.

## Validation

From this repository's root, in an environment with `nbformat` and IPython:

```bash
python board2notes/verify_handover.py
```

This checks the packaged hashes, notebook schemas, Python syntax (including the
embedded pipeline module bodies), and numerical claims in the presentation
summary. It does not download models or rerun training. See the pipeline guide
for the distinction between the existing real OCR smoke check and a new install.
