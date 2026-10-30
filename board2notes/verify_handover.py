"""Validate this handover without fetching data, weights, or running training."""

import csv
import hashlib
import json
from pathlib import Path
from statistics import mean

import nbformat
from IPython.core.inputtransformer2 import TransformerManager


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'sr_research/results'


def b3_rows(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return [row for row in csv.DictReader(stream) if '__B3_' in row['pair_id']]


def main():
    manifest = json.loads((ROOT / 'SOURCE_MANIFEST.json').read_text())
    for item in manifest['files']:
        path = ROOT / item['archived']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['archived_sha256'], path
    print(f"PASS: {len(manifest['files'])} archived file hashes")

    transform = TransformerManager()
    notebooks = list(ROOT.rglob('*.ipynb'))
    for path in notebooks:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != 'code':
                continue
            source = cell.source
            if source.startswith('%%team_module '):
                _, source = source.split('\n', 1)
            else:
                source = transform.transform_cell(source)
            compile(source, f'{path.name}:cell{index}', 'exec')
    for path in ROOT.rglob('*.py'):
        compile(path.read_text(), str(path), 'exec')
    print(f'PASS: {len(notebooks)} notebook schemas and all Python cell/module syntax')

    # Recompute the small table from the actual paired-view measurements.
    comparisons = {
        '_sr_review/run3_ink_matched/accepted_model_ocr_comparison.csv': .154,
        '_sr_review/run4_pixel_loss/accepted_model_ocr_comparison.csv': .174,
        'paired_dataset_17oct/m1_stroke_loss_ocr_comparison.csv': .168,
        'paired_dataset_17oct/m2_recognition_loss_ocr_comparison.csv': .181,
    }
    for name, expected in comparisons.items():
        rows = b3_rows(RESULTS / name)
        assert len(rows) == 24, (name, len(rows))
        assert round(mean(float(row['raw input CER']) for row in rows), 3) == .157
        assert round(mean(float(row['input, no CNN CER']) for row in rows), 3) == .169
        actual = mean(float(row['restored CER']) for row in rows)
        assert round(actual, 3) == expected, (name, actual, expected)
    for tag, expected in [('run3_ink_matched', .41), ('run4_pixel_loss', 1.36)]:
        rows = b3_rows(RESULTS / '_sr_review' / tag / 'accepted_model_psnr_full_canvas.csv')
        assert len(rows) == 24
        actual = mean(float(row['gain_ink']) for row in rows)
        assert round(actual, 2) == expected, (tag, actual)
    print('PASS: presentation paired-view CER and stroke PSNR values recomputed')

    for path in ROOT.rglob('*'):
        if path.is_file():
            assert path.suffix.lower() not in {'.pt', '.pth', '.pdparams', '.pdiparams', '.mdb', '.zip', '.heic'}, path
            assert path.stat().st_size < 5 * 1024**2, path
    print('PASS: no checkpoints, dataset archives or original camera photos; files under 5 MiB')
    print('HANDOVER CHECK PASSED (static checks and saved metrics; no training rerun)')


if __name__ == '__main__':
    main()
