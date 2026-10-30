# Models

Nothing here is committed to git.

- **PaddleOCR models** (`PP-OCRv5_mobile_det`, `PP-OCRv6_medium_rec`, ...) download automatically on first use to `~/.paddlex/official_models/`.
- **Super-resolution models** (only for the upscaling comparison in the experiments notebook; the main pipeline uses bicubic + sharpening and does not need them). Download into `models/superres/`:

```bash
mkdir -p models/superres && cd models/superres
for s in 2 3 4; do
  curl -fLO https://raw.githubusercontent.com/fannymonori/TF-ESPCN/master/export/ESPCN_x$s.pb
  curl -fLO https://raw.githubusercontent.com/Saafke/FSRCNN_Tensorflow/master/models/FSRCNN_x$s.pb
  curl -fLO https://raw.githubusercontent.com/Saafke/EDSR_Tensorflow/master/models/EDSR_x$s.pb
done

# Real-ESRGAN x4plus (generative/adversarial SR comparison)
curl -fLO https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
```

`super_resolution_pretrained.ipynb` handles the compatibility shim needed by newer
`torchvision` releases and loads this weight on CPU. Real-ESRGAN is an experiment,
not part of the main OCR pipeline unless its CER and speed justify enabling it.

- **PaddleOCR text-specific SR models** (Text Gestalt and Text Telescope). From the
  `whiteboard-ocr` repository root, download the official research source and checkpoints:

```bash
git clone --depth 1 https://github.com/PaddlePaddle/PaddleOCR.git ../models/PaddleOCR-text-sr
git -C ../models/PaddleOCR-text-sr fetch --depth 1 origin dab3fe35379033fdcb2d0e9572fac0b36c9a9ebf
git -C ../models/PaddleOCR-text-sr checkout dab3fe35379033fdcb2d0e9572fac0b36c9a9ebf
mkdir -p ../models/superres/text_gestalt ../models/superres/text_telescope
curl -fL https://paddleocr.bj.bcebos.com/sr_tsrn_transformer_strock_train.tar -o /tmp/TextGestalt_train.tar
curl -fL https://paddleocr.bj.bcebos.com/contribution/Telescope_train.tar.gz -o /tmp/TextTelescope_train.tar.gz
tar -xf /tmp/TextGestalt_train.tar -C ../models/superres/text_gestalt sr_tsrn_transformer_strock_train/best_accuracy.pdparams
tar -xzf /tmp/TextTelescope_train.tar.gz -C ../models/superres/text_telescope Telescope_train/best_accuracy.pdparams Telescope_train/config.yml
```

These models use PaddleOCR's official fixed-size word preprocessing: 16x64 input
to 32x128 output. The experiment deliberately records this limitation when applying
them to our longer handwritten line crops.
- **YOLO whiteboard segmentation** weights are produced by `notebooks/experiments/whiteboard_segmentation_yolo.ipynb` into `runs/` (experiment only, not used by the pipeline).

- **Handwriting line upscaler** (Experiment 2 in `notebooks/experiments/super_resolution_trained.ipynb`):
  `models/superres/handwriting_sr_best.pt` (PyTorch state dict of `LineSR`) and `handwriting_sr_best.json`
  (training settings, kept epoch, validation history). Produced by the notebook; nothing to download. The notebook
  reuses the file only if the settings in the json match its current settings, otherwise it trains again.
