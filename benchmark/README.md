# RFS-SRD internal fivefold benchmark

This directory reproduces the internal benchmark on the released 500 images. It includes the six detector implementations through Torchvision and Ultralytics, training and checkpoint selection, validation/test inference, and common metric computation. All class IDs match `../classes.txt`.

## Environment

The experiment used Linux, Python 3.9.23, PyTorch 2.1.0, Torchvision 0.16.0, CUDA 12.1, Ultralytics 8.4.51, NumPy 1.26.4, and Pillow 11.3.0. Install the CUDA-compatible PyTorch build for your machine, then the versions in `requirements.txt`. A CUDA GPU is required for training. Exact package and initialization-weight records are in `manifests/training_environment.json`. No model checkpoints are included in this archive.

## Protocol

- Five template-group-disjoint folds, with 300 training, 100 validation, and 100 test images per fold. Every damage category contributes 60/20/20 images, respectively.
- Adjacent image pairs within each category share a damage template and remain in the same role within a fold. Every image is tested once per seed.
- Three seeds: 20261112, 20261113, and 20261114. All six architectures use the same fixed folds and 640-pixel input dimensions.
- COCO initialization. Torchvision models train for 30 epochs with batch size 8. YOLO models and RT-DETR train for up to 100 epochs with patience 20; batch sizes are 16 for YOLO and 8 for RT-DETR. These are the settings used for the reported benchmark, not equalized training budgets.
- Checkpoints are selected on validation data. One confidence threshold per model/fold/seed maximizes validation macro-F1 over 0.01 to 0.99; ties favor the higher threshold. Test data do not select thresholds.
- Operating metrics use class-aware, score-ordered one-to-one matching at IoU 0.50. AP uses 101-point interpolation; AP50-95 averages IoU 0.50:0.05:0.95. False-alarm proportion is FP/(TP+FP), and missed-detection rate is FN/(TP+FN). FPPI counts all false positives across classes per image.
- Class-macro values are averaged over folds per seed, then over the three seeds. Reported F1 SD is the sample SD of the three seed-level means. Template grouping is documented; the source-background grouping limits are recorded in `manifests/split_audit.json`.

## Prepare and train

Run from the extracted `RFS-SRD` data directory. If using the separate GitHub repository, run the commands in its root README and pass the extracted data directory with `--dataset`. The preparation command creates a separate experiment directory and a dataset symlink (Linux is recommended).

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
python benchmark/prepare_cv.py --dataset . --output ../rfs_srd_cv
python benchmark/run_cv.py --root ../rfs_srd_cv --gpus 0
python benchmark/evaluate_cv.py --root ../rfs_srd_cv --workers 4
```

For multiple GPUs, pass their physical IDs, for example `--gpus 0,1,2,3,4,5,7`. The original experiment excluded GPU 6. The launcher resumes completed runs. Use `--models yolo11m,yolov10m` or `--seeds 20261112` for a subset, but the full evaluation command requires all six models, five folds, and three seeds.

Ultralytics initialization files are `yolo11m.pt`, `yolov10m.pt`, and `rtdetr-l.pt`. Supply their directory with `--pretrained-dir /path/to/coco_weights`; otherwise the framework resolves the named weights. Torchvision uses its official pretrained-weight cache/download mechanism. An internet connection is needed if initialization weights are not already cached.

To train one task:

```bash
python benchmark/train_cv.py --root ../rfs_srd_cv --model yolo11m --fold 0 --seed 20261112 --device 0
```

Training automatically saves the selected checkpoint and both `predictions_val.json` and `predictions_test.json` in `runs/<model>/fold_<fold>_seed_<seed>/`.

## Regenerate validation or test predictions without retraining

```bash
python benchmark/predict_cv.py --root ../rfs_srd_cv --model yolo11m --fold 0 --seed 20261112 --split val --device 0
python benchmark/predict_cv.py --root ../rfs_srd_cv --model yolo11m --fold 0 --seed 20261112 --split test --device 0
python benchmark/evaluate_cv.py --root ../rfs_srd_cv --workers 4
```

Inference retains low-confidence candidates for AP calculation; `evaluate_cv.py` applies the validation-selected operating threshold. Box coordinates are absolute xyxy values in the original image orientation and resolution. No rotation or mirroring is applied during test-image loading.

## Files

| File or directory | Purpose |
|---|---|
| `prepare_cv.py` | Materialize the fixed image lists and model data files |
| `train_cv.py` | Train one model/fold/seed, select its checkpoint, export predictions |
| `run_cv.py` | Schedule the full 90-task experiment |
| `predict_cv.py`, `model_loader.py` | Reload selected checkpoints and regenerate predictions |
| `evaluate_cv.py`, `metrics.py` | Select validation thresholds and calculate test metrics |
| `manifests/` | Fixed experimental splits, split audit, and environment record |
| `reference_results/` | Previously computed fold-, seed-, and class-level benchmark tables |

New outputs are written to the experiment directory; `reference_results/` is provided for comparison. External real-world experiments and their data are not included in this internal-benchmark code package.
