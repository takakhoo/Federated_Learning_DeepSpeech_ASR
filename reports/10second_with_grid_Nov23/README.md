# 10-second DS1 Reconstruction with Grid (Nov 23)

This folder packages the exact artifacts from the Nov 23 DeepSpeech1
reconstruction job that used Minh's original `batched_ctc_v2` CTC loss. Both
the teacher gradients and the meta-loss shared that implementation, so the
objective matches Minh's paper setup.

- **Model & Loss**: DeepSpeech1 with MFCC features and ±6 context frames.
- **Grid Setup**: `--use_grid_optimization` with `grid_size=300`,
  `grid_overlap=150` (stride 150) to chunk the 627 frames over four windows.
- **Learning-Rate Reset**: Each grid resets Adam to `lr=0.05` and restarts the
  MultiStepLR milestones to keep later grids stable.
- **Iterations**: 4 grids × 1000 first-order steps (no zero-order stage).

Included files:

- `sampleidx_0_grid_concat_firstorder.png` – full reconstructed spectrogram vs
  ground truth plus loss traces.
- `sampleidx_0_grid_grid{0-3}_iter900*.png` – late-iteration snapshots for each
  grid.
- `sampleidx_0_grid_loss_curves_0.png` – auto-generated loss curves from the
  run directory.
- `experiment.log` – raw console output for reproducibility.

### Reproduce the run

From `/scratch2/f004h1v/recon` with the `fl_ds1` conda env active:

```
python src/main.py \
  --model_name ds1 \
  --dataset_path ../local_datasets/librispeech_long_10s \
  --batch_start_idx 0 --batch_end_idx 1 \
  --min_duration_ms 0 --max_duration_ms 13000 \
  --learning_rate 0.05 --max_iterations 4000 \
  --use_grid_optimization --grid_size 300 --grid_overlap 150 \
  --context_frames 6 --dropout_prob 0.0
```
