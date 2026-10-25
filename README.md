# RGB and depth manipulative gesture recognition

Train RGB and depth independently using **TSM** or **MVTN**, with the same
gesture vocabulary and subject/repetition protocols as the IMU/sEMG project.
Root entry points are `train_rgb.py` and `train_depth.py`.

## Dataset and portability

```text
Manipulative_dataset/
  dataset/                         # READ ONLY
    <subject>/
      annotation/rgb_depth_gesture_segments*_EN.csv
      rgbd/session1/front/color.avi
      rgbd/session1/front/timestamps.csv
      rgbd/session1/front/depth_raw/000001.png
      ... session1/session2/session3; front/above/left/right
  rgb_d/
    train_rgb.py
    train_depth.py
    configs/
    scripts/                       # annotations, labels, protocols, CLI, reports
    vision/                        # video/depth loading, model adapters, training
    models/tsm/                    # original upstream project, preserved
    models/mvtn/                   # original upstream project, preserved
    tests/
    runs/                          # every generated artifact
```

The dataset path is `../dataset` **relative to the repository**, independent of
the terminal's working directory and the computer's drive letter. Source CSVs,
videos and depth images are never modified. All experiment output stays under
`runs/rgb/` or `runs/depth/`; run names cannot overwrite an existing directory.

`configs/dataset.json` maps `step1`, `step2`, `step3` to recording folders
`session1`, `session2`, `session3`. Session4 is not selected. Above-view times
come from `start_time/end_time`, front from `start_time2/end_time2`. The original
CSV line is retained in each parent/sample ID. Whitespace is stripped only in
memory. Left/right views are supported by the loader, but the current CSVs have
no timings for them, so selecting those views produces a preparation error.

## Environment

Activate your training environment and install its appropriate CUDA-enabled
PyTorch build if GPU training is intended. Then install the video dependencies:

```powershell
python -m pip install -r requirements-video.txt
python -c "import torch, cv2; print(torch.__version__, torch.cuda.is_available(), cv2.__version__)"
```

The adapters use PyTorch directly and do not require torchvision or einops.
`--device auto` uses CUDA when available; `--device cuda` fails clearly if it is
unavailable. The verified runtime is Python 3.11 / PyTorch 2.7 with CUDA 12.8.

## Begin with preparation

Run these commands from this repository:

```powershell
python train_rgb.py --protocol configs/protocol2.json --model tsm --views front --prepare-only
python train_rgb.py --protocol configs/protocol1.json --model mvtn --views front above --prepare-only
```

Preparation validates selected annotations, source availability, container
metadata, endpoint decoding, class coverage and splits. It saves manifests and
sampled frame indices without running optimization or extracting complete
videos. It does not decode every frame or guarantee the visual correctness of
every label. For depth, it verifies selected depth paths and probes a depth
image per recording.

The accepted source-time overlaps remain **warnings**, including intersections
between the same participant's training and test repetitions. This is explicit
in `data_quality.allow_within_subject_test_overlap: true` in both modality
configs. Set it to false for strict temporal checks. Warnings remain in
`issues.json` and are counted in the terminal. Duplicate record IDs and trial
assignment across partitions remain errors. Raw cross-participant pixel copies
are not content-hashed by this baseline. Accepted overlaps may affect the
interpretation of test scores; they are retained as requested.

## Train RGB

```powershell
# TSM, one camera, held-out repetitions
python train_rgb.py --protocol configs/protocol2.json --model tsm --views front --epochs 50 --device cuda

# MVTN, above camera
python train_rgb.py --protocol configs/protocol2.json --model mvtn --views above --epochs 50 --device cuda

# Two cameras, shared model weights and averaged class logits
python train_rgb.py --protocol configs/protocol2.json --model tsm --views front above --device cuda

# One unseen-participant fold
python train_rgb.py --protocol configs/protocol1.json --model mvtn --views front --test-subject 1Alice --device cuda

# All eight unseen-participant folds
python train_rgb.py --protocol configs/protocol1.json --model tsm --views front --device cuda
```

Defaults are 8 frames per view, 128x128 input, batch size 4, 50 maximum epochs,
AdamW at 1e-4, and automatic device selection. Override these with `--frames`,
`--image-size`, `--batch-size`, `--epochs`, `--device` and `--workers`.
Worker count defaults to zero for straightforward Windows execution. If GPU
memory is insufficient, reduce batch size first, then input size/frame count.

**Initialization is random by default.** For a local standard ResNet18 backbone
state dictionary, use `--backbone-weights runs/pretrained/resnet18.pth`.
No weights are downloaded automatically. Only the backbone is initialized from
that file; the gesture classifier and MVTN attention layers start fresh. A full
training checkpoint `best.pt` is a different format; this option is not resume
training. Training resume is not implemented in this baseline.

## Train depth separately

```powershell
python train_depth.py --protocol configs/protocol2.json --model tsm --views front --prepare-only
python train_depth.py --protocol configs/protocol2.json --model tsm --views front --device cuda
python train_depth.py --protocol configs/protocol1.json --model mvtn --views front above --test-subject 1Alice --device cuda
```

Depth uses the same CSV parents, protocols, camera choices and model switches.
It reads native single-channel uint16 PNG values, then clips/scales using
`input.depth_min` and `input.depth_max` in `configs/depth.json`. The initial
range is **0 to 5000 raw recorded units**; it does not assert these values are
millimetres. Zero remains the missing-depth value. Nearest-neighbour resizing
avoids averaging valid depth with missing pixels. The input layer has one
channel; a supplied three-channel ResNet18 conv1 is averaged for this layer.

## Sampling and model behavior

- Videos are decoded lazily; there is no required frame-extraction stage.
- Each ordinary gesture yields one clip. Fixed temporal bin centers select
  frames whose playback timestamps are in `[start,end)`. Short intervals repeat
  frame indices when needed. No temporal/spatial augmentation is enabled in
  this first baseline; RGB is resized and uses fixed ImageNet mean/std.
- Each screw parent yields five evenly spaced, nonoverlapping approximately
  one-second clips. Six original parents produce 30 screw clips per subject.
  Parents shorter than five seconds use five shorter clips and produce a
  warning, controlled by `screw_sampling.short_parent_policy`.
- Parents are split before sampling. Every screw child and every selected view
  stays in its parent's partition. Two views are one labelled sample; they
  do not double sample counts or cross the train/test boundary.
- Depth follows sampled RGB frame indices through `timestamps.csv`
  (`color_idx` to `depth_saved_idx`). Device acquisition timestamps are not
  substituted for annotated video playback seconds.
- TSM uses ResNet18 with `blockres` temporal shifts and mean temporal consensus.
  All BatchNorm layers are trainable. MVTN preserves the supplied ResNet18
  encoder and six temporal attention scales 512, 256, 128, 64, 32, 16, including
  its attention/feed-forward/residual ordering. Inactive upstream parameters
  are omitted, and fixed CUDA device indices are removed.
- Both adapters receive `[batch,views,frames,channels,height,width]`. Views
  share weights; final class logits are averaged. This added fusion rule is
  explicit and is not claimed to reproduce a published multi-camera result.

## Evaluation protocols

The protocol JSONs and generic split helpers match the IMU/sEMG project.

| Protocol | Training pool | Validation | Test |
|---|---|---|---|
| 1: unseen participant | Remaining participants | Next sorted participant after test, or `--validation-subject` | One entire participant; all eight folds by default |
| 2: ordinary gestures | Repetitions 1-4 | Seeded selection of complete trials from training pool, target 20% | Repetition 5 |
| 2: insert/deinsert | Repetitions 1-4 and 6-9 | Same grouped trial selection | Repetitions 5 and 10 |
| 2: screw | Original long repetitions 1-2 | Same grouped trial selection | Original long repetition 3, including every sampled child |

Class coverage is retained in training/validation/test. With all eight
participants and both selected views, each subject contributes 300 samples
(including 60 move and 30 screw samples). A protocol1 fold has 1800 training,
300 validation and 300 test samples. Alphabetical class IDs match IMU/sEMG.
The same seed and source identities make modality/model comparisons use the
same parent trial assignments.

## Training reports and scheduling

Validation macro F1 selects `best.pt`, with validation loss breaking ties.
ReduceLROnPlateau monitors validation loss and multiplies learning rate by 0.5
after its configured patience of 3 bad epochs, down to 1e-6. Early stopping
uses 10 epochs without checkpoint-score improvement. Both are configurable in
`training`. CUDA mixed precision is enabled by default. Test data is evaluated
once, after restoring the selected checkpoint.

Each run saves its resolved settings, issues, source manifest, labels and
state. Each fold saves sample metadata, protocol/normalization information,
`best.pt`, `last.pt`, `history.csv`, `environment.json`, `model.json`,
`test_metrics.json`, `predictions.csv`, `confusion_matrix.csv` and `result.json`.
Terminal output includes training loss/accuracy, validation metrics, learning
rate and epoch time, then final test accuracy, macro/weighted F1, per-gesture
precision/recall/F1 and the confusion matrix. Multiple folds also report mean
and standard deviation.

## Tests

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

Tests use temporary synthetic recordings and check source immutability, temporal
sampling, depth indexing, trial/screw/view split isolation, both architectures,
checkpoint selection, scheduler behavior, terminal reports and output boundaries.
Test fixtures and experiment artifacts are not committed.

Verification on 2026-10-25: 29 automated tests passed. Both protocols prepared
successfully for all eight participants, RGB and depth, using front and above.
Parent-trial assignments were checked against IMU/sEMG and matched for both
protocols. All four modality/model combinations completed a one-epoch CUDA
integration check with nine real samples per partition. Those tiny checks are
marked `benchmark: false` in their run metadata and are not experiment results.
Launching the RGB preparation command from outside this repository also passed.
The eight source annotation hashes remained unchanged throughout verification.
