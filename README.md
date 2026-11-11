# RGB and depth manipulative gesture recognition

## Command and parameter guide

Saved runs are versioned, with checkpoint files stored through Git LFS.
See [uploading/downloading runs](runs/README.md) for commands on each computer.
Joint-task run names begin with `<modality>_gesture_object_<model>_batch...`.

Run from `rgb_d` with your environment activated. Each command is an independent
example. CLI values override config values. Dataset/annotation files stay
read-only; outputs go to `runs/rgb` or `runs/depth`.

### Prepare, then train

```powershell
# Check selected data and splits without optimizing a model
python train_rgb.py --protocol protocol2 --model tsm --views front --prepare-only
python train_depth.py --protocol protocol2 --model tsm --views front --prepare-only

# TSM, one camera
python train_rgb.py --protocol protocol2 --model tsm --views front --epochs 50 --device cuda --workers 4
python train_depth.py --protocol protocol2 --model tsm --views front --epochs 50 --device cuda --workers 4
# MVTN, two cameras
python train_rgb.py --protocol protocol2 --model mvtn --views above front --epochs 50 --device cuda --workers 4
python train_depth.py --protocol protocol2 --model mvtn --views above front --epochs 50 --device cuda --workers 4

# Gesture-object recognition (44 observed pairs in steps1-3)
python train_rgb.py --task gesture_object --protocol protocol2 --model tsm --views front --prepare-only
python train_rgb.py --task gesture_object --protocol protocol2 --model tsm --views front --device cuda
python train_depth.py --task gesture_object --protocol protocol2 --model mvtn --views above front --device cuda

# Protocol1: all participant folds, or one specified test/validation participant
python train_rgb.py --protocol protocol1 --model tsm --views front --device cuda
python train_depth.py --protocol protocol1 --test-subject 1Alice --validation-subject 2Bingqian --device cuda
# Protocol3: gesture-only recognition with configured held-out objects
python train_rgb.py --task gesture --protocol protocol3 --views front --prepare-only
python train_rgb.py --task gesture --protocol protocol3 --views front --device cuda
python train_depth.py --task gesture --protocol protocol3 --views front --device cuda

# Regenerate existing plots; replace YOUR_RUN_FOLDER
python -m scripts.plots --run "runs/rgb/YOUR_RUN_FOLDER"
python train_rgb.py --help
python train_depth.py --help
```

Both tasks support TSM/MVTN, RGB/depth, and either annotated view or both views.
`gesture` (default) predicts 9 gestures; `gesture_object` predicts the exact pair.
The joint task supports protocol1 and both protocol2 modes. Protocol3 is
gesture-only because its test objects create unseen joint classes. These entry
points use steps1–3; step4 experiments belong to the sibling `macro` project.

### All training CLI parameters

| Parameter | Meaning / choices |
| --- | --- |
| `--protocol NAME` | Required: `protocol1`, `protocol2`, `protocol3`, or custom JSON path. |
| `--task gesture` / `gesture_object` | Config task if omitted; otherwise default `gesture`. |
| `--config PATH` | Defaults to `configs/rgb.json` or `configs/depth.json`. |
| `--prepare-only` | Check recordings and splits, save a compact summary; no model training. |
| `--model tsm` / `mvtn` | Recognition model; defaults to the modality config. |
| `--views front above` | One or more cameras, in order. Loader accepts front/above/left/right; current annotations support front/above only. |
| `--subjects ID ...` | Optional participant folder subset. |
| `--test-subject ID` | Protocol1: one LOSO test fold instead of all folds. |
| `--validation-subject ID` | Protocol1: explicit validation participant. |
| `--epochs N` | Maximum training epochs. |
| `--device auto` / `cpu` / `cuda` / `cuda:0` | Auto selects CUDA when available. |
| `--batch-size N` | Examples per training batch. |
| `--workers N` | Data-loader workers; default 0 is portable on Windows. |
| `--frames N` | Frames sampled per gesture and camera; default config uses 8. |
| `--image-size N` | Square model input size; default config uses 128. |
| `--backbone-weights PATH` | Optional local standard ResNet18 state dict; repository-relative or absolute path. |
| `--run-name NAME` | New custom run directory; existing directories are never overwritten. |
| `-h`, `--help` | Display the selected entry point's options. |

These entry points have no `--experiment`, `--audit-only`, `--amp` or
`--learning-rate` flags; use the modality config for AMP/learning rate.

### Settings edited in JSON

| File / setting | Effect |
| --- | --- |
| `configs/protocol1.json`: `test_subject_id` | Null rotates LOSO test participants; an ID fixes one test fold. |
| `configs/<modality>.json`: `training.validation` | Auto uses a validation participant for protocol1 and grouped trials for repetition/object modes; `fraction` controls grouped validation. |
| `configs/protocol2.json`: `grouped_percentages: false` | Use `train_repetitions`, `test_repetitions`, gesture-specific rules, and model validation settings. |
| Same file: `grouped_percentages: true` | Use `split_percentages: {"train": 70, "validation": 15, "test": 15}` and `split_seed`; repetition/model validation rules are inactive. |
| `configs/protocol3.json`: `test_object_ids` | Fixed test objects per gesture. These lists override `object_selection: last_per_gesture`; defaults are 6 for general gestures, 5 for insert/deinsert, 2 for screw. |
| `configs/rgb.json`, `configs/depth.json`: `task`, `subject_ids`, `views` | Persistent task and selected participants/cameras. |
| Same files: `training.learning_rate`, `weight_decay`, `seed`, `cpu_threads`, `deterministic`, `amp` | Optimizer, reproducibility and mixed precision. |
| Same files: `training.scheduler`, `early_stopping_patience`, `gradient_clip_norm` | Learning-rate scheduling, stopping and gradient clipping. |
| Same files: `training.batch_size`, `epochs`, `device`, `num_workers` | Defaults for their corresponding CLI flags. |
| Same files: `input.num_frames`, `input.image_size` | Temporal/spatial input sizes; CLI overrides available. |
| `configs/depth.json`: `input.depth_min`, `input.depth_max` | Fixed depth-value range used for scaling. |
| Modality configs: `screw_sampling`, `model_parameters.pretrained_backbone`, `data_quality` | Screw excerpts, optional local backbone weights and accepted overlap settings. |
| `configs/dataset.json` | Portable `../dataset`, annotation selection and step-to-recording mappings; keep `read_only: true`. |

Percentage ratios total 100; train/test are positive, validation may be zero.
Whole source trials remain together, so sample percentages can differ after
rounding/windowing. Select either mode in the same protocol2 config and keep
using `--protocol protocol2`. `protocol2_percentages` remains a legacy alias.

## Project overview

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
python train_rgb.py --protocol protocol2 --model tsm --views front --prepare-only
python train_rgb.py --protocol protocol1 --model mvtn --views front above --prepare-only
```

Preparation validates selected annotations, source availability, container
metadata, endpoint decoding, class coverage and splits. It saves a compact
summary in `config.json` without running optimization or extracting complete
videos. It does not decode every frame or guarantee the visual correctness of
every label. For depth, it verifies selected depth paths and probes a depth
image per recording.

The accepted source-time overlaps remain **warnings**, including intersections
between the same participant's training and test repetitions. This is explicit
in `data_quality.allow_within_subject_test_overlap: true` in both modality
configs. Set it to false for strict temporal checks. Warning counts are saved in
`config.json` and printed in the terminal. Duplicate record IDs and trial
assignment across partitions remain errors. Raw cross-participant pixel copies
are not content-hashed by this baseline. Accepted overlaps may affect the
interpretation of test scores; they are retained as requested.

## Train RGB

```powershell
# TSM, one camera, held-out repetitions
python train_rgb.py --protocol protocol2 --model tsm --views front --epochs 50 --device cuda

# MVTN, above camera
python train_rgb.py --protocol protocol2 --model mvtn --views above --epochs 50 --device cuda

# Two cameras, shared model weights and averaged class logits
python train_rgb.py --protocol protocol2 --model tsm --views front above --device cuda

# One unseen-participant fold
python train_rgb.py --protocol protocol1 --model mvtn --views front --test-subject 1Alice --device cuda

# All eight unseen-participant folds
python train_rgb.py --protocol protocol1 --model tsm --views front --device cuda
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
python train_depth.py --protocol protocol2 --model tsm --views front --prepare-only
python train_depth.py --protocol protocol2 --model tsm --views front --device cuda
python train_depth.py --protocol protocol1 --model mvtn --views front above --test-subject 1Alice --device cuda
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

Each run saves its settings, status, labels, warning counts and compact split
summary together in `config.json`. Completed results and fold averages are in
`results.json` and `aggregate_metrics.json`, with a linked `report.html`.
Each trained fold saves `best.pt`, `last.pt`, `history.csv`, `predictions.csv`,
`confusion_matrix.csv`, `result.json`, `report.html`, and PNG/SVG plots.
`result.json` includes test metrics and model/device/training details.
Separate sample lists, recording manifests, labels, normalization, model/environment
JSON files and plot-status files are no longer exported. Checkpoints retain the
model settings and label mapping. Preparation-only runs save just `config.json`;
failures alone create `error.json`. Existing run folders remain unchanged.
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


Run folder names are generated from the effective experiment settings, without
any date or time: `modality_model_batchN_epochN_protocol_selection_...`.
Selection lists the sensors, grids, or camera views in input order. Names also
include input points (IMU/sEMG) or frame count and image size (RGB/depth), initial
learning rate, seed, and explicitly configured test/validation participants.
The epoch value is the requested maximum, not the early-stopped epoch count.
Preparation-only runs end in `_prepare`; synthetic checks end in `_synthetic`.
Repeated automatic names get `_run02`, `_run03`, etc., without overwriting runs.
`--run-name` still chooses an exact custom name and rejects an existing name.
Existing folders keep their names; all new folders remain under `runs/<modality>/`.
Full settings, including parameters omitted from the name, remain in `config.json`.


After training, open `report.html` in the run folder. It links each fold's
`report.html` and, for multiple folds, compares final test accuracy, macro F1,
and weighted F1 (unweighted mean and population SD). Each fold's `plots/`
contains PNG and SVG figures for loss, accuracy, validation macro F1, learning
rate, epoch duration, confusion matrices (counts and percentages per true class),
per-gesture precision/recall/F1, and test support. The selected checkpoint epoch
is marked on learning curves. Test is evaluated only at the end, not every epoch.
No browser window opens automatically; the terminal prints the report location.

Plots use saved `history.csv` and `result.json`, so completed older runs can be
plotted without reading recordings or retraining. From the repository root:

```powershell
python -m scripts.plots --run "runs/rgb/your_run_folder"
```

A single fold directory also works. Missing historical fields (such as epoch
duration) are shown as unavailable. Regeneration replaces generated plots and
HTML, preserving checkpoints and saved metrics. A plotting error is recorded in
`plot_error.txt` and printed without failing completed training. Successful plots create no status file. Matplotlib is
included in the requirements; its cache is kept in `runs/_plot_cache/`.

## Micro-gesture tasks and protocol 3

The classification target and the evaluation split are independent settings:

| Task | Target example | Supported protocols |
| --- | --- | --- |
| `gesture` (default) | `grasp` | 1: unseen participant; 2: held-out repetitions; 3: held-out objects per gesture |
| `gesture_object` | `grasp::object=1` | 1 and 2 |

Set `--task gesture_object`, or set top-level `"task": "gesture_object"` in a
model configuration. The CLI overrides that configuration field. Omitting it
keeps gesture-only recognition. Models learn one joint class for each observed
gesture/object combination, rather than two independent output heads. The
current steps 1–3 annotations contain **9 gestures and 44 gesture–object pairs**.
The output layer size is inferred from those labels. Existing
`expected_class_count` / `expected_gesture_labels` still validate the raw
gesture vocabulary; do not change the expected count from 9 to 44.

`configs/protocol3.json` reserves these complete object trials for test:

| Gestures | Training/validation objects | Test object |
| --- | --- | --- |
| approach, grasp, release, lift, move, place | 1–5 | 6 |
| screw | 1 | 2 |
| insert, deinsert | 3–4 | 5 |

Insert/deinsert use three objects whose recorded IDs are **3, 4, 5**.
The config lists actual IDs in `test_object_ids`; edit those lists to select
different test objects. Removing an override uses the last available object
within that gesture (numeric IDs are ordered numerically). Keep related
gestures from the same subject/session/object/repetition in the same split;
inconsistent rules are rejected.

This evaluates **held-out objects within each gesture**. It does not mean an
object is absent from all training gestures: object 5 can appear in training
for grasp and in test for insert. All repetitions of each held-out pair are
reserved for test before segmentation, window generation, or view expansion.
Validation uses complete trials from the remaining objects. Its fraction/seed
come from the existing model configuration's `training.validation` and
`training.seed`; the default fraction is 0.2 of eligible training trials.
Class coverage may require a slightly different realized fraction. The test
fraction depends on each gesture's object count, rather than a fixed global
percentage.

For `gesture_object`, protocols 1/2 require every joint class in each active
partition. Protocol 2 retains the existing screw/insert repetition rules and
selects complete validation trials while retaining joint-class coverage.
Consequently validation membership can differ from a gesture-only run even
with the same fraction/seed. Protocol 3 with `gesture_object` is rejected:
its held-out pairs would be classes absent from training. Such an experiment
would require a separate zero-shot or compositional recognition design.

Preprocessing, selected sensors/grids/views, optimizer, scheduler and checkpoint
selection are unchanged. Selection uses validation macro F1 for the chosen
target. Joint-task test accuracy requires **both gesture and object to match**;
reports also include gesture-only and object-only accuracy/F1 derived from the
joint predictions. Confusion matrices and per-class scores describe the 44
pairs. Prediction CSVs include both pair labels and separate gesture/object
columns. Joint run names include `gesture_object`; full settings and the label
mapping remain in the existing compact run outputs and checkpoints.

These options cover micro gestures from steps 1–3 only. The sibling macro
project is not extended by this feature. Source recordings and annotations
remain read-only.

Protocol 3's `allow_within_subject_test_overlap: true` explicitly retains the
accepted reused-interval policy. Some existing IMU/sEMG annotations use the
same time interval under different object IDs (the initial split audit found
18 IMU and 26 sEMG test-overlap warnings). These are warnings, including matching
sEMG content, and the compact run configuration retains warning counts.
Records/trial IDs crossing splits remain errors. The source annotations are
never changed. Scores with reused train/test signals do not measure completely
independent test signals. Set the protocol option to `false` for strict checks;
for RGB/depth also disable the existing
`data_quality.allow_within_subject_test_overlap` exception in the model config.


From the repository root, prepare first or train using TSM or MVTN:

```powershell
# Gesture-only recognition on held-out objects
python train_rgb.py --protocol protocol3 --model tsm --views front --prepare-only
python train_rgb.py --protocol protocol3 --model tsm --views front --device cuda
python train_depth.py --protocol protocol3 --model tsm --views front --device cuda

# Gesture-object recognition with held-out repetitions
python train_rgb.py --task gesture_object --protocol protocol2 --model tsm --views front --device cuda
python train_depth.py --task gesture_object --protocol protocol2 --model tsm --views front --device cuda
```

Use `configs/protocol1.json` instead for participant holdout, or `--model mvtn`
for the existing MVTN adapter. Camera selection and other training options
remain available. Add `--prepare-only` to any normal command to validate it
without optimization.

## Protocol names and the unified protocol 2

Use short names in training commands: `--protocol protocol1`, `--protocol protocol2`,
or `--protocol protocol3`. Explicit JSON paths still work for custom configs.

Both protocol 2 modes are controlled by `configs/protocol2.json`:

```json
"grouped_percentages": false,
"split_percentages": {
  "train": 70,
  "validation": 15,
  "test": 15
},
"split_seed": 42
```

- **false (default):** keep the existing train/test repetition IDs and gesture
  overrides. The existing training validation settings apply. `split_percentages`
  and `split_seed` are inactive.
- **true:** shuffle and divide complete source trials according to
  `split_percentages`. Repetition IDs no longer determine the partition.
  Repetition rules and the model's validation fraction/strategy are inactive;
  validation is already included in the three-way percentage split.

Values must sum to 100; train and test must be positive, while validation may
be zero. The split unit is (subject, session, object, original repetition),
including all gestures and any screw excerpts from that trial. Groups are
stratified by their target-class composition and pooled across participants.
Every active partition must contain every target class; insufficient groups
produce an explanatory error. This works for both `gesture` and
`gesture_object`. Use protocol 1 for unseen-participant evaluation.

The ratios target trial counts. Integer rounding and unequal trial lengths can
make annotation/window counts differ from 70/15/15. The terminal and saved
configuration show requested ratios and realized trial/annotation percentages;
window/clip counts remain in the usual split summary. Input row order does not
affect assignment. `split_seed` controls the shuffle; use matching annotations,
task, selection and split settings across modalities for matching parent splits.

The existing accepted-overlap policy is retained for percentage mode:
`allow_within_subject_test_overlap: true` reports reused test intervals as
warnings. This option does not change normal protocol 2's repetition policy.
Models, filtering, optimization, and final evaluation are otherwise unchanged.

There is no separate percentage configuration to maintain.
`--protocol protocol2_percentages` remains a compatibility alias that reads
this same protocol2.json and forces percentage mode for that invocation.
For normal use, edit the boolean and always run `--protocol protocol2`.
Percentage run names include the requested ratios and split seed.
