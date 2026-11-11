# Saved training runs

Experiment settings, metrics, predictions, plots and checkpoints in the modality
folders are versioned. Model checkpoints use Git Large File Storage (Git LFS).
Local test folders, logs at the runs root, caches and prepared NumPy arrays stay
ignored. Raw recordings and annotations remain outside this repository.

Joint-task folders start with `<modality>_gesture_object_<model>_batch...`.
Existing joint-task folders have been renamed to match; their results are unchanged.

Install Git LFS on each computer, then run these commands in this repository:

```powershell
git lfs install --local
git pull --ff-only
git lfs pull
```

To upload new completed runs after training:

```powershell
git add runs
git commit -m "Save training results and checkpoints"
git push origin main
```

Training does not push automatically. GitHub's web page shows checkpoint files
as LFS objects; `git lfs pull` downloads the actual weights. Keep `.gitattributes`
committed so large checkpoints are handled correctly on every computer.
