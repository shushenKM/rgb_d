"""Save training diagnostics and final-test figures without reading source data.

Automatically called after checkpoint selection and final test evaluation.
Reads only history.csv/result.json in a completed fold; never evaluates a model.
Writes plots/*.png (screen/report use), plots/*.svg (vector export), and report.html.
Plot failures alone create plot_error.txt. Multi-fold reports compare scores without averaging
learning curves of different lengths. Test scores are final-checkpoint results,
not per-epoch curves; the selected epoch is marked on training curves.

From the repository root, regenerate an existing run or a single fold with:
    python -m scripts.plots --run runs/imu/<run_name>
The same command supports semg, rgb and depth in their respective repositories.
Paths resolve from the repository, not the working directory. All outputs and
Matplotlib's font cache stay under runs/. No GUI windows or network are required.
Missing historical columns (e.g. epoch duration) are shown as unavailable.
Plot errors are reported separately so successful training/checkpoints survive.
"""
import argparse
import csv
import html
import json
import math
import os
from pathlib import Path
import sys
from urllib.parse import quote

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _inside_runs(value):
    root = PROJECT_ROOT.resolve()
    boundary = (root / "runs").resolve()
    path = Path(value)
    path = (path if path.is_absolute() else root / path).resolve()
    if not boundary.is_relative_to(root) or not path.is_relative_to(boundary) or path == boundary:
        raise ValueError("Plot inputs and outputs must remain inside this project's runs/ directory")
    return path


def _target(folder, name):
    target = folder / name
    if target.resolve().parent != folder.resolve():
        raise ValueError("Plot/report files must not redirect outside their output folder")
    return target


def _read(folder, name):
    return json.loads(_target(folder, name).read_text(encoding="utf-8-sig"))


def _plot_tools():
    cache = _inside_runs(PROJECT_ROOT / "runs" / "_plot_cache")
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache)
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    return Figure, FigureCanvasAgg


def _figure(width=9, height=5):
    Figure, Canvas = _plot_tools()
    figure = Figure(figsize=(width, height), layout="constrained", facecolor="white")
    Canvas(figure)
    return figure, figure.subplots()


def _save(figure, folder, stem):
    for extension in ("png", "svg"):
        figure.savefig(_target(folder, stem + "." + extension), dpi=180, facecolor="white")
    figure.clear()


def _number(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else float("nan")
    except (ValueError, TypeError):
        return float("nan")


def normalized_confusion(matrix):
    """Percentages within each true class; zero-support rows stay zero."""
    return [[100 * value / sum(row) if sum(row) else 0.0 for value in row] for row in matrix]


def _curve(rows, fields, title, ylabel, selected_epoch, folder, stem, percent=False, log=False):
    fig, ax = _figure()
    epochs = [_number(row.get("epoch")) for row in rows]
    plotted = False
    for key, label, color in fields:
        values = [_number(row.get(key)) * (100 if percent else 1) for row in rows]
        if any(math.isfinite(value) for value in values):
            ax.plot(epochs, values, label=label, color=color, linewidth=2, marker="o", markersize=3)
            plotted = True
    if not plotted:
        ax.text(.5, .5, "Not recorded for this run", transform=ax.transAxes, ha="center")
    if plotted and selected_epoch is not None:
        ax.axvline(selected_epoch, color="#64748b", linestyle="--", linewidth=1,
                   label="Selected checkpoint (epoch {})".format(selected_epoch))
    if percent:
        ax.set_ylim(0, 105)
    if log and plotted:
        finite = [_number(row.get(fields[0][0])) for row in rows]
        if all(value > 0 for value in finite if math.isfinite(value)):
            ax.set_yscale("log")
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set(title=title, xlabel="Epoch", ylabel=ylabel)
    ax.grid(alpha=.2)
    if plotted:
        ax.legend(fontsize=9)
    _save(fig, folder, stem)


def _confusion(metrics, folder, normalized=False):
    names, counts = metrics["class_names"], metrics["confusion_matrix"]
    target = "gesture-object pair" if metrics.get("task") == "gesture_object" else "gesture"
    matrix = normalized_confusion(counts) if normalized else counts
    size = max(7, len(names) * .65 + 2)
    fig, ax = _figure(size, size * .85)
    maximum = 100 if normalized else max(1, max(max(row) for row in counts))
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=maximum)
    fig.colorbar(im, ax=ax, label="% within true " + target if normalized else "Test examples", shrink=.8)
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(names)), names)
    for row in range(len(names)):
        for col in range(len(names)):
            value = matrix[row][col]
            label = "{:.1f}%".format(value) if normalized else str(value)
            ax.text(col, row, label, ha="center", va="center", fontsize=8,
                    color="white" if value > maximum * .55 else "#172033")
    ax.set(xlabel="Predicted " + target, ylabel="True " + target,
           title="Final test confusion matrix" + (" — row-normalized" if normalized else " — counts"))
    _save(fig, folder, "confusion_matrix_normalized" if normalized else "confusion_matrix")


def _class_plots(metrics, folder):
    names = metrics["class_names"]
    target = "gesture-object pair" if metrics.get("task") == "gesture_object" else "gesture"
    fig, ax = _figure(max(9, len(names) * .8), 5)
    for offset, (key, label, color) in enumerate((("precision", "Precision", "#2563eb"),
                                                ("recall", "Recall", "#d97706"),
                                                ("f1", "F1", "#0d9488"))):
        ax.bar([index + (offset - 1) * .25 for index in range(len(names))],
               [100 * metrics["per_class"][name][key] for name in names],
               width=.25, label=label, color=color)
    ax.set_xticks(range(len(names)), names, rotation=35, ha="right")
    ax.set(ylim=(0, 105), ylabel="Score (%)", title="Final test scores by " + target)
    ax.legend()
    ax.grid(axis="y", alpha=.2)
    _save(fig, folder, "per_gesture_scores")
    fig, ax = _figure(max(9, len(names) * .8), 4.5)
    bars = ax.bar(names, [metrics["per_class"][name]["support"] for name in names], color="#64748b")
    ax.bar_label(bars, padding=3)
    ax.tick_params(axis="x", rotation=35)
    ax.margins(y=.18)
    ax.set(ylabel="Test examples", title="Test support by " + target)
    _save(fig, folder, "test_support")


def _html_report(folder, title, intro, plots, links=()):
    cards = "".join('<section><h2>{}</h2><a href="plots/{}.png"><img src="plots/{}.png" alt="{}"></a>'
                    '<p><a href="plots/{}.svg">Download vector SVG</a></p></section>'.format(
                        html.escape(label), stem, stem, html.escape(label), stem) for stem, label in plots)
    navigation = "".join('<li><a href="{}">{}</a></li>'.format(quote(path, safe="/"), html.escape(label))
                         for path, label in links)
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>{title}</title><style>body{{font:16px/1.6 system-ui,sans-serif;background:#f1f5f9;color:#172033;margin:0}}
main{{max-width:1100px;margin:auto;padding:32px}}section{{background:white;padding:20px;margin:24px 0;border-radius:12px}}
img{{display:block;width:100%;height:auto}}a{{color:#1d4ed8}}h1{{overflow-wrap:anywhere}}p{{white-space:pre-line}}</style>
<main><h1>{title}</h1><p>{intro}</p><ul>{navigation}</ul>{cards}</main></html>'''.format(
        title=html.escape(title), intro=html.escape(intro), navigation=navigation, cards=cards)
    _target(folder, "report.html").write_text(page, encoding="utf-8")


def generate_fold_report(output_dir):
    """Regenerate plots from saved metrics/history; return report.html."""
    output = _inside_runs(output_dir)
    result = _read(output, "result.json")
    metrics = result["test_metrics"]
    with _target(output, "history.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("history.csv has no completed epochs")
    plots = _target(output, "plots")
    plots.mkdir(exist_ok=True)
    selected = result.get("selected_epoch")
    curves = [
        ("loss", "Training and validation loss", "Cross-entropy loss", [("train_loss", "Train", "#2563eb"), ("validation_loss", "Validation", "#d97706")], False, False),
        ("accuracy", "Training and validation accuracy", "Accuracy (%)", [("train_accuracy", "Train", "#2563eb"), ("validation_accuracy", "Validation", "#d97706")], True, False),
        ("validation_f1", "Validation macro F1", "Macro F1 (%)", [("validation_macro_f1", "Validation", "#0d9488")], True, False),
        ("learning_rate", "Learning rate used during each epoch", "Learning rate", [("learning_rate", "Learning rate", "#7c3aed")], False, True),
        ("epoch_duration", "Epoch duration", "Seconds (train + validation + checkpoints)", [("epoch_seconds", "Duration", "#64748b")], False, False),
    ]
    for stem, title, ylabel, fields, percent, log in curves:
        _curve(rows, fields, title, ylabel, selected, plots, stem, percent, log)
    _confusion(metrics, plots)
    _confusion(metrics, plots, normalized=True)
    _class_plots(metrics, plots)
    figures = [(row[0], row[1]) for row in curves] + [
        ("confusion_matrix", "Test confusion matrix — counts"),
        ("confusion_matrix_normalized", "Test confusion matrix — percentages"),
        ("per_gesture_scores", "Per-gesture precision, recall and F1"), ("test_support", "Test examples per gesture")]
    intro = ("Test accuracy: {:.2%} | Macro F1: {:.2%} | Weighted F1: {:.2%} | Test examples: {}\n"
             "Selected checkpoint: {} (epoch {}) | Completed epochs: {}\n"
             "Test scores are from the selected checkpoint, evaluated after training. "
             "Training curves are unsmoothed. Training scores are measured during updates; validation is measured in evaluation mode. "
             "Zero-support classes have zero scores and zero normalized confusion rows. Missing historical columns are marked unavailable.").format(
        metrics["accuracy"], metrics["macro_f1"], metrics["weighted_f1"], metrics["sample_count"],
        result.get("selected_checkpoint", "unknown"), selected, result.get("epochs_completed", len(rows)))
    if metrics.get("task") == "gesture_object":
        intro = "Task: gesture-object recognition (exact pair scores).\n" + intro
        for component, scores in metrics.get("component_metrics", {}).items():
            intro += " | {} component: accuracy {:.2%}, macro F1 {:.2%}".format(
                component.capitalize(), scores["accuracy"], scores["macro_f1"])
        figures = [(stem, label.replace("Per-gesture", "Per-pair").replace("per gesture", "per pair"))
                   for stem, label in figures]
    _html_report(output, "Training report — " + result.get("name", output.name), intro, figures)
    return output / "report.html"


def generate_run_report(run_dir, results=None):
    """Link completed fold reports; compare fold metrics with equal fold weight."""
    run = _inside_runs(run_dir)
    results = _read(run, "results.json") if results is None else results
    if not results:
        raise ValueError("No completed fold results")
    names = [result["name"] for result in results]
    links = []
    for name in names:
        fold = _target(run, name)
        if _target(fold, "report.html").is_file():
            links.append((name + "/report.html", name))
    plots = []
    if len(results) > 1:
        folder = _target(run, "plots")
        folder.mkdir(exist_ok=True)
        fig, ax = _figure(max(10, len(results) * 1.1), 5.5)
        for offset, (key, label, color) in enumerate((("accuracy", "Accuracy", "#2563eb"),
                                                    ("macro_f1", "Macro F1", "#0d9488"),
                                                    ("weighted_f1", "Weighted F1", "#d97706"))):
            values = [100 * item["test_metrics"][key] for item in results]
            mean = sum(values) / len(values)
            sd = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
            ax.bar([i + (offset - 1) * .25 for i in range(len(names))], values, width=.25,
                   label="{} (mean {:.1f}%, SD {:.1f}%)".format(label, mean, sd), color=color)
        ax.set_xticks(range(len(names)), names, rotation=35, ha="right")
        ax.set(ylim=(0, 110), ylabel="Final test score (%)", title="Test results across folds")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=.2)
        _save(fig, folder, "fold_comparison")
        plots = [("fold_comparison", "Final test results across folds")]
    intro = "{} completed fold(s). Open a fold below for learning curves and test diagnostics.\nFold comparisons use unweighted mean and population standard deviation; they are not pooled test scores.".format(len(results))
    if len(links) < len(results):
        intro += "\nSome fold reports are unavailable; see terminal output or plot_error.txt in those folders."
    _html_report(run, "Experiment report — " + run.name, intro, plots, links)
    return run / "report.html"


def save_training_report(output_dir, *, results=None):
    """Best-effort automatic reporting: a plotting error never discards training."""
    output = _inside_runs(output_dir)
    try:
        report = generate_fold_report(output) if results is None else generate_run_report(output, results)
        print("Saved plots/report: {}".format(report.relative_to(PROJECT_ROOT)), flush=True)
        return report
    except Exception as error:
        try:
            _target(output, "plot_error.txt").write_text(str(error) + "\n", encoding="utf-8")
        except OSError:
            pass
        print("Plot generation failed: {}. Training results are saved. Retry with python -m scripts.plots --run \"{}\"".format(
            error, output.relative_to(PROJECT_ROOT)), file=sys.stderr, flush=True)
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Existing run or fold under this repository's runs/")
    args = parser.parse_args(argv)
    try:
        folder = _inside_runs(args.run)
        if _target(folder, "result.json").is_file():
            generate_fold_report(folder)
        else:
            results = _read(folder, "results.json")
            for result in results:
                generate_fold_report(_target(folder, result["name"]))
            generate_run_report(folder, results)
        print("Saved plots/report: {}".format(folder / "report.html"), flush=True)
        return 0
    except (ImportError, OSError, ValueError, KeyError) as error:
        print("Plot generation failed: {}".format(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
