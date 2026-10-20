"""Classification metrics and terminal reports with a fixed class ordering.

Reports display scores as percentages and confusion-matrix entries as counts.
Confusion-matrix rows are true gestures; columns are predicted gestures.
"""

import math
import operator
from statistics import mean, pstdev

from scripts.labels import validate_label_mapping


def _indices(values, class_count, name):
    output = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError("{} must contain integer class indices, excluding booleans.".format(name))
        try:
            index = operator.index(value)
        except TypeError as exc:
            raise ValueError("{} must contain integer class indices.".format(name)) from exc
        if not 0 <= index < class_count:
            raise ValueError("Unknown class index {} in {}.".format(index, name))
        output.append(int(index))
    return output


def classification_metrics(y_true, y_pred, label_mapping):
    """Calculate accuracy/F1 and a confusion matrix over every mapped class.

    Rows in the confusion matrix are true labels; columns are predicted labels.
    An absent class receives zero precision, recall, and F1 and still contributes
    to macro-F1. This keeps comparisons consistent across folds and modalities.
    """
    validate_label_mapping(label_mapping)
    names = [name for name, _ in sorted(label_mapping.items(), key=lambda item: item[1])]
    size = len(names)
    actual = _indices(y_true, size, "y_true")
    predicted = _indices(y_pred, size, "y_pred")
    if not actual or len(actual) != len(predicted):
        raise ValueError("y_true and y_pred must be nonempty and have the same length.")
    confusion = [[0] * size for _ in range(size)]
    for truth, prediction in zip(actual, predicted):
        confusion[truth][prediction] += 1
    per_class = {}
    for index, name in enumerate(names):
        true_positive = confusion[index][index]
        support = sum(confusion[index])
        predicted_count = sum(row[index] for row in confusion)
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {
            "index": index, "precision": precision, "recall": recall,
            "f1": f1, "support": support,
        }
    total = len(actual)
    return {
        "accuracy": sum(confusion[index][index] for index in range(size)) / total,
        "macro_f1": sum(values["f1"] for values in per_class.values()) / size,
        "weighted_f1": sum(values["f1"] * values["support"] for values in per_class.values()) / total,
        "per_class": per_class, "confusion_matrix": confusion,
        "class_names": names, "sample_count": total,
    }


def aggregate_fold_metrics(fold_metrics):
    """Return unweighted mean/population SD across completed fold metrics.

    Every participant fold has equal weight. Pooled predictions, when needed,
    should be supplied separately to ``classification_metrics``.
    """
    folds = list(fold_metrics)
    if not folds:
        raise ValueError("Cannot aggregate an empty collection of fold metrics.")
    summary = {"fold_count": len(folds)}
    for metric in ("accuracy", "macro_f1", "weighted_f1"):
        values = [fold[metric] for fold in folds]
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
            raise ValueError("Fold {} values must be finite numbers.".format(metric))
        summary[metric] = {"mean": mean(values), "std": pstdev(values)}
    return summary


def format_test_report(result):
    """Format a completed fold's saved test results without evaluating again.

    The selected checkpoint may come from an earlier epoch than the last one.
    Macro F1 weights each gesture equally; weighted F1 weights by test support.
    """
    metrics = result["test_metrics"]
    names = metrics["class_names"]
    confusion = metrics["confusion_matrix"]
    correct = sum(confusion[index][index] for index in range(len(names)))
    lines = [
        "",
        "Final test results: {}".format(result["name"]),
        "Checkpoint: {} | selected epoch {} | epochs completed {}".format(
            result["selected_checkpoint"], result["selected_epoch"], result["epochs_completed"]),
        "Test examples: {}".format(metrics["sample_count"]),
        "Test loss: {:.4f}".format(metrics["loss"]),
        "Test accuracy: {:.2%} ({}/{} correct)".format(
            metrics["accuracy"], correct, metrics["sample_count"]),
        "Test macro F1: {:.2%}".format(metrics["macro_f1"]),
        "Test weighted F1: {:.2%}".format(metrics["weighted_f1"]),
        "Training and evaluation time: {:.1f} s".format(result["elapsed_s"]),
        "",
        "Per-gesture test metrics (scores in %; support = test examples):",
    ]
    name_width = max(len("Gesture"), max(map(len, names)))
    row_format = "{:<" + str(name_width) + "}  {:>9}  {:>9}  {:>9}  {:>7}"
    lines.append(row_format.format("Gesture", "Precision", "Recall", "F1", "Support"))
    for name in names:
        values = metrics["per_class"][name]
        lines.append(row_format.format(
            name, "{:.2f}".format(100 * values["precision"]),
            "{:.2f}".format(100 * values["recall"]),
            "{:.2f}".format(100 * values["f1"]), values["support"]))

    lines.extend(["", "Confusion matrix (counts): rows = true, columns = predicted.",
                  "Column indices match the gesture indices shown on the rows."])
    row_labels = ["{}: {}".format(index, name) for index, name in enumerate(names)]
    label_width = max(len("True / Pred"), max(map(len, row_labels)))
    cell_width = max(4, len(str(len(names) - 1)),
                     max(len(str(value)) for row in confusion for value in row))
    cell_format = "{:>" + str(cell_width) + "}"
    lines.append("{:<{}}  {}".format(
        "True / Pred", label_width,
        " ".join(cell_format.format(index) for index in range(len(names)))))
    for label, row in zip(row_labels, confusion):
        lines.append("{:<{}}  {}".format(
            label, label_width, " ".join(cell_format.format(value) for value in row)))
    return "\n".join(lines)
