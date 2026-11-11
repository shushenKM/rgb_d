"""Classification targets for micro gestures, independent of split protocols.

gesture (default) predicts the original gesture. gesture_object predicts an
observed (gesture, object_id) pair, encoded as e.g. grasp::object=1. The original
gesture/object fields remain available for repetition rules and preprocessing.
No recordings or annotations are changed. Pair classification is closed-set:
use protocols 1 or 2; protocol 3 is reserved for gesture-only recognition.
"""
from urllib.parse import quote, unquote

TASKS = ("gesture", "gesture_object")
_SEPARATOR = "::object="


def validate_task(task="gesture", protocol=None):
    if task not in TASKS:
        raise ValueError("task must be gesture or gesture_object")
    if task == "gesture_object" and (protocol or {}).get("type") == "held_out_objects":
        raise ValueError("Protocol 3 supports task=gesture only: held-out objects "
                         "create unseen gesture-object classes. Use protocol 1 or 2 "
                         "for gesture_object.")
    return task


def joint_label(gesture, object_id):
    if not isinstance(gesture, str) or not gesture.strip():
        raise ValueError("A gesture-object label requires a nonempty gesture")
    if not isinstance(object_id, str) or not object_id.strip():
        raise ValueError("A gesture-object label requires a nonempty object_id")
    return quote(gesture.strip(), safe="") + _SEPARATOR + quote(object_id.strip(), safe="")


def split_joint_label(label):
    if _SEPARATOR not in label:
        raise ValueError("Not a gesture-object label: {!r}".format(label))
    gesture, obj = label.split(_SEPARATOR, 1)
    if not gesture or not obj:
        raise ValueError("Incomplete gesture-object label")
    return unquote(gesture), unquote(obj)


def label_of(record):
    """Return the classification target, retaining compatibility with old records."""
    return record.get("target_label", record["gesture_label"])


def with_task_labels(records, task="gesture", protocol=None):
    """Return new dictionaries; never replace the original gesture or object."""
    validate_task(task, protocol)
    return [dict(record, target_label=(
        joint_label(record["gesture_label"], record["object_id"])
        if task == "gesture_object" else record["gesture_label"]))
        for record in records]

def prediction_labels(target, prediction):
    """Readable CSV columns; joint predictions retain both individual components."""
    if "::object=" in target and "::object=" in prediction:
        tg, to = split_joint_label(target)
        pg, po = split_joint_label(prediction)
        return dict(target_label=target, predicted_label=prediction,
                    target_gesture=tg, predicted_gesture=pg,
                    target_object=to, predicted_object=po)
    return dict(target_gesture=target, predicted_gesture=prediction)

