"""Provide the SAME class indices for independent IMU and sEMG experiments.

Build the mapping once from the complete, agreed gesture vocabulary, before
creating evaluation folds. Labels are stripped of surrounding whitespace and
sorted alphabetically; indices start at zero. Reuse that mapping for every
subject, modality, and split. Never rebuild indices from a subset of labels.

This module does not rename gestures, modify annotation records, read signals,
or write files. Save the returned dictionary with the experiment configuration
under runs/ when the training pipeline is implemented.

Example:
    mapping = build_label_mapping(r['gesture_label'] for r in records)
    targets = encode_labels((r['gesture_label'] for r in records), mapping)
    names = decode_labels(targets, mapping)
"""


def _label(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Every gesture label must be a nonempty string.")
    return value.strip()


def validate_label_mapping(mapping):
    """Require canonical names and unique contiguous integer indices from zero."""
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("The label mapping must be a nonempty dictionary.")
    for name, index in mapping.items():
        if _label(name) != name:
            raise ValueError("Mapping keys must have no surrounding whitespace.")
        if type(index) is not int:
            raise ValueError("Class indices must be integers, excluding booleans.")
    if sorted(mapping.values()) != list(range(len(mapping))):
        raise ValueError("Class indices must be unique and contiguous from zero.")
    return mapping


def build_label_mapping(gesture_labels, existing_mapping=None):
    """Return alphabetical class indices, or verify/reuse an existing mapping.

    Duplicate names are expected when passing a label for every annotation.
    An existing mapping retains its original indices, including classes absent
    from the current subset; unknown labels raise an error.
    """
    labels = {_label(value) for value in gesture_labels}
    if not labels:
        raise ValueError("Cannot build a mapping from an empty vocabulary.")
    if existing_mapping is not None:
        validate_label_mapping(existing_mapping)
        unknown = labels.difference(existing_mapping)
        if unknown:
            raise ValueError("Unknown gesture labels: {}".format(", ".join(sorted(unknown))))
        return dict(existing_mapping)
    return {name: index for index, name in enumerate(sorted(labels))}


def encode_labels(gesture_labels, mapping):
    """Convert names to indices using an existing mapping; reject unknown names."""
    validate_label_mapping(mapping)
    targets = []
    for value in gesture_labels:
        name = _label(value)
        if name not in mapping:
            raise ValueError("Unknown gesture label: {!r}".format(name))
        targets.append(mapping[name])
    return targets


def decode_labels(class_indices, mapping):
    """Convert predicted/target indices back to gesture names."""
    validate_label_mapping(mapping)
    names = {index: name for name, index in mapping.items()}
    decoded = []
    for index in class_indices:
        if type(index) is not int or index not in names:
            raise ValueError("Unknown class index: {!r}".format(index))
        decoded.append(names[index])
    return decoded
