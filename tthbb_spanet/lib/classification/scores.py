"""Derived discriminants and purity/efficiency computations.

Built on top of the raw ``(N, n_classes)`` multiclassifier output ``cl_out``.
"""

from collections import OrderedDict

import numpy as np

from .constants import CLASS_NAMES, MAPPING_ENCODING

EPS = 1e-6


def build_score_dict(cl_out, eps=EPS):
    """Return per-node scores (by class name) plus the three derived scores.

    Derived (Neyman-Pearson style) discriminants:
      * ``tthbb_vs_ttbb``      = tthbb / (tthbb + ttB)
      * ``tthbb_vs_qcd_ttbb``  = tthbb / (tthbb + ttB + QCD)
      * ``tthbb_vs_qcd_ttbar`` = tthbb / (tthbb + QCD + ttLF + ttC + ttB)
    """
    scores = OrderedDict()
    for idx, name in MAPPING_ENCODING.items():
        # contiguous copy: column views are non-contiguous and break arrow/parquet
        scores[name] = np.ascontiguousarray(cl_out[:, idx])

    tthbb = scores["ttH(bb)"]
    qcd = scores["QCD"]
    ttlf = scores["ttLF"]
    ttc = scores["ttC"]
    ttbb = scores["ttB"]

    scores["tthbb_vs_ttbb"] = tthbb / (tthbb + ttbb + eps)
    scores["tthbb_vs_qcd_ttbb"] = tthbb / (tthbb + ttbb + qcd + eps)
    scores["tthbb_vs_qcd_ttbar"] = tthbb / (tthbb + qcd + ttlf + ttc + ttbb + eps)
    return scores


def build_true_pred_labels(cl_out, y_true):
    """Return ``(true_labels, pred_labels)`` dicts keyed by class name.

    ``true_labels[name]`` is a boolean mask of events truly of that class;
    ``pred_labels[name]`` is that class's node score (used for ROC curves).
    """
    true_labels = OrderedDict()
    pred_labels = OrderedDict()
    for idx, name in MAPPING_ENCODING.items():
        true_labels[name] = np.asarray(y_true) == idx
        pred_labels[name] = np.asarray(cl_out[:, idx])
    return true_labels, pred_labels


def argmax_prediction(cl_out):
    """0-based predicted class index (column of the largest score)."""
    return np.argmax(cl_out, axis=1)


def weighted_composition(mask, true_labels, weights, class_names=None):
    """Weighted yield of each true class inside ``mask``.

    Returns an ``OrderedDict(class_name -> weighted_count)`` keeping only
    classes with a strictly positive yield.
    """
    if class_names is None:
        class_names = CLASS_NAMES
    comp = OrderedDict()
    for name in class_names:
        w = float(np.sum(weights[np.asarray(true_labels[name], dtype=bool) & mask]))
        if w > 0:
            comp[name] = w
    return comp


def compute_argmax_purity_efficiency(cl_out, y_true, weights, class_names=None):
    """Purity and efficiency of each argmax region.

    Region ``i`` = events whose argmax is class ``i``; its target is class ``i``.
      * purity     = W(true==i & argmax==i) / W(argmax==i)
      * efficiency = W(true==i & argmax==i) / W(true==i)
    Returns ``OrderedDict(class_name -> {purity, efficiency, n_region, n_target,
    composition})``.
    """
    if class_names is None:
        class_names = CLASS_NAMES
    y_true = np.asarray(y_true)
    weights = np.asarray(weights)
    y_pred = argmax_prediction(cl_out)
    true_labels, _ = build_true_pred_labels(cl_out, y_true)

    out = OrderedDict()
    for idx, name in MAPPING_ENCODING.items():
        region_mask = y_pred == idx
        target_mask = y_true == idx
        w_region = float(np.sum(weights[region_mask]))
        w_target = float(np.sum(weights[target_mask]))
        w_signal_in_region = float(np.sum(weights[region_mask & target_mask]))
        out[name] = {
            "purity": w_signal_in_region / w_region if w_region > 0 else 0.0,
            "efficiency": w_signal_in_region / w_target if w_target > 0 else 0.0,
            "n_region": w_region,
            "n_target": w_target,
            "composition": weighted_composition(region_mask, true_labels, weights, class_names),
        }
    return out


def compute_region_purity_efficiency(regions, true_labels, weights, class_names=None):
    """Purity and efficiency of each cut-based region.

    ``regions`` is an ``OrderedDict(region_name -> {"mask", "target"})`` as
    returned by :func:`tthbb_spanet.lib.classification.regions.default_cut_regions`.
    Purity/efficiency are computed for the region's ``target`` class.
    Returns ``OrderedDict(region_name -> {purity, efficiency, target, n_region,
    n_target, composition})``.
    """
    if class_names is None:
        class_names = CLASS_NAMES
    weights = np.asarray(weights)
    out = OrderedDict()
    for region_name, region in regions.items():
        mask = np.asarray(region["mask"], dtype=bool)
        target = region["target"]
        target_mask = np.asarray(true_labels[target], dtype=bool)
        w_region = float(np.sum(weights[mask]))
        w_target = float(np.sum(weights[target_mask]))
        w_target_in_region = float(np.sum(weights[mask & target_mask]))
        out[region_name] = {
            "target": target,
            "purity": w_target_in_region / w_region if w_region > 0 else 0.0,
            "efficiency": w_target_in_region / w_target if w_target > 0 else 0.0,
            "n_region": w_region,
            "n_target": w_target,
            "composition": weighted_composition(mask, true_labels, weights, class_names),
        }
    return out
