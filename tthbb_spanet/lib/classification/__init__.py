"""Classification inference + evaluation toolkit for the fully-hadronic SPANet
multiclassifier.

Public API used by ``apply_spanet_classification.py``.
"""

from . import constants, inference, plots, regions, scores
from .constants import (
    CLASS_NAMES,
    DERIVED_SCORES,
    MAPPING_ENCODING,
    SPANET_OUTPUT_FIELDS,
    colors,
    labels_legend,
)
from .inference import (
    build_onnx_feed,
    group_feature_arrays,
    infer_slice,
    load_event_config,
    load_h5_slice,
    make_session,
    n_events,
    resolve_providers,
    run_onnx_inference,
)
from .regions import default_cut_regions
from .scores import (
    argmax_prediction,
    build_score_dict,
    build_true_pred_labels,
    compute_argmax_purity_efficiency,
    compute_region_purity_efficiency,
)

__all__ = [
    "constants",
    "inference",
    "plots",
    "regions",
    "scores",
    "CLASS_NAMES",
    "DERIVED_SCORES",
    "MAPPING_ENCODING",
    "SPANET_OUTPUT_FIELDS",
    "colors",
    "labels_legend",
    "build_onnx_feed",
    "group_feature_arrays",
    "infer_slice",
    "load_event_config",
    "load_h5_slice",
    "make_session",
    "n_events",
    "resolve_providers",
    "run_onnx_inference",
    "default_cut_regions",
    "argmax_prediction",
    "build_score_dict",
    "build_true_pred_labels",
    "compute_argmax_purity_efficiency",
    "compute_region_purity_efficiency",
]
