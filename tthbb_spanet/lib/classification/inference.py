"""ONNX inference helpers for the fully-hadronic SPANet multiclassifier.

Everything about the inputs is driven by the SPANet ``event.yaml`` config:
the feature *names*, their *order*, and their *normalization* are all read
from the yaml so the ONNX tensor layout matches what the model was trained on.
In particular the pre-inference ``log1p`` is applied only to features whose
yaml transform is ``log_normalize`` -- nothing is hardcoded.
"""

from collections import OrderedDict

import h5py
import numpy as np
import onnxruntime as ort
import yaml

# Name of the classification head in the ONNX graph.
DEFAULT_OUTPUT_NAME = "EVENT/signal"
# Number of columns of the classification output (index 0 is the empty class).
DEFAULT_N_CLASSES = 10


def load_event_config(config_path):
    """Read the SPANet ``event.yaml`` and return the input structure.

    Returns a dict with two ordered sub-dicts, ``sequential`` and ``global``,
    each mapping ``group_name -> OrderedDict(feature -> transform)``. Both the
    feature order and the transform string are preserved verbatim from the yaml.
    """
    with open(config_path, "r") as fh:
        data = yaml.safe_load(fh)
    inputs = data["INPUTS"]
    sequential = OrderedDict()
    for group, feats in (inputs.get("SEQUENTIAL") or {}).items():
        sequential[group] = OrderedDict(feats)
    global_ = OrderedDict()
    for group, feats in (inputs.get("GLOBAL") or {}).items():
        global_[group] = OrderedDict(feats)
    return {"sequential": sequential, "global": global_}


def n_events(h5_file):
    """Number of events stored in the H5 file."""
    with h5py.File(h5_file, "r") as f:
        return int(f["WEIGHTS"]["weight"].shape[0])


def load_h5_slice(h5_file, cfg, start=0, stop=None):
    """Load a ``[start, stop)`` slice of the H5 inputs.

    Arrays are stacked in the feature order given by ``cfg`` (read from the
    yaml), *not* the alphabetical order h5py reports. Values are returned raw
    (no normalization applied); ``build_onnx_feed`` is responsible for the
    transforms. Sequential groups carry their ``MASK`` dataset; global groups
    get an all-ones mask.
    """
    with h5py.File(h5_file, "r") as f:
        total = int(f["WEIGHTS"]["weight"].shape[0])
        if stop is None or stop > total:
            stop = total
        start = max(0, int(start))
        sl = slice(start, stop)

        result = {"start": start, "stop": stop, "n_total": total}

        seq = OrderedDict()
        for group, feats in cfg["sequential"].items():
            g = f["INPUTS"][group]
            data = np.stack([g[feat][sl] for feat in feats], axis=2).astype(np.float32)
            mask = np.asarray(g["MASK"][sl], dtype=bool)
            seq[group] = {"data": data, "mask": mask, "features": list(feats)}
        result["sequential"] = seq

        glob = OrderedDict()
        for group, feats in cfg["global"].items():
            g = f["INPUTS"][group]
            data = np.stack([g[feat][sl] for feat in feats], axis=1)[:, None, :].astype(np.float32)
            mask = np.ones((data.shape[0], 1), dtype=bool)
            glob[group] = {"data": data, "mask": mask, "features": list(feats)}
        result["global"] = glob

        result["weight"] = np.asarray(f["WEIGHTS"]["weight"][sl], dtype=np.float32)
        if "CLASSIFICATIONS" in f and "EVENT" in f["CLASSIFICATIONS"] and "signal" in f["CLASSIFICATIONS"]["EVENT"]:
            result["y_true"] = np.asarray(f["CLASSIFICATIONS"]["EVENT"]["signal"][sl])
        else:
            result["y_true"] = None
    return result


def build_onnx_feed(slice_dict, cfg):
    """Build the ONNX ``input_feed`` dict from a loaded slice.

    ``log1p`` is applied to a feature column only when its yaml transform is
    ``log_normalize``; the set of log-normalized columns is derived from ``cfg``.
    """
    feed = {}
    for kind in ("sequential", "global"):
        for group, info in slice_dict[kind].items():
            data = info["data"].copy()
            feats = info["features"]
            transforms = cfg[kind][group]
            for i, feat in enumerate(feats):
                if transforms.get(feat) == "log_normalize":
                    data[..., i] = np.log1p(data[..., i])
            feed[f"{group}_data"] = data
            feed[f"{group}_mask"] = info["mask"]
    return feed


def group_feature_arrays(slice_dict, group):
    """Return an ``OrderedDict(feature -> raw array)`` for a loaded group.

    Used to build the parquet output and to count jets / b-jets. Sequential
    groups give 2D arrays ``(N, n_obj)``; global groups give 1D arrays ``(N,)``.
    """
    for kind in ("sequential", "global"):
        if group in slice_dict[kind]:
            info = slice_dict[kind][group]
            arrays = OrderedDict()
            for i, feat in enumerate(info["features"]):
                col = info["data"][..., i]
                if kind == "global":
                    col = col[:, 0]
                # column slices are non-contiguous views; arrow/parquet needs
                # contiguous buffers.
                arrays[feat] = np.ascontiguousarray(col)
            return arrays
    raise KeyError(f"group {group!r} not found in loaded slice")


def resolve_providers(cpu_only=False):
    """ONNX execution providers; GPU first unless ``cpu_only``."""
    if cpu_only:
        return ["CPUExecutionProvider"]
    return ["CUDAExecutionProvider", "CPUExecutionProvider"]


def make_session(onnx_path, providers=None, intra_op_threads=4, inter_op_threads=1):
    """Create an ONNX Runtime inference session with sensible defaults."""
    so = ort.SessionOptions()
    so.intra_op_num_threads = intra_op_threads
    so.inter_op_num_threads = inter_op_threads
    so.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if providers is None:
        providers = resolve_providers()
    return ort.InferenceSession(onnx_path, sess_options=so, providers=providers)


def run_onnx_inference(
    onnx_path,
    feed,
    batch_size=2048,
    providers=None,
    output_name=DEFAULT_OUTPUT_NAME,
    n_classes=DEFAULT_N_CLASSES,
    session=None,
):
    """Run batched ONNX inference and return the ``(N, n_classes)`` score array.

    Unlike the original notebook loop (which dropped the final partial batch),
    every event is processed.
    """
    if session is None:
        session = make_session(onnx_path, providers)
    n = next(iter(feed.values())).shape[0]
    out = np.empty((n, n_classes), dtype=np.float32)
    for i in range(0, n, batch_size):
        j = min(i + batch_size, n)
        batch = {k: v[i:j] for k, v in feed.items()}
        out[i:j] = session.run([output_name], batch)[0]
    return out


def infer_slice(h5_file, config_path, onnx_path, start=0, stop=None,
                batch_size=2048, cpu_only=False):
    """Convenience: load a slice, build the feed, run inference, return scores."""
    cfg = load_event_config(config_path)
    slice_dict = load_h5_slice(h5_file, cfg, start=start, stop=stop)
    feed = build_onnx_feed(slice_dict, cfg)
    providers = resolve_providers(cpu_only=cpu_only)
    return run_onnx_inference(onnx_path, feed, batch_size=batch_size, providers=providers)
