"""Evaluation plots for the fully-hadronic SPANet multiclassifier.

All functions are headless (no ``plt.show``): they save to an explicit path and
close the figure. Adapted and generalized from the original notebook and
``scripts/evaluate_spanet_onnx.py``.
"""

import os
import re

import matplotlib

matplotlib.use("Agg")  # headless: safe for condor / non-interactive runs

import matplotlib.pyplot as plt  # noqa: E402
import mplhep as hep  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import auc, confusion_matrix, roc_curve  # noqa: E402

from .constants import (  # noqa: E402
    CLASS_NAMES,
    CMAP_10,
    MAPPING_ENCODING,
    colors,
    labels_legend,
)

hep.style.use(hep.style.ROOT)


# -------------------------------------------------------------------------
# small helpers
# -------------------------------------------------------------------------
def _slug(name):
    """Filesystem-safe slug for a class / score / region name."""
    return re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_").lower()


def _save(fig, path, dpi=150):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


# -------------------------------------------------------------------------
# score histograms
# -------------------------------------------------------------------------
def plot_score_on_ax(ax, score_values, true_labels, weights, score_name,
                     log=False, density=False, class_names=None, legend=True):
    """Draw, on ``ax``, the distribution of one node score split by true class."""
    if class_names is None:
        class_names = CLASS_NAMES
    score_values = np.asarray(score_values)
    for name in class_names:
        m = np.asarray(true_labels[name], dtype=bool)
        if not m.any():
            continue
        w = weights[m] if weights is not None else None
        ax.hist(score_values[m], bins=50, range=(0, 1), weights=w,
                histtype="step", linewidth=2, color=colors.get(name),
                label=labels_legend.get(name, name), density=density)
    ax.set_xlabel(f"{labels_legend.get(score_name, score_name)} SPANet score")
    ax.set_ylabel("A.U." if density else "Counts")
    if log:
        ax.set_yscale("log")
    if legend:
        ax.legend(fontsize=10)


def plot_scores_individual(true_labels, pred_labels, out_dir, weights=None,
                           density=False, score_names=None):
    """One figure per score node (linear + log)."""
    if score_names is None:
        score_names = list(pred_labels.keys())
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for score_name in score_names:
        for log in (False, True):
            fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
            plot_score_on_ax(ax, pred_labels[score_name], true_labels, weights,
                             score_name, log=log, density=density)
            fig.tight_layout()
            fname = f"score_{_slug(score_name)}{'_log' if log else ''}.png"
            path = os.path.join(out_dir, fname)
            _save(fig, path)
            written.append(path)
    return written


def plot_scores_joint(true_labels, pred_labels, out_path, weights=None,
                      density=False, log=False, score_names=None, ncols=3):
    """All score nodes in one grid figure."""
    if score_names is None:
        score_names = list(pred_labels.keys())
    n = len(score_names)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 6 * nrows), dpi=100)
    axes = np.atleast_1d(axes).flatten()
    for i, score_name in enumerate(score_names):
        plot_score_on_ax(axes[i], pred_labels[score_name], true_labels, weights,
                         score_name, log=log, density=density)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.tight_layout(pad=1.5)
    _save(fig, out_path, dpi=200)
    return out_path


# -------------------------------------------------------------------------
# ROC curves
# -------------------------------------------------------------------------
def plot_roc_on_ax(ax, true_label, score, label, weight=None, log=False):
    true_label = np.asarray(true_label, dtype=bool)
    fpr, tpr, _ = roc_curve(true_label, np.asarray(score), sample_weight=weight)
    i_sorted = np.argsort(fpr)
    AUC = auc(fpr[i_sorted], tpr[i_sorted])
    ax.plot(tpr, fpr, label=f"{labels_legend.get(label, label)} AUC={AUC:.3f}",
            color=colors.get(label))
    if log:
        ax.set_yscale("log")
    ax.set_xlabel("Signal efficiency")
    ax.set_ylabel("Background efficiency")
    ax.set_title(f"{labels_legend.get(label, label)} ROC")
    ax.legend(fontsize=10)
    return AUC


def plot_roc_individual(true_labels, pred_labels, out_dir, weights=None, class_names=None):
    """One ROC figure per class (linear + log)."""
    if class_names is None:
        class_names = CLASS_NAMES
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for name in class_names:
        for log in (False, True):
            fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
            plot_roc_on_ax(ax, true_labels[name], pred_labels[name], name,
                           weight=weights, log=log)
            fig.tight_layout()
            path = os.path.join(out_dir, f"roc_{_slug(name)}{'_log' if log else ''}.png")
            _save(fig, path)
            written.append(path)
    return written


def plot_rocs_joint(true_labels, pred_labels, out_path, weights=None,
                    log=False, class_names=None, ncols=3):
    """All class ROC curves in one grid figure."""
    if class_names is None:
        class_names = CLASS_NAMES
    n = len(class_names)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows), dpi=100)
    axes = np.atleast_1d(axes).flatten()
    for i, name in enumerate(class_names):
        plot_roc_on_ax(axes[i], true_labels[name], pred_labels[name], name,
                       weight=weights, log=log)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.tight_layout(pad=2.0)
    _save(fig, out_path, dpi=200)
    return out_path


# -------------------------------------------------------------------------
# njet / nbjet
# -------------------------------------------------------------------------
def plot_njet_nbjet(jet_pt, jet_btag_M, out_dir, weights=None,
                    njet_range=(6, 16), nbjet_range=(0, 8), btag_M_value=1):
    """Histogram the jet and (medium) b-jet multiplicities."""
    os.makedirs(out_dir, exist_ok=True)
    njet = np.sum(np.asarray(jet_pt) > 0, axis=1)
    nbjet = np.sum(np.asarray(jet_btag_M) == btag_M_value, axis=1)
    written = []
    for arr, name, rng, xlabel in (
        (njet, "njet", njet_range, "Number of jets"),
        (nbjet, "nbjet", nbjet_range, "Number of b-tagged jets (M)"),
    ):
        fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
        ax.hist(arr, bins=np.arange(rng[0], rng[1] + 1), weights=weights,
                histtype="step", linewidth=2, color=CMAP_10[0])
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Counts")
        fig.tight_layout()
        path = os.path.join(out_dir, f"{name}.png")
        _save(fig, path)
        written.append(path)
    return written


# -------------------------------------------------------------------------
# confusion matrix
# -------------------------------------------------------------------------
def plot_confusion_matrix(y_true, y_pred, out_path, norm=None, weights=None,
                          title=None, cmap="Blues"):
    """Weighted confusion matrix over the 9 physics classes (norm true/pred)."""
    label_ids = sorted(MAPPING_ENCODING)
    names = [MAPPING_ENCODING[i] for i in label_ids]
    cm = confusion_matrix(y_true, y_pred, labels=label_ids, normalize=norm,
                          sample_weight=weights)
    fig = plt.figure(figsize=(12, 12), dpi=100)
    sns.heatmap(cm, annot=True, fmt=".2f", cmap=cmap,
                xticklabels=names, yticklabels=names)
    plt.title(title or f"Confusion matrix (norm={norm})")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    _save(fig, out_path, dpi=100)
    return out_path


# -------------------------------------------------------------------------
# purity / efficiency: pie charts and bars
# -------------------------------------------------------------------------
def plot_composition_pie_charts(stats, out_path, ncols=3, annotate_metrics=True):
    """Grid of pie charts, one per region, showing the weighted sample mix.

    ``stats`` is the dict returned by ``compute_argmax_purity_efficiency`` or
    ``compute_region_purity_efficiency`` (each value has ``composition`` and,
    optionally, ``purity``/``efficiency``).
    """
    names = list(stats.keys())
    n = len(names)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows), dpi=100)
    axs = np.atleast_1d(axs).flatten()
    for i, region in enumerate(names):
        ax = axs[i]
        comp = stats[region]["composition"]
        title = labels_legend.get(region, region)
        if annotate_metrics and "purity" in stats[region]:
            title += (f"\npurity={stats[region]['purity']:.2f} "
                      f"eff={stats[region]['efficiency']:.2f}")
        if comp:
            ax.pie(
                list(comp.values()),
                labels=[labels_legend.get(k, k) for k in comp],
                autopct="%1.1f%%", startangle=90,
                colors=[colors.get(k) for k in comp],
                textprops={"fontsize": 12},
            )
            ax.set_title(title, pad=20)
        else:
            ax.text(0.5, 0.5, "No events", ha="center", va="center")
            ax.set_title(title)
        ax.axis("equal")
    for j in range(n, len(axs)):
        axs[j].axis("off")
    fig.tight_layout()
    _save(fig, out_path, dpi=200)
    return out_path


def plot_purity_efficiency_bars(stats, out_path, title="Purity & efficiency"):
    """Grouped purity/efficiency bars for all regions on one axis."""
    names = list(stats.keys())
    purity = [stats[n]["purity"] for n in names]
    efficiency = [stats[n]["efficiency"] for n in names]
    x = np.arange(len(names))
    width = 0.4
    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(names)), 6), dpi=100)
    ax.bar(x - width / 2, purity, width, label="Purity", color=CMAP_10[2])
    ax.bar(x + width / 2, efficiency, width, label="Efficiency", color=CMAP_10[0])
    ax.set_xticks(x)
    ax.set_xticklabels([labels_legend.get(n, n) for n in names], rotation=45, ha="right")
    ax.set_ylabel("Fraction")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    _save(fig, out_path)
    return out_path


def _plot_metric_1d(stats, metric, out_path, ylabel, title, color):
    names = list(stats.keys())
    vals = [stats[n][metric] for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(names)), 6), dpi=100)
    ax.bar(x, vals, color=color)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([labels_legend.get(n, n) for n in names], rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    fig.tight_layout()
    _save(fig, out_path)
    return out_path


def plot_efficiency_1d(stats, out_path, title="Efficiency per region"):
    return _plot_metric_1d(stats, "efficiency", out_path, "Efficiency", title, CMAP_10[0])


def plot_purity_1d(stats, out_path, title="Purity per region"):
    return _plot_metric_1d(stats, "purity", out_path, "Purity", title, CMAP_10[2])
