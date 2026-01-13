#!/usr/bin/env python3
# coding: utf-8

import os
import argparse
import numpy as np
import h5py
import yaml
import awkward as ak
import numba
import pandas as pd
import vector
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import mplhep as hep
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import onnxruntime as ort
from multiprocessing import Pool, cpu_count

vector.register_numba()
vector.register_awkward()
hep.style.use(hep.style.ROOT)

# -------------------------
# Plot settings and colors
# -------------------------
CMAP_10 = [
    "#3f90da","#ffa90e","#bd1f01","#94a4a2","#832db6",
    "#a96b59","#e76300","#b9ac70","#717581","#92dadd",
]
COLOR_ALIASES = {
    "CMS_blue": CMAP_10[0],
    "CMS_orange": CMAP_10[1],
    "CMS_red": CMAP_10[2],
    "CMS_gray": CMAP_10[3],
    "CMS_purple": CMAP_10[4],
    "CMS_brown": CMAP_10[5],
    "CMS_dark_orange": CMAP_10[6],
    "CMS_beige": CMAP_10[7],
    "CMS_dark_gray": CMAP_10[8],
    "CMS_light_blue": CMAP_10[9],
}
colors = {
    "ttHbb": COLOR_ALIASES["CMS_red"],
    "ttbb": COLOR_ALIASES["CMS_dark_orange"],
    "ttcc": COLOR_ALIASES["CMS_orange"],
    "ttlf": COLOR_ALIASES["CMS_blue"],
    "ttV": COLOR_ALIASES["CMS_brown"],
    "singleTop": COLOR_ALIASES["CMS_purple"],
}
labels = {
    "ttHbb": 1,
    "ttbb": 2,
    "ttcc": 3,
    "ttlf": 4,
    "ttV": 5,
    "singleTop": 6
}
nice_labels = {
    "ttHbb" : "$t\\bar{t}H(b\\bar{b})$",
    "ttbb" : "$t\\bar{t}$+B",
    "ttcc" : "$t\\bar{t}$+C",
    "ttlf" : "$t\\bar{t}$+LF",
}

# -------------------------
# Helper functions
# -------------------------
def plot_score(pred_label, score, label, ax, weights=None, title=None, mask=None, log=False, ylim=None, density=False, legend=True):
    if mask is not None:
        pred_label = pred_label[mask]
        if weights is not None:
            weights = weights[mask]
    color = colors.get(label, ax._get_lines.get_next_color())
    n, bins, patches = ax.hist(pred_label, bins=50, range=(0,1),
                               weights=weights, histtype="step",
                               linewidth=2, color=color, label=label, density=density)
    if legend:
        ax.legend(fontsize=15)
    ax.set_xlabel(f"{score} SPANet score")
    ax.set_ylabel("A.U." if density else "Counts")
    if log:
        ax.set_yscale("log")
    if ylim is not None:
        ax.set_ylim(*ylim)
    if title:
        ax.set_title(title)
    return n, bins, patches

def plot_scores(true_labels, pred_labels, plot_dir=None, weights=None, title=None, mask=None, log=False, ylim=None, density=False):
    fig, axes = plt.subplots(2, 3, figsize=[20,14], dpi=100)
    for i, score in enumerate(labels):
        ax = axes[i // 3][i % 3]
        for label in labels:
            mask_by_sample = true_labels[label]
            if mask is not None:
                mask_by_sample &= mask
            plot_score(pred_labels[score], score, label, ax=ax,
                       mask=mask_by_sample, weights=weights,
                       title=title, log=log, ylim=ylim, density=density)
    fig.tight_layout(pad=1.5)
    if plot_dir:
        filename = os.path.join(plot_dir, "classification_scores.png")
        if weights is not None:
            filename = filename.replace(".png","_weighted.png")
        if log:
            filename = filename.replace(".png","_log.png")
        if density:
            filename = filename.replace(".png","_normalised.png")
        plt.savefig(filename, dpi=300)
    plt.close(fig)

def plot_confusion_matrix(y_true, y_pred, plot_dir=None, norm=None, class_names=None, title=None, cmap='Blues', weights=None):
    #cm = confusion_matrix(y_true, y_pred, normalize=norm, sample_weight=weights)[1:,1:]
    #labels_ = class_names if class_names is not None else np.arange(cm.shape[0])
    label_ids   = [1, 2, 3, 4, 5, 6]
    label_names = ["ttHbb", "ttbb", "ttcc", "ttlf", "ttV", "singleTop"]
    cm = confusion_matrix(y_true, y_pred, labels=label_ids, normalize=norm, sample_weight=weights)
    plt.figure(figsize=(12,12))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap=cmap, xticklabels=label_names, yticklabels=label_names,)
    #sns.heatmap(cm, annot=True, cmap=cmap, xticklabels=labels_, yticklabels=labels_, fmt=".2f")
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    if plot_dir:
        fname = "confusion_matrix.png"
        if norm=="true":
            fname = "confusion_matrix_norm_true.png"
        elif norm=="pred":
            fname = "confusion_matrix_norm_pred.png"
        plt.savefig(os.path.join(plot_dir, fname), dpi=100)
    plt.close()

def plot_roc(true_label, pred_label, ax, label, weight=None):
    """
    Plots a ROC curve and calculates the AUC using roc_auc_score.
    """
    # Compute ROC curve
    fpr, tpr, _ = roc_curve(true_label, pred_label, sample_weight=weight)
    
    # Compute AUC safely
    try:
        AUC = roc_auc_score(true_label, pred_label, sample_weight=weight)
    except ValueError:
        AUC = float('nan')  # in case all labels are the same, avoid crashing

    # Plot ROC curve
    ax.plot(fpr, tpr, label=f"{label} AUC={round(AUC, 3)}")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{label} ROC Curve")
    ax.legend()
    ax.grid(True)

def plot_rocs(true_labels, pred_labels, weight=None, plot_dir=None):
    """
    Plots ROC curves for all classes in the labels dictionary.
    """
    fig, axes = plt.subplots(2, 3, figsize=[18,12])
    axes = axes.flatten()
    
    for i, (score, label_index) in enumerate(labels.items()):
        ax = axes[i]
        true_label = true_labels[score]
        pred_label = pred_labels[score]
        plot_roc(true_label, pred_label, ax, score + " weighed", weight=weight)

    fig.tight_layout(pad=2.5)
    if plot_dir is not None:
        filename = os.path.join(plot_dir, "roc_curves.png")
        plt.savefig(filename, dpi=300)

def plot_correlation_matrix(numpy_array, plot_dir, feature_names=None, title="Correlation Matrix", annot=True, cmap="coolwarm"):
    if feature_names is None:
        feature_names = [f"Feature {i+1}" for i in range(numpy_array.shape[1]-1)]
    df = pd.DataFrame(numpy_array[:,1:], columns=feature_names)
    corr_matrix = df.corr()
    plt.figure(figsize=(12,12))
    sns.heatmap(corr_matrix, annot=annot, cmap=cmap, xticklabels=feature_names, yticklabels=feature_names)
    plt.title(title)
    if plot_dir:
        plt.savefig(os.path.join(plot_dir,"correlation_matrix.png"), dpi=100)
    plt.close()

def plot_cumulative_significance(true_scores, scores, weights, signal_index=None, bins=30, range=(0,1), xlabel="Threshold", ylabel="$S/\\sqrt{B}$", plot_dir=None):
    if signal_index is None: signal_index = 1
    thresholds = np.linspace(0,1,100)
    true_signal = (true_scores==signal_index)
    true_bkg = ~true_signal
    pred_sig_scores = scores[:,signal_index]
    pred_scores = np.argmax(scores, axis=1)
    is_signal_pred = (pred_scores==signal_index)
    score_pred_signal = scores[is_signal_pred, signal_index]
    true_scores_selected = true_scores[is_signal_pred]
    weights_selected = weights[is_signal_pred]
    cum_sig_pred = np.zeros_like(thresholds)
    cum_sig_true = np.zeros_like(thresholds)
    for i, thr in enumerate(thresholds):
        pass_signal = score_pred_signal>thr
        pass_true_signal = (pred_sig_scores>thr)&true_signal
        pass_true_bkg = (pred_sig_scores>thr)&true_bkg
        S_true = np.sum(weights[pass_true_signal])
        B_true = np.sum(weights[pass_true_bkg])
        cum_sig_true[i] = S_true/np.sqrt(B_true) if B_true>0 else 0
        S_pred = np.sum(weights_selected[pass_signal & (true_scores_selected==1)])
        B_pred = np.sum(weights_selected[pass_signal & (true_scores_selected!=1)])
        cum_sig_pred[i] = S_pred/np.sqrt(B_pred) if B_pred>0 else 0
    fig, ax = plt.subplots(figsize=[10,8])
    ax.plot(thresholds, cum_sig_pred, marker='o', color=CMAP_10[0], linestyle='-', label="Predicted as ttHbb")
    ax.plot(thresholds, cum_sig_true, marker='o', color=CMAP_10[1], linestyle='-', label="ttHbb node")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("Cumulative Significance")
    ax.legend(fontsize=15)
    if plot_dir:
        plt.savefig(os.path.join(plot_dir,"cumulative_significance.png"), dpi=300)
    plt.close()

def compute_s_over_sqrtb(y_true, scores, weights, signal_index=None, bins=30, range=(0,1)):
    if signal_index is None: signal_index=1
    if isinstance(bins,int):
        bins = np.linspace(range[0],range[1],bins+1)
    bin_centers = 0.5*(bins[:-1]+bins[1:])
    s_over_sqrtb = np.zeros(len(bin_centers))
    for i,(low,high) in enumerate(zip(bins[:-1], bins[1:])):
        in_bin = (scores>=low)&(scores<high)
        S = np.sum(weights[in_bin & (y_true==signal_index)])
        B = np.sum(weights[in_bin & (y_true!=signal_index)])
        s_over_sqrtb[i] = S/np.sqrt(B) if B>0 else 0
    return s_over_sqrtb

# -------------------------
# ONNX inference worker
# -------------------------
def run_chunk(args):
    onnx_path, inputs, masks, start, end, batch_size, sess_opts_dict, providers = args
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = sess_opts_dict["intra"]
    sess_options.inter_op_num_threads = sess_opts_dict["inter"]
    sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(onnx_path, sess_options=sess_options, providers=providers)

    jet_input, met_input, lepton_input, event_input = inputs
    jet_mask, met_mask, lepton_mask, event_mask = masks
    out = np.empty((end-start,7), dtype=np.float32)
    offset=0
    for i in range(start,end,batch_size):
        j=min(i+batch_size,end)
        out[offset:offset+(j-i)] = session.run(
            ["EVENT/signal"],
            {
                "Jet_data": jet_input[i:j],
                "Jet_mask": jet_mask[i:j],
                "Met_data": met_input[i:j],
                "Met_mask": met_mask[i:j],
                "Lepton_data": lepton_input[i:j],
                "Lepton_mask": lepton_mask[i:j],
                "Event_data": event_input[i:j],
                "Event_mask": event_mask[i:j],
            }
        )[0]
        offset += (j-i)
    return start, out

# -------------------------
# Main function
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="""
    Parallel ONNX inference for SPANet-style ttH analyses.

    This script:
    • Loads inputs from an HDF5 ntuple
    • Runs batched ONNX inference
    • Splits events across multiple CPU processes
    • Writes the output scores to a NumPy file

    IMPORTANT:
    • Use --workers > 1 ONLY for CPU inference
    • For GPU inference, always use --workers 1 and increase --batch-size
    """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""
    Examples
    --------
    CPU inference on 16 cores:
    python run_inference.py \\
        --h5 input.h5 \\
        --config event.yaml \\
        --onnx model.onnx \\
        --out scores.npy \\
        --workers 16

    GPU inference:
    python run_inference.py \\
        --h5 input.h5 \\
        --config event.yaml \\
        --onnx model.onnx \\
        --batch-size 8192 \\
        --workers 1
    """,
    )
    parser.add_argument("--h5", required=True, help="Input HDF5 file")
    parser.add_argument("--config", required=True, help="YAML configuration file")
    parser.add_argument("--onnx", required=True, help="ONNX model file")
    parser.add_argument("--out-folder", required=True, help="Output folder to save scores and figures")
    parser.add_argument("--batch-size", type=int, default=2048, help="Batch size for inference")
    parser.add_argument("--workers", type=int, default=cpu_count(), help="Number of parallel CPU workers")
    parser.add_argument("--cpu-only", action="store_true", help="Force CPU only")
    parser.add_argument("--save-scores", action="store_true", help="Save output scores to .npy file")
    parser.add_argument("--scores-npy", help="Use existing scores.npy instead of running inference")
    args = parser.parse_args()

    os.makedirs(args.out_folder, exist_ok=True)

    if args.scores_npy:
        # -------------------------
        # Load intermediate scores
        # -------------------------
        cl_out = np.load(args.scores_npy)
        print(f"Loaded intermediate scores from {args.scores_npy}")
        # We still need the true labels and weights, so HDF5 file is still required
        f = h5py.File(args.h5, "r")
        weights = f["WEIGHTS"]["weight"][()]
        y_true = f["CLASSIFICATIONS"]["EVENT"]["signal"][()]

    else:
        # -------------------------
        # Load inputs from HDF5
        # -------------------------
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        f = h5py.File(args.h5,"r")
        jet_data = f["INPUTS"]["Jet"]
        met_data = f["INPUTS"]["Met"]
        lepton_data = f["INPUTS"]["Lepton"]
        event_data = f["INPUTS"]["Event"]
        weights = f["WEIGHTS"]["weight"][()]
        jet_inputs = list(cfg["INPUTS"]["SEQUENTIAL"]["Jet"].keys())
        met_inputs = cfg["INPUTS"]["GLOBAL"]["Met"]
        lepton_inputs = cfg["INPUTS"]["GLOBAL"]["Lepton"]
        event_inputs = cfg["INPUTS"]["GLOBAL"]["Event"]
        jet_input = np.stack([jet_data[j][()] for j in jet_inputs], axis=2)
        met_input = np.stack([met_data[j][()] for j in met_inputs], axis=1)[:,None,:]
        lepton_input = np.stack([lepton_data[j][()] for j in lepton_inputs], axis=1)[:,None,:]
        event_input = np.stack([event_data[j][()] for j in event_inputs], axis=1)[:,None,:]
        jet_input[:,:,0] = np.log1p(jet_input[:,:,0])
        met_input[:,:,0] = np.log1p(met_input[:,:,0])
        lepton_input[:,:,0] = np.log1p(lepton_input[:,:,0])
        jet_mask = jet_data["MASK"][()]
        met_mask = np.ones((met_input.shape[0],1), dtype=bool)
        lepton_mask = met_mask
        event_mask = met_mask
        Nevents = jet_input.shape[0]

        # -------------------------
        # Parallel inference
        # -------------------------
        print(f"Running inference on {Nevents} events using {args.workers} workers...")
        print("it may take a while depending on the number of events and batch size.")
        chunk_size = (Nevents + args.workers-1)//args.workers
        providers = ["CPUExecutionProvider"] if args.cpu_only else ["CUDAExecutionProvider","CPUExecutionProvider"]
        sess_opts = {"intra":1, "inter":1}
        jobs=[]
        for w in range(args.workers):
            start = w*chunk_size
            end = min(start+chunk_size, Nevents)
            if start>=end: continue
            jobs.append((args.onnx, (jet_input, met_input, lepton_input, event_input),
                        (jet_mask, met_mask, lepton_mask, event_mask),
                        start,end,args.batch_size,sess_opts,providers))
        cl_out = np.empty((Nevents,7), dtype=np.float32)
        with Pool(processes=args.workers) as pool:
            for start,out_chunk in pool.imap_unordered(run_chunk,jobs):
                cl_out[start:start+out_chunk.shape[0]] = out_chunk
        print("Inference completed.")  # no intermediate scores are saved
        # Save output scores
        if args.save_scores:
            out_path = os.path.join(args.out_folder, "spanet_onnx_scores.npy")
            np.save(out_path, cl_out)
            print(f"Output scores saved to {out_path}.")
    # -------------------------
    # Compute labels
    # -------------------------
    y_true = f["CLASSIFICATIONS"]["EVENT"]["signal"][()]
    y_pred = np.argmax(cl_out,axis=1)
    true_labels, pred_labels = {},{}
    for label, index in labels.items():
        true_labels[label] = (y_true==index)
        pred_labels[label] = cl_out[:,index]

    # -------------------------
    # Plots
    # -------------------------
    print("Generating evaluation plots...")
    print(" - Classification scores")
    plot_scores(true_labels, pred_labels, plot_dir=args.out_folder, weights=weights, density=True)
    plot_scores(true_labels, pred_labels, plot_dir=args.out_folder, weights=weights, log=True)
    print(" - ROC curves")
    plot_rocs(true_labels, pred_labels, weight=weights, plot_dir=args.out_folder)
    print(" - Confusion matrices")
    plot_confusion_matrix(y_true, y_pred, plot_dir=args.out_folder, norm="true", class_names=labels, weights=weights)
    plot_confusion_matrix(y_true, y_pred, plot_dir=args.out_folder, norm="pred", class_names=labels, weights=weights)
    print(" - Correlation matrices and cumulative significance")
    for label, class_index in labels.items():
        mask = y_true==class_index
        plot_correlation_matrix(cl_out[mask], plot_dir=args.out_folder, feature_names=list(labels.keys()), title=f"{label} Correlation Matrix")
    plot_cumulative_significance(y_true, cl_out, weights, plot_dir=args.out_folder)
    print(f"All plots saved in {args.out_folder}.")

if __name__=="__main__":
    main()
