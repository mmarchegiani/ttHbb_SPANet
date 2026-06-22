#!/usr/bin/env python3
# coding: utf-8
"""Standalone SPANet classification inference + evaluation pipeline.

Takes an H5 dataset and the trained fully-hadronic multiclassifier (ONNX) and:

  * runs ONNX inference -- either locally or by launching GPU jobs on HTCondor,
    splitting the dataset into N slices and merging the per-slice scores;
  * builds the derived discriminants tthbb_vs_ttbb, tthbb_vs_qcd_ttbb,
    tthbb_vs_qcd_ttbar;
  * writes a parquet with fields Jet / Met / Event / spanet_output;
  * produces the full set of evaluation plots (scores, ROC, njet/nbjet,
    confusion matrices, argmax & cut-based purity/efficiency).

Sub-commands
------------
  submit   split the H5 and submit one GPU condor job per slice (scores .npy)
  infer    run inference on a single [start, stop) slice (used by the jobs)
  report   merge per-slice scores, write parquet, make every plot
  local    run inference over the whole file in-process, then report

Examples
--------
  # 1) submit 20 GPU jobs, then (after they finish) build everything:
  python apply_spanet_classification.py submit \\
      --h5 in.h5 --config event.yaml --onnx spanet.onnx \\
      --out-folder OUT --n-jobs 20 --good-gpus
  python apply_spanet_classification.py report \\
      --h5 in.h5 --config event.yaml --onnx spanet.onnx --out-folder OUT

  # all-in-one on a local GPU (or CPU for a quick test with --entrystop):
  python apply_spanet_classification.py local \\
      --h5 in.h5 --config event.yaml --onnx spanet.onnx \\
      --out-folder OUT --entrystop 5000 --cpu-only
"""

import argparse
import json
import os
import subprocess

import numpy as np

from tthbb_spanet.lib import classification as cls
from tthbb_spanet.lib.classification.constants import (
    DERIVED_SCORES,
    SPANET_OUTPUT_FIELDS,
)

# cmsml singularity image used by the existing condor jobs.
DEFAULT_SINGULARITY_IMAGE = (
    "/cvmfs/unpacked.cern.ch/registry.hub.docker.com/cmsml/cmsml:latest"
)


# -------------------------------------------------------------------------
# slice bookkeeping
# -------------------------------------------------------------------------
def compute_slices(n_total, n_jobs):
    """Split ``n_total`` events into at most ``n_jobs`` contiguous slices."""
    edges = np.linspace(0, n_total, n_jobs + 1).astype(int)
    return [(int(edges[i]), int(edges[i + 1]))
            for i in range(n_jobs) if edges[i] < edges[i + 1]]


def slice_npy_name(start, stop):
    return f"scores_{start}_{stop}.npy"


def scores_dir(out_folder):
    return os.path.join(out_folder, "scores")


def manifest_path(out_folder):
    return os.path.join(scores_dir(out_folder), "manifest.json")


# -------------------------------------------------------------------------
# infer (single slice, runs on the worker)
# -------------------------------------------------------------------------
def cmd_infer(args):
    cfg = cls.load_event_config(args.config)
    sl = cls.load_h5_slice(args.h5, cfg, start=args.start, stop=args.stop)
    feed = cls.build_onnx_feed(sl, cfg)
    providers = cls.resolve_providers(cpu_only=args.cpu_only)
    print(f"[infer] events [{sl['start']}, {sl['stop']}) of {sl['n_total']} "
          f"| providers={providers} | batch={args.batch_size}")
    cl_out = cls.run_onnx_inference(
        args.onnx, feed, batch_size=args.batch_size, providers=providers
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out_npy)), exist_ok=True)
    np.save(args.out_npy, cl_out)
    print(f"[infer] saved scores {cl_out.shape} -> {args.out_npy}")


# -------------------------------------------------------------------------
# submit (split into GPU condor jobs)
# -------------------------------------------------------------------------
def cmd_submit(args):
    import htcondor  # lazy: only available on the submit host (lxplus)

    n_total = cls.n_events(args.h5)
    slices = compute_slices(n_total, args.n_jobs)
    sdir = scores_dir(args.out_folder)
    log_root = os.path.join(args.out_folder, "jobs")
    for sub in ("output", "error", "log"):
        os.makedirs(os.path.join(log_root, sub), exist_ok=True)
    os.makedirs(sdir, exist_ok=True)

    script = os.path.abspath(__file__)
    project_dir = os.path.abspath(
        os.path.join(os.path.dirname(script), "..", "..", "..")
    )
    job_sh = os.path.join(project_dir, "jobs", "spanet_inference.sh")

    # Manifest consumed by `report` to reassemble the slices in order.
    manifest = {
        "h5": os.path.abspath(args.h5),
        "config": os.path.abspath(args.config),
        "onnx": os.path.abspath(args.onnx),
        "n_total": n_total,
        "batch_size": args.batch_size,
        "slices": [
            {"start": s, "stop": e, "npy": os.path.join(sdir, slice_npy_name(s, e))}
            for s, e in slices
        ],
    }
    os.makedirs(sdir, exist_ok=True)
    with open(manifest_path(args.out_folder), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[submit] {len(slices)} slices over {n_total} events "
          f"-> manifest {manifest_path(args.out_folder)}")

    if args.good_gpus:
        requirements = (
            '(regexp("A100", TARGET.GPUs_DeviceName) || '
            'regexp("V100", TARGET.GPUs_DeviceName) || '
            'regexp("H100", TARGET.GPUs_DeviceName)) && '
            '!regexp("MIG", TARGET.GPUs_DeviceName)'
        )
    else:
        requirements = '!regexp("MIG", TARGET.Gpus_DeviceName)'

    col = htcondor.Collector()
    print("Adding Kerberos credentials for Condor submission...")
    krb5ccname = os.environ.get('KRB5CCNAME', f'/tmp/krb5cc_{os.getuid()}')
    if krb5ccname.startswith('FILE:'):
        krb5ccname = krb5ccname[5:]
    subprocess.run(
        ['condor_store_cred', 'add-krb', '-i', krb5ccname],
        check=True,
    )

    for start, stop in slices:
        out_npy = os.path.join(sdir, slice_npy_name(start, stop))
        tag = f"infer_{start}_{stop}"
        sub = htcondor.Submit()
        sub["Executable"] = job_sh
        sub["arguments"] = (
            f"{script} {os.path.abspath(args.h5)} {os.path.abspath(args.config)} "
            f"{os.path.abspath(args.onnx)} "
            f"{start} {stop} {out_npy} {args.batch_size}"
        )
        env_vars = f"PROJECT_DIR={project_dir}"
        if args.venv:
            env_vars += f" VENV_ACTIVATE={args.venv}"
        sub["environment"] = env_vars
        sub["Output"] = os.path.join(log_root, "output", f"{tag}-$(ClusterId).$(ProcId).out")
        sub["Error"] = os.path.join(log_root, "error", f"{tag}-$(ClusterId).$(ProcId).err")
        sub["Log"] = os.path.join(log_root, "log", f"{tag}-$(ClusterId).log")
        sub["MY.SendCredential"] = True
        sub["MY.SingularityImage"] = f'"{args.singularity_image}"'
        sub["+JobFlavour"] = f'"{args.job_flavour}"'
        sub["request_cpus"] = str(args.ncpu)
        sub["request_gpus"] = "1"
        sub["requirements"] = requirements

        if args.dry:
            print(f"\n--- DRY RUN: {tag} ---")
            print(sub)
            continue
        print("Starting Condor scheduler...")
        client = htcondor.Schedd()
        result = client.submit(sub, count=1)
        print(f"[submit] {tag}: cluster {result.cluster()} ({result.num_procs()} job)")

    if args.dry:
        print("\n[submit] dry run -- nothing submitted.")
    else:
        print(f"[submit] all jobs submitted. After they finish run:\n"
              f"  python {script} report --h5 {args.h5} --config {args.config} "
              f"--onnx {args.onnx} --out-folder {args.out_folder}")


# -------------------------------------------------------------------------
# merge per-slice scores
# -------------------------------------------------------------------------
def merge_scores(out_folder, scores_override=None):
    """Load and concatenate per-slice score arrays in slice order."""
    mpath = manifest_path(out_folder)
    sdir = scores_override or scores_dir(out_folder)
    if os.path.exists(mpath):
        with open(mpath) as fh:
            manifest = json.load(fh)
        parts = []
        for entry in manifest["slices"]:
            npy = entry["npy"]
            if scores_override:
                npy = os.path.join(scores_override, os.path.basename(npy))
            if not os.path.exists(npy):
                raise FileNotFoundError(f"missing slice scores: {npy}")
            parts.append(np.load(npy))
        cl_out = np.concatenate(parts, axis=0)
        if "n_total" in manifest and cl_out.shape[0] != manifest["n_total"]:
            print(f"[report] WARNING: merged {cl_out.shape[0]} rows != "
                  f"manifest n_total {manifest['n_total']}")
        return cl_out
    # Fallback: no manifest -> glob and sort by start index.
    files = [f for f in os.listdir(sdir)
             if f.startswith("scores_") and f.endswith(".npy")]
    if not files:
        raise FileNotFoundError(f"no per-slice scores found in {sdir}")
    files.sort(key=lambda f: int(f.split("_")[1]))
    return np.concatenate([np.load(os.path.join(sdir, f)) for f in files], axis=0)


# -------------------------------------------------------------------------
# parquet + plots
# -------------------------------------------------------------------------
def write_parquet(out_folder, cl_out, slice_dict):
    import awkward as ak

    score_dict = cls.build_score_dict(cl_out)
    jet_arrays = cls.group_feature_arrays(slice_dict, "Jet")
    met_arrays = cls.group_feature_arrays(slice_dict, "Met")
    event_arrays = cls.group_feature_arrays(slice_dict, "Event")

    spanet_output = {field: score_dict[name]
                     for name, field in SPANET_OUTPUT_FIELDS.items()}
    for d in DERIVED_SCORES:
        spanet_output[d] = score_dict[d]

    df = ak.zip(
        {
            "Jet": ak.zip({k: v for k, v in jet_arrays.items()}, depth_limit=1),
            "Met": ak.zip(dict(met_arrays)),
            "Event": ak.zip(dict(event_arrays)),
            "spanet_output": ak.zip(spanet_output),
        },
        depth_limit=1,
    )
    out_path = os.path.join(out_folder, "output.parquet")
    ak.to_parquet(df, out_path)
    print(f"[report] parquet fields={df.fields} n={len(df)} -> {out_path}")
    return out_path


def produce_outputs(out_folder, cl_out, slice_dict, make_parquet=True):
    """Write parquet (optional) and every evaluation plot for ``cl_out``."""
    os.makedirs(out_folder, exist_ok=True)
    plots_dir = os.path.join(out_folder, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    weights = slice_dict["weight"]
    y_true = slice_dict["y_true"]
    score_dict = cls.build_score_dict(cl_out)

    # merged scores for the record
    if os.path.isdir(scores_dir(out_folder)):
        np.save(os.path.join(scores_dir(out_folder), "spanet_scores.npy"), cl_out)

    if make_parquet:
        write_parquet(out_folder, cl_out, slice_dict)

    # njet / nbjet (truth-independent)
    jet_arrays = cls.group_feature_arrays(slice_dict, "Jet")
    if "pt" in jet_arrays and "btag_M" in jet_arrays:
        print("[report] njet / nbjet")
        cls.plots.plot_njet_nbjet(jet_arrays["pt"], jet_arrays["btag_M"],
                                  plots_dir, weights=weights)

    if y_true is None:
        print("[report] no truth labels in H5 -- skipping truth-based plots.")
        return

    true_labels, pred_labels = cls.build_true_pred_labels(cl_out, y_true)
    y_pred = cls.argmax_prediction(cl_out)

    print("[report] score histograms (individual + joint, linear + log)")
    cls.plots.plot_scores_individual(true_labels, pred_labels,
                                     os.path.join(plots_dir, "scores"),
                                     weights=weights)
    cls.plots.plot_scores_joint(true_labels, pred_labels,
                                os.path.join(plots_dir, "all_scores.png"),
                                weights=weights)
    cls.plots.plot_scores_joint(true_labels, pred_labels,
                                os.path.join(plots_dir, "all_scores_log.png"),
                                weights=weights, log=True)

    print("[report] ROC curves (individual + joint, linear + log)")
    cls.plots.plot_roc_individual(true_labels, pred_labels,
                                  os.path.join(plots_dir, "roc"), weights=weights)
    cls.plots.plot_rocs_joint(true_labels, pred_labels,
                              os.path.join(plots_dir, "all_roc.png"), weights=weights)
    cls.plots.plot_rocs_joint(true_labels, pred_labels,
                              os.path.join(plots_dir, "all_roc_log.png"),
                              weights=weights, log=True)

    print("[report] confusion matrices (norm true / pred)")
    cls.plots.plot_confusion_matrix(
        y_true, y_pred, os.path.join(plots_dir, "confusion_matrix_norm_true.png"),
        norm="true", weights=weights, title="Confusion matrix (normalized by row)")
    cls.plots.plot_confusion_matrix(
        y_true, y_pred, os.path.join(plots_dir, "confusion_matrix_norm_pred.png"),
        norm="pred", weights=weights, title="Confusion matrix (normalized by column)")

    print("[report] argmax purity / efficiency")
    argmax_stats = cls.compute_argmax_purity_efficiency(cl_out, y_true, weights)
    cls.plots.plot_composition_pie_charts(
        argmax_stats, os.path.join(plots_dir, "argmax_pie_charts.png"))
    cls.plots.plot_purity_efficiency_bars(
        argmax_stats, os.path.join(plots_dir, "argmax_purity_efficiency.png"),
        title="Argmax regions: purity & efficiency")

    print("[report] cut-based regions purity / efficiency")
    regions = cls.default_cut_regions(score_dict)
    region_stats = cls.compute_region_purity_efficiency(regions, true_labels, weights)
    cut_dir = os.path.join(plots_dir, "cut_regions")
    cls.plots.plot_composition_pie_charts(
        region_stats, os.path.join(cut_dir, "pie_charts.png"))
    cls.plots.plot_efficiency_1d(region_stats, os.path.join(cut_dir, "efficiency.png"))
    cls.plots.plot_purity_1d(region_stats, os.path.join(cut_dir, "purity.png"))

    # console summary
    print("\n[report] argmax purity/efficiency:")
    for name, s in argmax_stats.items():
        print(f"    {name:>12s}: purity={s['purity']:.3f} efficiency={s['efficiency']:.3f}")
    print("[report] cut-region purity/efficiency:")
    for name, s in region_stats.items():
        print(f"    {name:>12s} (target {s['target']}): "
              f"purity={s['purity']:.3f} efficiency={s['efficiency']:.3f}")


# -------------------------------------------------------------------------
# report (merge + parquet + plots)
# -------------------------------------------------------------------------
def cmd_report(args):
    cfg = cls.load_event_config(args.config)
    cl_out = merge_scores(args.out_folder, scores_override=args.scores)
    n = cl_out.shape[0]
    print(f"[report] merged scores {cl_out.shape}")
    slice_dict = cls.load_h5_slice(args.h5, cfg, start=0, stop=n)
    produce_outputs(args.out_folder, cl_out, slice_dict, make_parquet=not args.no_parquet)
    print(f"[report] done -> {args.out_folder}")


# -------------------------------------------------------------------------
# local (infer whole file in-process, then report)
# -------------------------------------------------------------------------
def cmd_local(args):
    cfg = cls.load_event_config(args.config)
    slice_dict = cls.load_h5_slice(args.h5, cfg, start=0, stop=args.entrystop)
    feed = cls.build_onnx_feed(slice_dict, cfg)
    providers = cls.resolve_providers(cpu_only=args.cpu_only)
    print(f"[local] inference on {slice_dict['stop']} events | providers={providers}")
    cl_out = cls.run_onnx_inference(
        args.onnx, feed, batch_size=args.batch_size, providers=providers
    )
    os.makedirs(scores_dir(args.out_folder), exist_ok=True)
    produce_outputs(args.out_folder, cl_out, slice_dict, make_parquet=not args.no_parquet)
    print(f"[local] done -> {args.out_folder}")


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp, onnx_required=True, out_folder_required=True):
        sp.add_argument("--h5", required=True, help="Input HDF5 file")
        sp.add_argument("--config", required=True, help="SPANet event.yaml config")
        sp.add_argument("--onnx", required=onnx_required, help="ONNX model file")
        sp.add_argument("--out-folder", required=out_folder_required, help="Output folder")
        sp.add_argument("--batch-size", type=int, default=2048)

    sp_submit = sub.add_parser("submit", help="Split H5 and submit GPU condor jobs")
    add_common(sp_submit)
    sp_submit.add_argument("--n-jobs", type=int, default=10, help="Number of slices/jobs")
    sp_submit.add_argument("--good-gpus", action="store_true",
                           help="Require A100/V100/H100 (non-MIG) GPUs")
    sp_submit.add_argument("--ncpu", type=int, default=2, help="CPUs per job")
    sp_submit.add_argument("--job-flavour", default="longlunch")
    sp_submit.add_argument("--singularity-image", default=DEFAULT_SINGULARITY_IMAGE)
    sp_submit.add_argument("--venv", default=None,
                           help="Path to a venv activate script for the worker")
    sp_submit.add_argument("--dry", action="store_true",
                           help="Print submission descriptors without submitting")
    sp_submit.set_defaults(func=cmd_submit)

    sp_infer = sub.add_parser("infer", help="Run inference on one slice (worker)")
    add_common(sp_infer, out_folder_required=False)
    sp_infer.add_argument("--start", type=int, default=0)
    sp_infer.add_argument("--stop", type=int, default=None)
    sp_infer.add_argument("--out-npy", required=True, help="Output .npy for the slice")
    sp_infer.add_argument("--cpu-only", action="store_true")
    sp_infer.set_defaults(func=cmd_infer)

    sp_report = sub.add_parser("report", help="Merge scores, write parquet, make plots")
    add_common(sp_report, onnx_required=False)
    sp_report.add_argument("--scores", default=None,
                           help="Directory of per-slice .npy (default: <out>/scores)")
    sp_report.add_argument("--no-parquet", action="store_true")
    sp_report.set_defaults(func=cmd_report)

    sp_local = sub.add_parser("local", help="Infer whole file in-process then report")
    add_common(sp_local)
    sp_local.add_argument("--entrystop", type=int, default=None,
                          help="Process only the first N events (testing)")
    sp_local.add_argument("--cpu-only", action="store_true")
    sp_local.add_argument("--no-parquet", action="store_true")
    sp_local.set_defaults(func=cmd_local)

    return p


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
