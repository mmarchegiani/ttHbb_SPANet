"""Cut-based analysis regions built from multiclassifier scores.

Each region is a boolean event mask plus a ``target`` class against which its
purity and efficiency are measured.
"""

from collections import OrderedDict

import numpy as np

# Default thresholds (from the original notebook).
DEFAULT_THRESHOLDS = {
    "ttlf_low": 0.2,
    "ttlf_high": 0.3,
    "tthbb_low": 0.6,
    "tthbb_high": 0.8,
    "qcd_high": 0.001,
}


def default_cut_regions(score_dict, thresholds=None):
    """Return the default cut-based regions as ``OrderedDict``.

    Keys are region names; values are ``{"mask": bool_array, "target": class}``.
    ``score_dict`` is the per-node/derived score dict from
    :func:`tthbb_spanet.lib.classification.scores.build_score_dict`.
    """
    t = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        t.update(thresholds)

    qcd = score_dict["QCD"]
    ttlf = score_dict["ttLF"]
    tthbb = score_dict["ttH(bb)"]

    qcd_veto = qcd < t["qcd_high"]
    sr_tthbb = qcd_veto & (ttlf < t["ttlf_low"]) & (tthbb > t["tthbb_high"])
    cr_ttlf = qcd_veto & (ttlf > t["ttlf_high"]) & (tthbb < t["tthbb_low"])

    regions = OrderedDict()
    regions["inclusive"] = {"mask": np.ones_like(qcd, dtype=bool), "target": "ttH(bb)"}
    regions["QCD veto"] = {"mask": qcd_veto, "target": "ttH(bb)"}
    regions["SR ttHbb"] = {"mask": sr_tthbb, "target": "ttH(bb)"}
    regions["CR ttlf"] = {"mask": cr_ttlf, "target": "ttLF"}
    return regions
