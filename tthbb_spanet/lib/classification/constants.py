"""Shared constants for the fully-hadronic SPANet multiclassifier.

Single source of truth for the class encoding, plotting colors and legend
labels that were previously duplicated across notebook cells.

The multiclassifier ``EVENT/signal`` output has 10 columns; index 0 is unused
(empty class) so the physics classes live at indices 1..9.
"""

# Class index (in the ONNX ``EVENT/signal`` output) -> human readable name.
# Index 0 is the unused/empty class and is intentionally absent.
MAPPING_ENCODING = {
    1: "ttH(bb)",
    2: "QCD",
    3: "ttLF",
    4: "ttC",
    5: "ttB",
    6: "single top",
    7: "ttV",
    8: "V+jets",
    9: "ttH(non-bb)",
}

# Ordered list of physics class names (indices 1..9).
CLASS_NAMES = [MAPPING_ENCODING[i] for i in sorted(MAPPING_ENCODING)]

# Names of the derived (Neyman-Pearson style) discriminants built on top of the
# raw multiclassifier nodes. Keys are the field names used in the parquet output.
DERIVED_SCORES = ["tthbb_vs_ttbb", "tthbb_vs_qcd_ttbb", "tthbb_vs_qcd_ttbar"]

# Mapping from class name -> field name used inside the parquet ``spanet_output``
# record. Kept explicit so the parquet schema is stable regardless of the
# display names above.
SPANET_OUTPUT_FIELDS = {
    "ttH(bb)": "tthbb",
    "QCD": "qcd",
    "ttLF": "ttlf",
    "ttC": "ttcc",
    "ttB": "ttbb",
    "single top": "singletop",
    "ttV": "ttv",
    "V+jets": "vjets",
    "ttH(non-bb)": "tthnonbb",
}

# CMS 10-color palette.
CMAP_10 = [
    "#3f90da",
    "#ffa90e",
    "#bd1f01",
    "#94a4a2",
    "#832db6",
    "#a96b59",
    "#e76300",
    "#b9ac70",
    "#717581",
    "#92dadd",
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

# Per-class line/fill colors used in the score and region plots.
colors = {
    "ttH(bb)": COLOR_ALIASES["CMS_red"],
    "QCD": COLOR_ALIASES["CMS_purple"],
    "ttB": COLOR_ALIASES["CMS_dark_orange"],
    "ttC": COLOR_ALIASES["CMS_orange"],
    "ttLF": COLOR_ALIASES["CMS_blue"],
    "single top": COLOR_ALIASES["CMS_gray"],
    "ttV": COLOR_ALIASES["CMS_light_blue"],
    "V+jets": COLOR_ALIASES["CMS_brown"],
    "VV": COLOR_ALIASES["CMS_dark_gray"],
    "ttH(non-bb)": COLOR_ALIASES["CMS_beige"],
}

# Legend labels. Class names default to themselves; the derived discriminants
# get nicer math-y labels.
labels_legend = {name: name for name in MAPPING_ENCODING.values()}
labels_legend.update(
    {
        "tthbb_vs_ttbb": "ttHbb vs ttB",
        "tthbb_vs_qcd_ttbb": "ttHbb vs ttB+QCD",
        "tthbb_vs_qcd_ttbar": "ttHbb vs QCD+ttbar",
    }
)
