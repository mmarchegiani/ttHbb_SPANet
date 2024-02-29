# Generates a table of purities for ttH. Creates plots of purity 
# against the number of jets in the event for each event particle.
# 2023-10-26 22:34:03
# Yuliia Maidannyk yuliia.maidannyk@ethz.ch

import h5py
import awkward as ak
import pandas as pd
import vector
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import mplhep as hep
hep.style.use(hep.style.ROOT)
vector.register_awkward()

from utils.kinematics import *
from utils.functions import *

hep.style.use(hep.style.ROOT)
vector.register_awkward()
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
mpl.rcParams['grid.alpha'] = 0.2
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.size'] = 24

# Directory to save the plots
figdir = "/eos/user/y/ymaidann/eth_project/Spanet_project/plots/"

# Load files (true .h5, prediction .h5, and jets .parquet)
df_pred = h5py.File('../predictions/0107_output_v7_matched.h5','r')
df_true = h5py.File('../data/tth_matched_3.h5','r')
df_jets = ak.from_parquet("/eos/user/d/dvalsecc/www/ttHbbAnalysis/training_dataset/all_jets_v6.parquet")
(jets,_,_,_,_,lep,met) = ak.unzip(df_jets)
jets = ak.with_name(jets, name="Momentum4D")

# Get fully matched jets
higgs = jets[jets.prov == 1]
mask_match = ak.num(higgs) == 2
w_or_t_jets = jets[(jets.prov == 5)|(jets.prov == 2)]
mask_match = mask_match & (ak.num(w_or_t_jets) == 3)
lep_top = jets[jets.prov == 3]
mask_match = mask_match & (ak.num(lep_top) == 1)
jets = jets[mask_match]

# ~~~~~~~~~~~~~~~~~~~~~~~~ LEPTONIC TOP ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("\n--Leptonic top--\n")
nmin = ak.min(ak.num(jets))
nmax = ak.max(ak.num(jets))
# Calculate purities by the number of jets in the event for each file
purities, df = get_purities_by_njets("t2", df_true, df_pred, jets, nmin, nmax)
leptop_purities = purities
print(df)

n = np.arange(nmin, nmax+1, 1)
fig, ax = plt.subplots(1, 1)

plt.plot(n, purities[:-1], marker='o', lw=2.5)
plt.axhline(purities[-1], label="inclusive efficiency", c='r', ls="--", lw=2.5)
plt.xlabel("Number of jets in the event", fontsize=20, labelpad=10)
plt.ylabel("Efficiency", fontsize=20, labelpad=10)
plt.title("Leptonic top", fontsize=24, pad=15)
plt.ylim(-0.05,1.05)

ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(2))
ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))

plt.legend(fontsize=18, loc="upper left")
plt.grid(alpha=0.2, zorder=0)
plt.rcParams['figure.facecolor'] = 'white'
name = figdir+"t2_eff_vs_njets_v7.png"
plt.savefig(f"{name}", transparent=False, dpi=300, bbox_inches='tight')
print(f"\nFigure saved as : {name}")

# ~~~~~~~~~~~~~~~~~~~~~~~~ HADRONIC TOP ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("\n--Hadronic top--\n")
# Calculate purities by the number of jets in the event for each file
purities, df = get_purities_by_njets("t1", df_true, df_pred, jets, nmin, nmax)
hadtop_purities = purities
print(df)

n = np.arange(nmin, nmax+1, 1)
fig, ax = plt.subplots(1, 1)

plt.plot(n, purities[:-1], marker='o', lw=2.5)
plt.axhline(purities[-1], label="inclusive efficiency", c='r', ls="--", lw=2.5)
plt.xlabel("Number of jets in the event", fontsize=20, labelpad=10)
plt.ylabel("Efficiency", fontsize=20, labelpad=10)
plt.title("Hadronic top", fontsize=24, pad=15)
plt.ylim(-0.05,1.05)

ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(2))
ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))

plt.legend(fontsize=18, loc="upper left")
plt.grid(alpha=0.2, zorder=0)
plt.rcParams['figure.facecolor'] = 'white'
name = figdir+"t1_eff_vs_njets_v7.png"
plt.savefig(f"{name}", transparent=False, dpi=300, bbox_inches='tight')
print(f"\nFigure saved as : {name}")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ HIGGS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("\n--Higgs--\n")
# Calculate purities by the number of jets in the event
purities, df = get_purities_by_njets("h", df_true, df_pred, jets, nmin, nmax)
higgs_purities = purities
print(df)

n = np.arange(nmin, nmax+1, 1)
fig, ax = plt.subplots(1, 1)

plt.plot(n, purities[:-1], marker='o', lw=2.5)
plt.axhline(purities[-1], label="inclusive efficiency", c='r', ls="--", lw=2.5)
plt.xlabel("Number of jets in the event", fontsize=20, labelpad=10)
plt.ylabel("Efficiency", fontsize=20, labelpad=10)
plt.title("Higgs", fontsize=24, pad=15)
plt.ylim(-0.05,1.05)

ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(2))
ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))

plt.legend(fontsize=18, loc="upper left")
plt.grid(alpha=0.2, zorder=0)
plt.rcParams['figure.facecolor'] = 'white'
name = figdir+"h_eff_vs_njets_v7.png"
plt.savefig(f"{name}", transparent=False, dpi=300, bbox_inches='tight')
print(f"\nFigure saved as : {name}")
