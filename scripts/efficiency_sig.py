# Generates a table of purities for ttH signal files. Creates plots of purity 
# against the number of jets in the event for each event particle.
# 2023-10-26 22:23:26
# Yuliia Maidannyk yuliia.maidannyk@ethz.ch

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

def get_matched(jets):
    """
    Get only fully matched jets
    """
    higgs = jets[jets.prov == 1]
    mask_match = ak.num(higgs) == 2

    w_or_t_jets = jets[(jets.prov == 5)|(jets.prov == 2)]
    mask_match = mask_match & (ak.num(w_or_t_jets) == 3)

    lep_top = jets[jets.prov == 3]
    mask_match = mask_match & (ak.num(lep_top) == 1)

    jets = jets[mask_match]
    return jets

# Directory to save the plots
figdir = "/eos/user/y/ymaidann/eth_project/Spanet_project/plots/"

# Load files (true .h5, prediction .h5, and jets .parquet)
sig1 = "ttHTobb_forTraining_2017_matched_v9"
sig1_true, sig1_pred, sig1_jets, _, _ = initialise_sig(sig1)
sig2 = "ttHTobb_forTraining_2018_matched_v9"
sig2_true, sig2_pred, sig2_jets, _, _ = initialise_sig(sig2)
sig3 = "ttHTobb_forTraining_2016_PreVFP_matched_v9"
sig3_true, sig3_pred, sig3_jets, _, _ = initialise_sig(sig3)
sig4 = "ttHTobb_forTraining_2016_PostVFP_matched_v9"
sig4_true, sig4_pred, sig4_jets, _, _ = initialise_sig(sig4)

# For signal, analyse only fully matched events
sig1_jets = get_matched(sig1_jets)
sig2_jets = get_matched(sig2_jets)
sig3_jets = get_matched(sig3_jets)
sig4_jets = get_matched(sig4_jets)

# ~~~~~~~~~~~~~~~~~~~~~~~~ LEPTONIC TOP ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
nmin = 6
leptop_purities = []

# Calculate purities by the number of jets in the event for each file

print(f"\n--Leptonic top {sig1}--\n")
nmax1 = ak.max(ak.num(sig1_jets))
purities, df = get_purities_by_njets("t2", sig1_true, sig1_pred, sig1_jets, nmin, nmax1)
leptop_purities.append(purities)
print(df)

print(f"\n--Leptonic top {sig2}--\n")
nmax2 = ak.max(ak.num(sig2_jets))
purities, df = get_purities_by_njets("t2", sig2_true, sig2_pred, sig2_jets, nmin, nmax2)
leptop_purities.append(purities)
print(df)

print(f"\n--Leptonic top {sig3}--\n")
nmax3 = ak.max(ak.num(sig3_jets))
purities, df = get_purities_by_njets("t2", sig3_true, sig3_pred, sig3_jets, nmin, nmax3)
leptop_purities.append(purities)
print(df)

print(f"\n--Leptonic top {sig4}--\n")
nmax4 = ak.max(ak.num(sig4_jets))
purities, df = get_purities_by_njets("t2", sig4_true, sig4_pred, sig4_jets, nmin, nmax4)
leptop_purities.append(purities)
print(df)

n = [np.arange(nmin,nmax1+1), np.arange(nmin,nmax2+1), np.arange(nmin,nmax3+1), np.arange(nmin,nmax4+1)]
fig, ax = plt.subplots(1, 1)

mean = 0
for i in range(len(n)):
    plt.plot(n[i], leptop_purities[i][:-1], marker='o', lw=2.5)
    mean += leptop_purities[i][-1]
mean /= len(n)
plt.axhline(mean, label="mean efficiency", c='r', ls="--", lw=2.5)

plt.xlabel("Number of jets in the event", fontsize=20, labelpad=10)
plt.ylabel("Efficiency", fontsize=20, labelpad=10)
plt.title("Leptonic top", fontsize=24, pad=15)
plt.ylim(-0.05,1.05)

ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(2))
ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))

plt.legend(fontsize=18, loc="upper left")
plt.grid(zorder=0)
plt.rcParams['figure.facecolor'] = 'white'
name = figdir+"t2_eff_vs_njets_sig_split_v9.png"
plt.savefig(f"{name}", transparent=False, dpi=300, bbox_inches='tight')
print(f"\nFigure saved as : {name}")

leptop_means = []
lst = leptop_purities
for i in range(14-6):
    lst2 = [item[i] for item in lst]
    leptop_means.append(np.mean(lst2))

fig, ax = plt.subplots(1, 1)

plt.plot(np.linspace(6,13,8), leptop_means, marker='o', lw=2.5)
plt.axhline(mean, label="mean efficiency", c='r', ls="--", lw=2.5)
print("leptop mean ", mean)
plt.xlabel("Number of jets in the event", fontsize=20, labelpad=10)
plt.ylabel("Efficiency", fontsize=20, labelpad=10)
plt.title("Leptonic top", fontsize=24, pad=15)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.ylim(-0.05,1.05)

ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(2))
ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))

plt.grid(zorder=0)
plt.legend(loc="upper right")
plt.rcParams['figure.facecolor'] = 'white'
name = figdir+"t2_eff_vs_njets_sig_v9.png"
plt.savefig(f"{name}", transparent=False, dpi=300, bbox_inches='tight')
print(f"\nFigure saved as : {name}")

# ~~~~~~~~~~~~~~~~~~~~~~~~ HADRONIC TOP ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
hadtop_purities = []

# Calculate purities by the number of jets in the event for each file

print(f"\n--Hadronic top {sig1}--\n")
nmax1 = ak.max(ak.num(sig1_jets))
purities, df = get_purities_by_njets("t1", sig1_true, sig1_pred, sig1_jets, nmin, nmax1)
hadtop_purities.append(purities)
print(df)

print(f"\n--Hadronic top {sig2}--\n")
nmax2 = ak.max(ak.num(sig2_jets))
purities, df = get_purities_by_njets("t1", sig2_true, sig2_pred, sig2_jets, nmin, nmax2)
hadtop_purities.append(purities)
print(df)

print(f"\n--Hadronic top {sig3}--\n")
nmax3 = ak.max(ak.num(sig3_jets))
purities, df = get_purities_by_njets("t1", sig3_true, sig3_pred, sig3_jets, nmin, nmax3)
hadtop_purities.append(purities)
print(df)

print(f"\n--Hadronic top {sig4}--\n")
nmax4 = ak.max(ak.num(sig4_jets))
purities, df = get_purities_by_njets("t1", sig4_true, sig4_pred, sig4_jets, nmin, nmax4)
hadtop_purities.append(purities)
print(df)

n = [np.arange(nmin,nmax1+1), np.arange(nmin,nmax2+1), np.arange(nmin,nmax3+1), np.arange(nmin,nmax4+1)]
fig, ax = plt.subplots(1, 1)

mean = 0
for i in range(len(n)):
    plt.plot(n[i], hadtop_purities[i][:-1], marker='o', lw=2.5)
    mean += hadtop_purities[i][-1]
mean /= len(n)
plt.axhline(mean, label="mean efficiency", c='r', ls="--", lw=2.5)

plt.xlabel("Number of jets in the event", fontsize=20, labelpad=10)
plt.ylabel("Efficiency", fontsize=20, labelpad=10)
plt.title("Hadronic top", fontsize=24, pad=15)
plt.ylim(-0.05,1.05)

ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(2))
ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))

plt.legend(fontsize=18, loc="upper left")
plt.grid(zorder=0)
plt.rcParams['figure.facecolor'] = 'white'
name = figdir+"t1_eff_vs_njets_sig_split_v9.png"
plt.savefig(f"{name}", transparent=False, dpi=300, bbox_inches='tight')
print(f"\nFigure saved as : {name}")

hadtop_means = []
lst = hadtop_purities
for i in range(14-6):
    lst2 = [item[i] for item in lst]
    hadtop_means.append(np.mean(lst2))

fig, ax = plt.subplots(1, 1)

plt.plot(np.linspace(6,13,8), hadtop_means, marker='o', lw=2.5)
plt.axhline(mean, label="mean efficiency", c='r', ls="--", lw=2.5)
print("Hadtop mean ", mean)
plt.xlabel("Number of jets in the event", fontsize=20, labelpad=10)
plt.ylabel("Efficiency", fontsize=20, labelpad=10)
plt.title("Hadronic top", fontsize=24, pad=15)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.ylim(-0.05,1.05)

ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(2))
ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))

plt.grid(zorder=0)
plt.legend(loc="upper right")
plt.rcParams['figure.facecolor'] = 'white'
name = figdir+"t1_eff_vs_njets_sig_v9.png"
plt.savefig(f"{name}", transparent=False, dpi=300, bbox_inches='tight')
print(f"\nFigure saved as : {name}")

# ~~~~~~~~~~~~~~~~~~~~~~~~ HIGGS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
higgs_purities = []

# Calculate purities by the number of jets in the event for each file

print(f"\n--Higgs {sig1}--\n")
nmax1 = ak.max(ak.num(sig1_jets))
purities, df = get_purities_by_njets("h", sig1_true, sig1_pred, sig1_jets, nmin, nmax1)
higgs_purities.append(purities)
print(df)

print(f"\n--Higgs {sig2}--\n")
nmax2 = ak.max(ak.num(sig2_jets))
purities, df = get_purities_by_njets("h", sig2_true, sig2_pred, sig2_jets, nmin, nmax2)
higgs_purities.append(purities)
print(df)

print(f"\n--Higgs {sig3}--\n")
nmax3 = ak.max(ak.num(sig3_jets))
purities, df = get_purities_by_njets("h", sig3_true, sig3_pred, sig3_jets, nmin, nmax3)
higgs_purities.append(purities)
print(df)

print(f"\n--Higgs {sig4}--\n")
nmax4 = ak.max(ak.num(sig4_jets))
purities,df = get_purities_by_njets("h", sig4_true, sig4_pred, sig4_jets, nmin, nmax4)
higgs_purities.append(purities)
print(df)

n = [np.arange(nmin,nmax1+1), np.arange(nmin,nmax2+1), np.arange(nmin,nmax3+1), np.arange(nmin,nmax4+1)]
fig, ax = plt.subplots(1, 1)

mean = 0
for i in range(len(n)):
    plt.plot(n[i], higgs_purities[i][:-1], marker='o', lw=2.5)
    mean += higgs_purities[i][-1]
mean /= len(n)
plt.axhline(mean, label="mean efficiency", c='r', ls="--", lw=2.5)

plt.xlabel("Number of jets in the event", fontsize=20, labelpad=10)
plt.ylabel("Efficiency", fontsize=20, labelpad=10)
plt.title("Higgs", fontsize=24, pad=15)
plt.ylim(-0.05,1.05)

ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(2))
ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))

plt.legend(fontsize=18, loc="upper left")
plt.grid(zorder=0)
plt.rcParams['figure.facecolor'] = 'white'
name = figdir+"h_eff_vs_njets_sig_split_v9.png"
plt.savefig(f"{name}", transparent=False, dpi=300, bbox_inches='tight')
print(f"\nFigure saved as : {name}")

higgs_means = []
lst = higgs_purities
for i in range(14-6):
    lst2 = [item[i] for item in lst]
    higgs_means.append(np.mean(lst2))

fig, ax = plt.subplots(1, 1)

plt.plot(np.linspace(6,13,8), higgs_means, marker='o', lw=2.5)
plt.axhline(mean, label="mean efficiency", c='r', ls="--", lw=2.5)
print("Higgs mean ", mean)

plt.xlabel("Number of jets in the event", fontsize=20, labelpad=10)
plt.ylabel("Efficiency", fontsize=20, labelpad=10)
plt.title("Higgs", fontsize=24, pad=15)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.ylim(-0.05,1.05)

ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(2))
ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))

plt.grid(zorder=0)
plt.legend(loc="upper right")
plt.rcParams['figure.facecolor'] = 'white'
name = figdir+"h_eff_vs_njets_sig_v9.png"
plt.savefig(f"{name}", transparent=False, dpi=300, bbox_inches='tight')
print(f"\nFigure saved as : {name}")
