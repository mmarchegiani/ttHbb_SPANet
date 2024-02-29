# Generates a table of purities for ttH background files. Creates plots of purity 
# against the number of jets in the event for each event particle.
# 2023-10-26 22:20:20
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

# Directory to save the plots
figdir = "/eos/user/y/ymaidann/eth_project/Spanet_project/plots/"

# Load files (true .h5, prediction .h5, and jets .parquet)
bkg1 = "TTbbSemiLeptonic_Powheg_2016_PostVFP_v9"
bkg1_true, bkg1_pred, bkg1_jets, _, _  = initialise_sig(bkg1)
bkg2 = "TTbbSemiLeptonic_Powheg_2016_PreVFP_v9"
bkg2_true, bkg2_pred, bkg2_jets, _, _ = initialise_sig(bkg2)
bkg3 = "TTbbSemiLeptonic_Powheg_2017_v9"
bkg3_true, bkg3_pred, bkg3_jets, _, _ = initialise_sig(bkg3)
bkg4 = "TTbbSemiLeptonic_Powheg_2018_v9"
bkg4_true, bkg4_pred, bkg4_jets, _, _ = initialise_sig(bkg4)

# ~~~~~~~~~~~~~~~~~~~~~~~~ LEPTONIC TOP ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
nmin = ak.min(ak.num(bkg1_jets))
leptop_purities = []

# Calculate purities by the number of jets in the event for each file

print(f"\n--Leptonic top {bkg1}--\n")
nmax1 = ak.max(ak.num(bkg1_jets))
purities, df = get_purities_by_njets("t2", bkg1_true, bkg1_pred, bkg1_jets, nmin, nmax1)
leptop_purities.append(purities)
print(df)

print(f"\n--Leptonic top {bkg2}--\n")
nmax2 = ak.max(ak.num(bkg2_jets))
purities, df = get_purities_by_njets("t2", bkg2_true, bkg2_pred, bkg2_jets, nmin, nmax2)
leptop_purities.append(purities)
print(df)

print(f"\n--Leptonic top {bkg3}--\n")
nmax3 = ak.max(ak.num(bkg3_jets))
purities, df = get_purities_by_njets("t2", bkg3_true, bkg3_pred, bkg3_jets, nmin, nmax3)
leptop_purities.append(purities)
print(df)

print(f"\n--Leptonic top {bkg4}--\n")
nmax4 = ak.max(ak.num(bkg4_jets))
purities, df = get_purities_by_njets("t2", bkg4_true, bkg4_pred, bkg4_jets, nmin, nmax4)
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
name = figdir+"t2_eff_vs_njets_bkg_split_v9.png"
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
name = figdir+"t2_eff_vs_njets_bkg_v9.png"
plt.savefig(f"{name}", transparent=False, dpi=300, bbox_inches='tight')
print(f"\nFigure saved as : {name}")

# ~~~~~~~~~~~~~~~~~~~~~~~~ HADRONIC TOP ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
nmin = ak.min(ak.num(bkg1_jets))
hadtop_purities = []

# Calculate purities by the number of jets in the event for each file

print(f"\n--Hadronic top {bkg1}--\n")
nmax1 = ak.max(ak.num(bkg1_jets))
purities, df = get_purities_by_njets("t1", bkg1_true, bkg1_pred, bkg1_jets, nmin, nmax1)
hadtop_purities.append(purities)
print(df)

print(f"\n--Hadronic top {bkg2}--\n")
nmax2 = ak.max(ak.num(bkg2_jets))
purities, df = get_purities_by_njets("t1", bkg2_true, bkg2_pred, bkg2_jets, nmin, nmax2)
hadtop_purities.append(purities)
print(df)

print(f"\n--Hadronic top {bkg3}--\n")
nmax3 = ak.max(ak.num(bkg3_jets))
purities, df = get_purities_by_njets("t1", bkg3_true, bkg3_pred, bkg3_jets, nmin, nmax3)
hadtop_purities.append(purities)
print(df)

print(f"\n--Hadronic top {bkg4}--\n")
nmax4 = ak.max(ak.num(bkg4_jets))
purities, df = get_purities_by_njets("t1", bkg4_true, bkg4_pred, bkg4_jets, nmin, nmax4)
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
name = figdir+"t1_eff_vs_njets_bkg_split_v9.png"
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
print("hadtop mean ", mean)
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
name = figdir+"t1_eff_vs_njets_bkg_v9.png"
plt.savefig(f"{name}", transparent=False, dpi=300, bbox_inches='tight')
print(f"\nFigure saved as : {name}")
